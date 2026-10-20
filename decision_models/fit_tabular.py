"""Fit a tabular foundation-model head on frozen document features; development cohorts only, never test.

Selection chooses the reduced width (and fine-tuning checkpoint); calibration fits temperature and policies.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np

from protocol import breakdown, calibrate, fingerprint, prediction_rows, training_subset
from schemas import (
    BINARY_LABELS,
    BINARY_TASK,
    HERE,
    annotation,
    body_hash,
    load_split,
    save,
    source_hashes,
    task_labels,
)
from tabular import FEATURES, MODELS, FeatureReducer, TabularModel


def peak_memory(device):
    if device == "cuda":
        import torch

        return {"peak_vram_bytes": torch.cuda.max_memory_allocated()}
    try:
        import resource
    except ImportError:
        return {}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes": peak if platform.system() == "Darwin" else peak * 1024}


def raw_features(kinds, rows, split, device, details):
    inputs = {}
    if "finbert" in kinds:
        from baselines import embeddings

        details[split] = {}
        inputs["finbert"] = embeddings("finbert", rows, split, device, details[split])
    if "tfidf" in kinds:
        inputs["tfidf"] = [r["body"] for r in rows]
    return inputs


def fit(args):
    import torch

    torch.manual_seed(42)
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train = training_subset(args.splits, args.samples)
    selection, calibration = [load_split(args.splits, name) for name in ("selection", "calibration")]
    manifest = json.loads((args.splits / "manifest.json").read_text(encoding="utf-8"))
    labels = task_labels(manifest)
    if labels != BINARY_LABELS:
        raise ValueError("Use a binary benchmark")
    encoder_device = args.encoder_device or args.device
    sources = source_hashes("fit_tabular.py")
    config = {
        "family": "research_tabular",
        "task_id": BINARY_TASK,
        "model": args.model,
        "sdm_class": MODELS[args.model][0],
        "features": list(args.features),
        "labels": list(labels),
        "seed": 42,
        "samples": len(train),
        "split_manifest": manifest,
        "train_subset_sha256": hashlib.sha256(
            json.dumps([(body_hash(r), annotation(r)["label"]) for r in train]).encode()
        ).hexdigest(),
        "dimensions": list(args.dimensions),
        "context_size": args.context_size,
        "bags": args.bags,
        "num_estimators": args.num_estimators,
        "finetune": {
            "steps": args.finetune_steps,
            "every": args.finetune_every,
            "context_size": args.finetune_context,
            "query_size": args.finetune_query,
            "lr": args.lr,
        },
        "device": args.device,
        "encoder_device": encoder_device,
        "cpu_threads": 4,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "cuda_runtime": torch.version.cuda if args.device == "cuda" else None,
        "selection": "Macro F1 on selection groups; ties retain the first width / earliest fine-tuning step",
        "evaluation": "Development cohorts only; the test split is never loaded",
        "source_sha256": sources,
        "versions": {
            name: version(name)
            for name in ("structured-data-models", "torch", "scikit-learn", "numpy", "scipy", "pandas")
        },
        "python": platform.python_version(),
        "api_cost_usd": 0.0,
    }
    if "finbert" in args.features:
        from encoders import MODELS as ENCODERS

        config["encoder"], config["encoder_revision"] = ENCODERS["finbert"]
        config["versions"]["transformers"] = version("transformers")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output exists; choose a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "source"
    snapshot.mkdir(exist_ok=True)
    for name in sources:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    save(config, args.output / "config.json")

    timings, started = {}, time.perf_counter()
    preparation = {}
    raw = {
        name: raw_features(args.features, rows, name, encoder_device, preparation)
        for name, rows in (("train", train), ("selection", selection), ("calibration", calibration))
    }
    print(f"Frozen features ready in {time.perf_counter() - started:.1f}s", flush=True)
    timings["feature_seconds"] = time.perf_counter() - started
    config["feature_preparation"] = preparation
    y = np.array([annotation(r)["label"] for r in train])

    loading = time.perf_counter()
    model = TabularModel(args.model, args.device, labels)
    config["pretrained_state_sha256"] = model.pretrained_sha256
    print(f"Loaded {args.model} on {args.device}", flush=True)
    timings["model_loading_seconds"] = time.perf_counter() - loading
    options = {"context_size": args.context_size, "bags": args.bags, "num_estimators": args.num_estimators}
    contexts = model.contexts(len(train), args.context_size, args.bags, 42)
    config["inference_context"] = {
        "bags": len(contexts),
        "rows_per_bag": [len(rows) for rows in contexts],
        "unique_training_rows": len(set(np.concatenate(contexts).tolist())),
        "ordered_indices_sha256": hashlib.sha256(json.dumps([rows.tolist() for rows in contexts]).encode()).hexdigest(),
        "query_batch": 512,
    }
    save(config, args.output / "config.json")

    def score(x, sx):
        rows = prediction_rows(selection, model.log_probabilities(x, y, sx, **options), labels=labels)
        return breakdown(rows, labels=labels)["overall"]["macro_f1"]

    trials, chosen = [], None
    for dimensions in args.dimensions:
        print(f"START width {dimensions}", flush=True)
        trial_started = time.perf_counter()
        reducer = FeatureReducer(args.features, dimensions)
        x = reducer.fit_transform(raw["train"])
        sx = reducer.transform(raw["selection"])
        trial = {"dimensions": dimensions, "macro_f1": score(x, sx), "seconds": time.perf_counter() - trial_started}
        trials.append(trial)
        print(trial, flush=True)
        if chosen is None or trial["macro_f1"] > chosen[0]["macro_f1"]:
            chosen = (trial, reducer, x, sx)
        save({"trials": trials}, args.output / "search.json")
    trial, reducer, x, sx = chosen
    config["selected"] = trial
    cx = reducer.transform(raw["calibration"])

    if args.finetune_steps:
        import torch

        tuning = time.perf_counter()
        history, step = model.finetune(
            x,
            y,
            lambda: score(x, sx),
            steps=args.finetune_steps,
            every=args.finetune_every,
            context_size=args.finetune_context,
            query_size=args.finetune_query,
            lr=args.lr,
        )
        timings["finetune_seconds"] = time.perf_counter() - tuning
        config["finetune"].update(history=history, selected_step=step)
        if step:
            torch.save(model.model.state_dict(), args.output / "finetuned.pt")
            config["finetune"]["artifact_sha256"] = fingerprint(args.output / "finetuned.pt")

    joblib.dump(reducer, args.output / "reducer.joblib")
    np.savez_compressed(args.output / "context.npz", x=x, y=y)
    config.update(
        reducer_sha256=fingerprint(args.output / "reducer.joblib"),
        context_sha256=fingerprint(args.output / "context.npz"),
    )
    inference = time.perf_counter()
    selection_logits = model.log_probabilities(x, y, sx, **options)
    calibration_logits = model.log_probabilities(x, y, cx, **options)
    timings["development_inference_seconds"] = time.perf_counter() - inference
    timings["documents_per_second"] = (len(selection) + len(calibration)) / timings["development_inference_seconds"]
    decision = calibrate(calibration, calibration_logits, args.output, labels=labels)
    rows = prediction_rows(selection, selection_logits, decision, labels=labels)
    metrics = breakdown(rows, labels=labels)
    save(
        {"metrics": metrics, "predictions": rows, "evaluation": "Selection fitting data"},
        args.output / "selection.json",
    )
    timings["total_seconds"] = time.perf_counter() - started
    memory = peak_memory(args.device)
    if args.device == encoder_device == "cuda":
        memory["peak_vram_bytes"] = max(
            memory["peak_vram_bytes"], max((p.get("peak_vram_bytes", 0) for p in preparation.values()), default=0)
        )
    save(
        {**timings, **memory, "platform": platform.platform(),
         "timing_scope": "Development batch inference includes context fitting; not a warm per-document latency benchmark"},
        args.output / "performance.json",
    )
    save(config, args.output / "config.json")
    save(
        {
            "config_sha256": fingerprint(args.output / "config.json"),
            "decision_sha256": fingerprint(args.output / "decision.json"),
        },
        args.output / "complete.json",
    )
    overall = metrics["overall"]
    print(
        f"Selection macro F1 {overall['macro_f1']:.4f}; misses {overall['false_negatives']}; "
        f"skipped {overall['skip_fraction']:.1%}; temperature {decision['temperature']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="kumo-small")
    parser.add_argument("--features", choices=FEATURES, nargs="+", default=["finbert"])
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--dimensions", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--context-size", type=int, default=4096, help="Context rows per bag")
    parser.add_argument("--bags", type=int, help="Context subsets averaged; default covers every training row")
    parser.add_argument("--num-estimators", type=int, default=4, help="sdm recipe ensemble per bag")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--encoder-device", choices=("cpu", "mps", "cuda"), help="FinBERT device; default --device")
    parser.add_argument("--finetune-steps", type=int, default=0, help="0 keeps the pretrained in-context model")
    parser.add_argument("--finetune-every", type=int, default=50, help="Selection evaluation interval")
    parser.add_argument("--finetune-context", type=int, default=1024)
    parser.add_argument("--finetune-query", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if min(args.dimensions) <= 0 or args.context_size <= 0 or args.num_estimators <= 0:
        parser.error("Dimensions, context size and estimators must be positive")
    if args.finetune_steps < 0 or args.finetune_every <= 0:
        parser.error("Fine-tuning steps must be non-negative and the interval positive")
    if args.bags is not None and args.bags < 1:
        parser.error("Bags must be positive")
    if min(args.finetune_context, args.finetune_query) < 1 or args.lr <= 0:
        parser.error("Fine-tuning context, query size and learning rate must be positive")
    if args.model.startswith("kumo") and args.device == "mps":
        parser.error("KumoTabular returns constant probabilities on MPS in sdm 0.2.1; use --device cpu")
    if "cuda" in (args.device, args.encoder_device):
        from filelock import FileLock

        lock = HERE / "data/research/gpu-job.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(lock):
            fit(args)
    else:
        fit(args)
