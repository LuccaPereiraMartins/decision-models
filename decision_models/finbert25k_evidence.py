"""Export and independently replay development evidence for the fixed FinBERT extension."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import zipfile

import numpy as np

from binary_replay import (LABELS, breakdown, check, rows_from_logits, verify_fitting,
                           verify_policies, verify_temperature)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def seal(directory):
    write(directory / "package-manifest.json", {
        "scope": "Post-hoc development only; no test evaluation",
        "files": {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "package-manifest.json"}})


def telemetry_summary(samples, boundaries):
    result = {}
    for name, start, end in boundaries:
        rows = [r for r in samples if start <= datetime.fromisoformat(r["utc"]).timestamp() < end]
        values = [r for r in rows if r.get("gpu_returncode") == 0]
        gpu = np.array([[float(x) for x in r["gpu_csv"].split(",")[:6]] for r in values])
        result[name] = {"samples": len(rows), "valid_gpu_samples": len(values), "start_timestamp": start, "end_timestamp": end,
                        "gpu_util_mean_percent": float(gpu[:, 0].mean()), "gpu_util_median_percent": float(np.median(gpu[:, 0])),
                        "gpu_util_p10_percent": float(np.quantile(gpu[:, 0], .1)),
                        "gpu_at_most_5_percent_fraction": float((gpu[:, 0] <= 5).mean()),
                        "power_mean_W": float(gpu[:, 3].mean()), "temperature_max_C": float(gpu[:, 4].max()),
                        "worker_cpu_mean_percent": float(np.mean([r["worker_cpu_percent"] for r in rows]))}
    return result


def verify(directory):
    manifest = read(directory / "package-manifest.json")
    check({p.name for p in directory.iterdir() if p.is_file() and p.name != "package-manifest.json"}, set(manifest["files"]), "files")
    for name, value in manifest["files"].items():
        check(sha(directory / name), value, name)
    bundle = json.loads(gzip.decompress((directory / "evidence.json.gz").read_bytes()))
    check(set(bundle["cohorts"]), {"train", "selection", "calibration"}, "development_only")
    check({k: len(v) for k, v in bundle["cohorts"].items()}, {"train": 25000, "selection": 1000, "calibration": 500}, "counts")
    groups = [{r["group_id"] for r in rows} for rows in bundle["cohorts"].values()]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1:]), "Cross-cohort group overlap"
    train = {r["body_sha256"]: r for r in bundle["cohorts"]["train"]}
    subsets = bundle["subsets"]["subsets"]
    for n in (4000, 16000, 25000):
        check(len(subsets[str(n)]), n, "subset size")
    check(subsets["4000"], subsets["16000"][:4000], "nested4k")
    check(subsets["16000"], subsets["25000"][:16000], "nested16k")
    with zipfile.ZipFile(directory / "source.zip") as source, zipfile.ZipFile(directory / "metadata.zip") as metadata:
        check(source.read("finbert25k_evidence.py"), (directory / "verify.py").read_bytes(), "verifier_source")
        check(source.read("binary_replay.py"), (directory / "binary_replay.py").read_bytes(), "arithmetic_source")
        for size, run in bundle["runs"].items():
            config, decision = run["configuration"], run["decision"]
            for filename, value in run["metadata_sha256"].items():
                check(hashlib.sha256(metadata.read(f"{size}/{filename}")).hexdigest(), value, "metadata_member")
            check(metadata.read(f"{size}/epoch-zero-selection.json"), metadata.read("4000/epoch-zero-selection.json"), "matched_initial_predictions")
            for filename, field in (("config.json", "configuration"), ("decision.json", "decision"), ("history.json", "history"),
                                    ("selection.json", "selection"), ("calibration.json", "calibration"),
                                    ("performance.json", "performance"), ("validation.json", "raw_selection"), ("complete.json", "complete")):
                raw = metadata.read(f"{size}/{filename}")
                check(hashlib.sha256(raw).hexdigest(), run["metadata_sha256"][filename], "metadata_hash")
                check(json.loads(raw), run[field], f"{size}.{field}")
            for name, value in config["source_sha256"].items():
                check(hashlib.sha256(source.read(name)).hexdigest(), value, f"{size}.source.{name}")
            check(run["complete"]["config_sha256"], run["metadata_sha256"]["config.json"], "completion.config")
            check(run["complete"]["decision_sha256"], run["metadata_sha256"]["decision.json"], "completion.decision")
            check(run["complete"]["artifact_sha256"], run["checkpoint_audit"]["best.pt"]["sha256"], "completion.checkpoint")
            identity = [(h, train[h]["reference"]) for h in subsets[size]]
            check(hashlib.sha256(json.dumps(identity).encode()).hexdigest(), config["train_subset_sha256"], f"{size}.train_order")
            check(run["composition"], {"sources": dict(Counter(train[h]["source"] for h in subsets[size])),
                  "classes": dict(Counter(train[h]["reference"] for h in subsets[size])),
                  "characters": sum(train[h]["characters"] for h in subsets[size])}, "composition")
            verify_fitting(run)
            check(run["checkpoint_audit"]["epoch-zero.pt"]["weights_sha256"], bundle["matched_initial_weights_sha256"], "initial_state")
            for cohort in ("selection", "calibration"):
                evaluation = run[cohort]
                expected = list(bundle["cohorts"][cohort])
                random.Random(42).shuffle(expected)
                rows = evaluation["predictions"]
                check([r["body_sha256"] for r in rows], [r["body_sha256"] for r in expected], f"{size}.{cohort}.order")
                for row, identity in zip(rows, expected, strict=True):
                    check({k: row[k] for k in identity if k != "reference_basis"},
                          {k: v for k, v in identity.items() if k != "reference_basis"}, "reference_metadata")
                    check(row["discard"], {k: bool(row["prediction"] == LABELS[0] and row["probabilities"][0] >= v["threshold"])
                                          for k, v in decision["policies"].items()}, "decisions")
                check(breakdown(rows), evaluation["metrics"], f"{size}.{cohort}.arithmetic")
                if cohort == "calibration":
                    identity = [(r["body_sha256"], r["reference"]) for r in rows]
                    check(hashlib.sha256(json.dumps(identity).encode()).hexdigest(), decision["calibration_sha256"], "calibration_order")
                    verify_policies(rows, decision)
                    verify_temperature(rows, decision["temperature"])
                if size == "25000":
                    logits = bundle["independent_logits"][cohort]
                    rebuilt = rows_from_logits(rows, logits, decision)
                    for actual, saved in zip(rebuilt, rows, strict=True):
                        np.testing.assert_allclose(actual["probabilities"], saved["probabilities"], atol=1e-7, rtol=1e-6)
                        check(actual["prediction"], saved["prediction"], "independent_prediction")
                        check(actual["discard"], saved["discard"], "independent_policy")
                    check(breakdown(rows_from_logits(rows, logits)), bundle["independent_raw_metrics"][cohort], "raw_metrics")
            selected = run["raw_selection"]["predictions"]
            check([r["body_sha256"] for r in selected], [r["body_sha256"] for r in run["selection"]["predictions"]], "raw_order")
            if size == "25000":
                np.testing.assert_allclose(bundle["independent_logits"]["selection"], [r["logits"] for r in selected], rtol=1e-6, atol=1e-6)
            performance = run["performance"]
            check(performance["training_peak_vram_bytes"], run["history"]["peak_vram_bytes"], "training_peak")
            for checkpoint in run["checkpoint_audit"].values():
                check(checkpoint["parameters"], config["trainable_parameters"], "checkpoint_parameter_count")
            latencies = np.array(performance["latency_seconds"]["measurements"])
            assert np.isfinite(latencies).all() and (latencies >= 0).all()
            check(len(latencies), performance["documents"], "timing_count")
            check(len(set(performance["sample"])), performance["documents"], "timing_unique")
            check(performance["sample"], bundle["runs"]["4000"]["performance"]["sample"], "matched_timing_sample")
            for key, value in (("mean", latencies.mean()), ("p50", np.quantile(latencies, .5)), ("p95", np.quantile(latencies, .95))):
                check(float(value), performance["latency_seconds"][key], "latency")
            for value in performance["batch_throughput"].values():
                check(performance["documents"] / value["seconds"], value["documents_per_second"], "throughput")
    samples = [json.loads(line) for line in gzip.decompress((directory / "telemetry.jsonl.gz").read_bytes()).decode().splitlines()]
    check(telemetry_summary(samples, bundle["phase_boundaries"]), bundle["telemetry_summary"], "telemetry")
    controller = bundle["controller"]
    check(controller["exit_code"], 0, "successful_exit")
    elapsed = (datetime.fromisoformat(controller["finished_utc"]) - datetime.fromisoformat(controller["started_utc"])).total_seconds()
    np.testing.assert_allclose(elapsed, controller["elapsed_seconds"], atol=.001, rtol=0)
    assert 0 < bundle["runs"]["25000"]["history"]["seconds"] <= controller["elapsed_seconds"]
    return {"status": "passed", "development_comparisons": 3, "training_samples": [4000, 16000, 25000],
            "selection_documents": 1000, "calibration_documents": 500, "test_evaluations": 0,
            "checks": "ordered cohorts, fitting/checkpoint selection, metrics/misses/policies, calibration, GPU replay, resources and telemetry"}


def export(directory):
    from filelock import FileLock
    import torch
    from export_binary import body_free, checkpoint_audit, local_sources, metadata
    from research_runtime import Runtime, verify as verify_runtime
    from schemas import HERE, load_split
    root = HERE / "training_runs/research-binary-finbert25k-20261015"
    recipe = root / "finbert-lora-n25000-c512"
    receipt = read(root / "receipt.json")
    assert receipt["status"] == "completed_artifacts_pending_independent_review" and receipt["exit_code"] == 0
    for name, value in receipt["preserved_sha256"].items():
        check(sha(Path(name)), value, "protected artifact")
    splits = HERE / "data/research/binary-v2"
    manifest = read(splits / "manifest.json")
    records = {name: load_split(splits, name) for name in ("train", "selection", "calibration")}
    cohorts = {name: [metadata(r) for r in rows] for name, rows in records.items()}
    subsets = read(splits / "subsets.json")
    torch.set_num_threads(4)
    verify_runtime(recipe, manifest)
    replay = root / "independent-development-logits.json"
    replay_identity = {"artifact_sha256": sha(recipe / "best.pt"), "config_sha256": sha(recipe / "config.json")}
    if not replay.exists():
        logits = {}
        started = time.perf_counter()
        with FileLock(HERE / "data/research/gpu-job.lock", timeout=0):
            runtime = Runtime(recipe)
            for cohort in ("selection", "calibration"):
                rows = list(records[cohort])
                random.Random(42).shuffle(rows)
                logits[cohort] = []
                for index, row in enumerate(rows, 1):
                    logits[cohort].append(runtime.score(row["body"])["logits"])
                    if index % 250 == 0:
                        print(f"Independent {cohort} replay: {index}/{len(rows)}", flush=True)
            torch.cuda.synchronize()
        write(replay, {**replay_identity, "logits": logits, "seconds": time.perf_counter() - started})
    independent = read(replay)
    check({k: independent[k] for k in replay_identity}, replay_identity, "GPU replay artifact")
    directory.mkdir(parents=True, exist_ok=True)
    runs, source_names = {}, {"finbert25k_evidence.py", "binary_replay.py", "run_finbert25k.py"}
    train = {r["body_sha256"]: r for r in cohorts["train"]}
    with zipfile.ZipFile(directory / "metadata.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for n in (4000, 16000, 25000):
            path = recipe if n == 25000 else HERE / f"training_runs/research-binary-v2/study/runs/finbert-lora-ce-w0.0-lr0.0001-n{n}-c512"
            verify_runtime(path, manifest)
            files = {filename: (path / filename).read_bytes() for filename in
                     ("config.json", "decision.json", "complete.json", "history.json", "selection.json", "calibration.json",
                      "performance.json", "validation.json", "epoch-zero-selection.json",
                      "validation-epoch-1.json", "validation-epoch-2.json", "validation-epoch-3.json")}
            decoded = {name: json.loads(raw) for name, raw in files.items()}
            for name, raw in files.items():
                body_free(decoded[name])
                archive.writestr(f"{n}/{name}", raw)
            fields = {field: decoded[name] for name, field in (("config.json", "configuration"), ("decision.json", "decision"),
                       ("complete.json", "complete"), ("history.json", "history"), ("selection.json", "selection"),
                       ("calibration.json", "calibration"), ("performance.json", "performance"), ("validation.json", "raw_selection"))}
            fields.update(checkpoint_audit=checkpoint_audit(path, fields["configuration"]),
                          metadata_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
                          composition={"sources": dict(Counter(train[h]["source"] for h in subsets["subsets"][str(n)])),
                                       "classes": dict(Counter(train[h]["reference"] for h in subsets["subsets"][str(n)])),
                                       "characters": sum(train[h]["characters"] for h in subsets["subsets"][str(n)])})
            check(fields["performance"]["adapter_bytes"], (path / "best.pt").stat().st_size, "measured_artifact_bytes")
            runs[str(n)] = fields
            source_names.update(fields["configuration"]["source_sha256"])
        archive.writestr("controller-receipt.json", (root / "receipt.json").read_bytes())
        archive.writestr("training.log", (root / "training.log").read_bytes())
    sources = local_sources(source_names)
    with zipfile.ZipFile(directory / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.writestr(name, (HERE / name).read_bytes())
    start = datetime.fromisoformat(receipt["started_utc"]).timestamp()
    zero = (recipe / "epoch-zero.pt").stat().st_mtime
    fit_end = (recipe / "validation-epoch-3.json").stat().st_mtime
    end = datetime.fromisoformat(receipt["finished_utc"]).timestamp()
    boundaries = [("startup_and_initial_selection", start, zero), ("fitting_and_epoch_validation", zero, fit_end),
                  ("final_calibration_selection_and_timing", fit_end, end)]
    telemetry = (root / "telemetry.jsonl").read_bytes()
    samples = [json.loads(line) for line in telemetry.decode().splitlines()]
    bundle = {"scope": "Post-hoc development evidence; choices followed original test inspection. No generalization claim.",
              "cohorts": cohorts, "subsets": subsets, "manifest_sha256": sha(splits / "manifest.json"), "runs": runs,
              "matched_initial_weights_sha256": runs["4000"]["checkpoint_audit"]["epoch-zero.pt"]["weights_sha256"],
              "independent_logits": independent["logits"], "independent_inference_receipt": {k: v for k, v in independent.items() if k != "logits"},
              "independent_raw_metrics": {name: breakdown(rows_from_logits(runs["25000"][name]["predictions"], independent["logits"][name]))
                                          for name in ("selection", "calibration")},
              "controller": receipt, "phase_boundaries": boundaries, "telemetry_summary": telemetry_summary(samples, boundaries),
              "timing_note": "Phase boundaries use saved file modification timestamps; telemetry samples are device-wide observations, not kernel busy time."}
    body_free(bundle)
    (directory / "evidence.json.gz").write_bytes(gzip.compress(json.dumps(bundle, allow_nan=False).encode(), mtime=0))
    (directory / "telemetry.jsonl.gz").write_bytes(gzip.compress(telemetry, mtime=0))
    shutil.copyfile(HERE / "finbert25k_evidence.py", directory / "verify.py")
    shutil.copyfile(HERE / "binary_replay.py", directory / "binary_replay.py")
    write(directory / "local-audit.json", {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
          "protected_files_verified": len(receipt["preserved_sha256"]), "finite_initial_best_final_checkpoints": True,
          "artifact_source_runtime_verified": True, "development_gpu_replay": independent["seconds"],
          "test_evaluations": 0, "raw_bodies_or_weights_exported": False})
    seal(directory)
    result = verify(directory)
    write(directory / "cpu-replay.json", result)
    seal(directory)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.directory) if args.export else verify(args.directory), indent=2))
