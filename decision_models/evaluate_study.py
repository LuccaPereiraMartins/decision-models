"""Frozen-study test inference and paired event-group bootstrap; never select on test."""

from collections import defaultdict
import json
import time

import numpy as np

from protocol import breakdown, fingerprint, metrics, prediction_rows
from research_runtime import Runtime, verify
from schemas import BINARY_LABELS, BINARY_TASK, LABELS, body_hash, load_split, save, task_labels


def validate_freeze(splits, output):
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    labels = task_labels(freeze["manifest"])
    if task_labels(plan) != labels:
        raise ValueError("Frozen plan and benchmark tasks disagree")
    for key, filename in (("plan", "plan.json"), ("selection", "selection.json"), ("state", "state.json")):
        if fingerprint(output / filename) != freeze[f"{key}_sha256"]:
            raise ValueError(f"Frozen study changed: {filename}")
    if (
        fingerprint(splits / "manifest.json") != plan["manifest_sha256"]
        or fingerprint(splits / "subsets.json") != plan["subsets_sha256"]
    ):
        raise ValueError("Frozen benchmark metadata changed")
    for name, entry in freeze["entries"].items():
        directory = output / entry["directory"]
        verify(directory, freeze["manifest"])
        for filename, digest in entry["sha256"].items():
            if fingerprint(directory / filename) != digest:
                raise ValueError(f"Frozen run changed: {name}/{filename}")
    for name, digest in plan["source_sha256"].items():
        from schemas import HERE

        if fingerprint(HERE / name) != digest:
            raise ValueError(f"Frozen evaluator changed: {name}")
    return freeze


def bootstrap(predictions, reference, repeats=2000, contrasts=None, *, labels=LABELS):
    """Shared group draws preserve pairing, document weights and within-event dependence."""
    if not predictions or reference not in predictions or repeats < 1:
        raise ValueError("Bootstrap requires models, a reference and positive repeats")
    first = next(iter(predictions.values()))
    if not first:
        raise ValueError("Bootstrap requires a nonempty cohort")
    width = len(labels)
    identities = [(r["body_sha256"], r["group_id"], r["reference"]) for r in first]
    keys = sorted({r["group_id"] for r in first})
    lookup = {key: i for i, key in enumerate(keys)}
    groups = np.array([lookup[r["group_id"]] for r in first])
    draws = np.random.default_rng(42).multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=repeats)
    distributions, result = {}, {}
    for name, rows in predictions.items():
        if [(r["body_sha256"], r["group_id"], r["reference"]) for r in rows] != identities:
            raise ValueError("Bootstrap models must have identical ordered cohorts and references")
        if any(len(row["probabilities"]) != width for row in rows):
            raise ValueError("Bootstrap probability columns disagree with the declared task")
        truth = np.array([labels.index(r["reference"]) for r in rows])
        predicted = np.array([labels.index(r["prediction"]) for r in rows])
        cells = np.zeros((len(keys), width * width), dtype=np.int64)
        np.add.at(cells, (groups, truth * width + predicted), 1)
        confusion = (draws @ cells).reshape(repeats, width, width)
        actual, guessed = confusion.sum(2), confusion.sum(1)
        correct = confusion.diagonal(axis1=1, axis2=2)
        f1 = np.divide(
            2 * correct, actual + guessed, out=np.zeros_like(actual, dtype=float), where=actual + guessed > 0
        )
        values = {
            "macro_f1": f1.mean(1),
            "dangerous_miss_rate": np.divide(
                confusion[:, 1, 0], actual[:, 1], out=np.full(repeats, np.nan), where=actual[:, 1] > 0
            ),
        }
        if tuple(labels) == BINARY_LABELS:
            values.update(
                review_precision=np.divide(
                    correct[:, 1], guessed[:, 1], out=np.zeros(repeats), where=guessed[:, 1] > 0
                ),
                review_recall=np.divide(correct[:, 1], actual[:, 1], out=np.zeros(repeats), where=actual[:, 1] > 0),
                review_workload=guessed[:, 1] / actual.sum(1),
                skip_fraction=guessed[:, 0] / actual.sum(1),
            )
        for policy in rows[0].get("discard", {}):
            counts = np.zeros((len(keys), 3), dtype=np.int64)
            discarded = np.array([r["discard"][policy] for r in rows])
            np.add.at(
                counts, groups, np.column_stack((discarded, (truth == 1) & discarded, np.ones(len(rows), dtype=bool)))
            )
            count = draws @ counts
            values[f"workload_{policy}"] = count[:, 0] / count[:, 2]
            values[f"dangerous_miss_rate_{policy}"] = np.divide(
                count[:, 1], actual[:, 1], out=np.full(repeats, np.nan), where=actual[:, 1] > 0
            )
        distributions[name] = values
        result[name] = {metric: interval(vector) for metric, vector in values.items()}
    for name, values in distributions.items():
        result[name]["paired_delta_vs_reference"] = {
            metric: interval(values[metric] - distributions[reference][metric]) for metric in values
        }
    differences = {}
    for name, (left, right) in (contrasts or {}).items():
        differences[name] = {
            "left": left,
            "right": right,
            "metrics": {
                metric: interval(values - distributions[right][metric])
                for metric, values in distributions[left].items()
            },
        }
    result = {
        "reference": reference,
        "groups": len(keys),
        "documents": len(first),
        "repeats": repeats,
        "seed": 42,
        "method": "Percentile 95% intervals; paired resampling of whole event/near-duplicate groups",
        "limits": "Conditional on this fixed trained run; no training-seed uncertainty or multiplicity correction",
        "models": result,
        "contrasts": differences,
    }
    if tuple(labels) == BINARY_LABELS:
        result.update(task_id=BINARY_TASK, labels=list(labels))
    return result


def interval(values):
    valid = np.asarray(values)[np.isfinite(values)]
    if not len(valid):
        return {"lower": None, "upper": None, "valid_replicates": 0}
    lower, upper = np.quantile(valid, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper), "valid_replicates": len(valid)}


def claim_holdout(splits, output, freeze_hash):
    """A new output directory must not silently reset the test-set firewall."""
    from filelock import FileLock

    path = splits / "test-evaluation.json"
    identity = {"manifest_sha256": fingerprint(splits / "manifest.json"), "freeze_sha256": freeze_hash}
    with FileLock(splits / "test-evaluation.lock", timeout=0):
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if any(saved[key] != value for key, value in identity.items()):
                raise ValueError("This holdout belongs to another frozen study; a new study needs fresh test data")
        else:
            save({**identity, "study": str(output.resolve())}, path)


def evaluate(splits, output, *, deadline=None):
    from filelock import FileLock
    from schemas import HERE

    def check_deadline():
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Authorized inference deadline reached; resumable evidence retained")

    check_deadline()
    # All model choices and artifacts are checked before opening the test file.
    freeze = validate_freeze(splits, output)
    labels = task_labels(freeze["manifest"])
    marker = {"freeze_sha256": fingerprint(output / "freeze.json")}
    path = output / "test-started.json"
    if path.exists() and json.loads(path.read_text()) != marker:
        raise ValueError("Test inference belongs to another freeze")
    claim_holdout(splits, output, marker["freeze_sha256"])
    save(marker, path)
    records = load_split(splits, "test")
    predictions = {}
    directory = output / "test"
    directory.mkdir(exist_ok=True)
    for name, entry in freeze["entries"].items():
        path = directory / f"{name}.json"
        complete = directory / f"{name}.complete.json"
        config = json.loads((output / entry["directory"] / "config.json").read_text(encoding="utf-8"))
        decision = json.loads((output / entry["directory"] / "decision.json").read_text(encoding="utf-8"))
        if path.exists() and complete.exists():
            finished = json.loads(complete.read_text(encoding="utf-8"))
            if finished != {"freeze_sha256": marker["freeze_sha256"], "result_sha256": fingerprint(path)}:
                raise ValueError("Completed test result changed")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["freeze_sha256"] != marker["freeze_sha256"]:
                raise ValueError("Saved test results have another freeze")
            predictions[name] = result["predictions"]
            continue
        cpu = config["family"] == "research_linear" and config["kind"] != "finbert"
        if not cpu:
            import torch
        with FileLock(HERE / "data/research/gpu-job.lock"):
            check_deadline()
            runtime = Runtime(output / entry["directory"], "cpu" if cpu else "cuda")
            cache = directory / f"{name}-logits.jsonl"
            scored = (
                [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()] if cache.exists() else []
            )
            if len(scored) > len(records) or any(
                r["body_sha256"] != body_hash(record) for r, record in zip(scored, records)
            ):
                raise ValueError("Resumable test predictions changed cohort/order")
            with cache.open("a", encoding="utf-8") as stream:
                for record in records[len(scored) :]:
                    check_deadline()
                    if not cpu:
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    row = {"body_sha256": body_hash(record), **runtime.score(record["body"])}
                    if not cpu:
                        torch.cuda.synchronize()
                    row["seconds"] = time.perf_counter() - started
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    scored.append(row)
                    if len(scored) % 100 == 0:
                        print(f"TEST {name}: {len(scored)}/{len(records)}", flush=True)
            rows = prediction_rows(records, [r["logits"] for r in scored], decision, labels=labels)
            raw_metrics = breakdown(
                prediction_rows(records, [r["logits"] for r in scored], labels=labels), labels=labels
            )
            for row, diagnostic in zip(rows, scored, strict=True):
                row.update({k: v for k, v in diagnostic.items() if k not in ("body_sha256", "logits")})
            save(
                {
                    "freeze_sha256": marker["freeze_sha256"],
                    "metrics": breakdown(rows, labels=labels),
                    "raw_metrics": raw_metrics,
                    "predictions": rows,
                    "inference_seconds": sum(r["seconds"] for r in scored),
                },
                path,
            )
            save({"freeze_sha256": marker["freeze_sha256"], "result_sha256": fingerprint(path)}, complete)
            predictions[name] = rows
            del runtime
            import gc

            gc.collect()
            if not cpu:
                torch.cuda.empty_cache()
    summaries(output, predictions, freeze)


def summaries(output, predictions, freeze):
    reference = freeze["primary_linear"]
    labels = task_labels(freeze["manifest"])
    contrasts, curves = {}, defaultdict(list)
    for name, (larger, smaller) in freeze.get("sample_contrasts", {}).items():
        if (
            larger not in freeze["entries"]
            or smaller not in freeze["entries"]
            or freeze["entries"][larger]["samples"] <= freeze["entries"][smaller]["samples"]
        ):
            raise ValueError("Frozen sample contrast must name an increasing training-size pair")
        contrasts[name] = (larger, smaller)
    for name, entry in freeze["entries"].items():
        if entry.get("matched_zero"):
            contrasts[name + "-adaptation"] = (name, entry["matched_zero"])
        config = json.loads((output / entry["directory"] / "config.json").read_text())
        if entry["role"] == "curve":
            key = tuple(config.get(k) for k in ("family", "kind", "adaptation", "classifier", "max_len", "loss_kind"))
            curves[key].append(name)
    for names in curves.values():
        names.sort(key=lambda n: freeze["entries"][n]["samples"])
        for smaller, larger in zip(names, names[1:]):
            contrasts[larger + "-sample-increment"] = (larger, smaller)
    save(bootstrap(predictions, reference, contrasts=contrasts, labels=labels), output / "bootstrap.json")
    save(
        {
            "models": {
                name: {
                    "default_dangerous_misses": [
                        r["body_sha256"]
                        for r in rows
                        if r["reference"] == "review_worthy" and r["prediction"] == "routine"
                    ],
                    "metrics": metrics(rows, labels=labels),
                }
                for name, rows in predictions.items()
            },
            "test_artifacts": {name: fingerprint(output / "test" / f"{name}.json") for name in predictions},
            "bootstrap_sha256": fingerprint(output / "bootstrap.json"),
        },
        output / "results.json",
    )
