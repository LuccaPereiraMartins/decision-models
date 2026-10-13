"""Independent CPU arithmetic for the body-free binary evidence package."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = ("routine", "review_worthy")


def check(actual, expected, path="value"):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"Different fields: {path}")
        for key, value in expected.items():
            check(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Different lengths: {path}")
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            check(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11, err_msg=path)
    elif actual != expected:
        raise ValueError(f"Mismatch at {path}: {actual!r} != {expected!r}")


def probability(logits, temperature=1.):
    x = np.asarray(logits, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all() or not 0 < temperature < np.inf:
        raise ValueError("Expected finite two-column logits and positive temperature")
    x = x / temperature
    x -= x.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def policy(truth, skipped):
    truth, skipped = np.asarray(truth), np.asarray(skipped, dtype=bool)
    count, support = int(skipped.sum()), int((truth == 1).sum())
    misses = int(((truth == 1) & skipped).sum())
    return {"discarded": count, "workload_reduction": count / len(truth),
            "review_worthy_support": support, "dangerous_misses": misses,
            "dangerous_miss_rate": misses / support if support else None,
            "discard_contamination": misses / count if count else None,
            "reviewed": len(truth) - count, "review_workload": 1 - count / len(truth)}


def scores(rows):
    if not rows:
        return {"count": 0}
    y = np.array([LABELS.index(r["reference"]) for r in rows])
    p = np.array([r["probabilities"] for r in rows], dtype=float)
    if p.shape != (len(rows), 2) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError("Invalid probability distribution")
    guessed = (p[:, 1] >= .5).astype(int)
    check([LABELS[k] for k in guessed], [r["prediction"] for r in rows], "predictions")
    cm = np.bincount(y * 2 + guessed, minlength=4).reshape(2, 2)
    actual, predicted, correct = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(2), where=predicted > 0)
    recall = np.divide(correct, actual, out=np.zeros(2), where=actual > 0)
    f1 = np.divide(2 * correct, actual + predicted, out=np.zeros(2), where=actual + predicted > 0)
    confidence, bins, ece = p.max(1), [], 0.
    for i in range(10):
        mask = (confidence >= i / 10) & ((confidence < (i + 1) / 10) if i < 9 else (confidence <= 1))
        accuracy = float((guessed[mask] == y[mask]).mean()) if mask.any() else None
        mean = float(confidence[mask].mean()) if mask.any() else None
        if mask.any():
            ece += mask.mean() * abs(accuracy - mean)
        bins.append({"lower": i / 10, "upper": (i + 1) / 10, "count": int(mask.sum()),
                     "accuracy": accuracy, "confidence": mean})
    default = policy(y, guessed == 0)
    return {"count": len(rows), "macro_f1": float(f1.mean()), "accuracy": float(correct.sum() / len(rows)),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(actual[i])} for i, name in enumerate(LABELS)},
            "confusion_matrix": cm.tolist(), "default_discard": default,
            "brier": float((2 * (p[:, 1] - y) ** 2).mean()),
            "log_loss": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
            "ece_10_bins": float(ece), "calibration_bins": bins,
            "policies": {key: policy(y, [r["discard"][key] for r in rows]) for key in rows[0].get("discard", {})},
            "task_id": "binary-v2", "labels": list(LABELS),
            "review_precision": float(precision[1]), "review_recall": float(recall[1]),
            "false_negatives": default["dangerous_misses"], "false_negative_rate": default["dangerous_miss_rate"],
            "review_workload": default["review_workload"], "skip_fraction": default["workload_reduction"]}


def breakdown(rows):
    result = {"overall": scores(rows)}
    for field in ("family", "source", "reference_source"):
        result[f"by_{field}"] = {key: scores([r for r in rows if r[field] == key])
                                 for key in sorted({r[field] for r in rows})}
    return result


def verify_policies(rows, decision):
    """Enumerate every whole-tie cutoff independently of the fitting implementation."""
    p = np.asarray([r["probabilities"] for r in rows])
    y = np.asarray([LABELS.index(r["reference"]) for r in rows])
    thresholds = [1.000001, *sorted(set(p[:, 0]), reverse=True)]
    candidates = [(t, policy(y, (p[:, 1] < .5) & (p[:, 0] >= t))) for t in thresholds]
    for key, saved in decision["policies"].items():
        if "target_miss_rate" in saved:
            valid = [(t, s) for t, s in candidates if s["dangerous_miss_rate"] is not None
                     and s["dangerous_miss_rate"] <= saved["target_miss_rate"]]
        else:
            budget = int(np.floor(saved["target_skip_fraction"] * len(rows)))
            valid = [(t, s) for t, s in candidates if s["discarded"] <= budget]
        chosen = max(valid, key=lambda pair: pair[1]["discarded"]) if valid else candidates[0]
        check({"threshold": chosen[0], **chosen[1]},
              {k: saved[k] for k in ("threshold", *chosen[1])}, f"calibration_policy.{key}")


def verify_temperature(rows, temperature):
    # Positive calibrated probabilities recover the original binary log-odds up to
    # a shared logit offset. NLL is convex in inverse temperature.
    p = np.asarray([r["probabilities"] for r in rows])
    if (p <= 0).any():
        raise ValueError("Underflow prevents temperature replay; preserve raw calibration logits")
    difference = temperature * (np.log(p[:, 1]) - np.log(p[:, 0]))
    y = np.array([LABELS.index(r["reference"]) for r in rows])
    low, high = np.exp(-3), np.exp(3)
    for _ in range(100):
        middle = (low + high) / 2
        z = middle * difference
        q = np.exp(-np.logaddexp(0, -z))
        if np.dot(difference, q - y) > 0:
            high = middle
        else:
            low = middle
    optimal = (low + high) / 2
    def loss(inverse):
        q = np.exp(-np.logaddexp(0, -inverse * difference))
        return -np.log(np.maximum(np.where(y == 1, q, 1 - q), 1e-12)).mean()
    if loss(1 / temperature) > loss(optimal) + 1e-8:
        raise ValueError("Saved temperature does not minimize calibration NLL")


def rows_from_logits(rows, logits, decision=None):
    p = probability(logits, decision["temperature"] if decision else 1.)
    rebuilt = []
    for row, vector in zip(rows, p, strict=True):
        review = bool(vector[1] >= .5)
        rebuilt.append({**row, "probabilities": vector.tolist(),
                        "prediction": LABELS[int(review)], "uncertainty": float(1 - vector.max()),
                        "discard": {k: bool(not review and vector[0] >= v["threshold"])
                                    for k, v in decision["policies"].items()} if decision else {}})
    return rebuilt


def replay_bootstrap(predictions, expected):
    """Replay paired draws via document multiplicities, rather than grouped confusion cells."""
    first = next(iter(predictions.values()))
    groups = sorted({r["group_id"] for r in first})
    lookup = {g: i for i, g in enumerate(groups)}
    index = np.array([lookup[r["group_id"]] for r in first])
    draws = np.random.default_rng(expected["seed"]).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=expected["repeats"])
    weights = draws[:, index]
    vectors = {}
    truth = np.array([LABELS.index(r["reference"]) for r in first])

    def interval(values):
        finite = values[np.isfinite(values)]
        bounds = np.quantile(finite, [.025, .975]) if len(finite) else (None, None)
        return {"lower": float(bounds[0]) if len(finite) else None,
                "upper": float(bounds[1]) if len(finite) else None, "valid_replicates": len(finite)}

    for name, rows in predictions.items():
        check([(r["body_sha256"], r["group_id"], r["reference"]) for r in rows],
              [(r["body_sha256"], r["group_id"], r["reference"]) for r in first], "bootstrap_cohort")
        guessed = np.array([LABELS.index(r["prediction"]) for r in rows])
        cells = [weights @ ((truth == t) & (guessed == g)).astype(np.int64) for t in (0, 1) for g in (0, 1)]
        tn, fp, fn, tp = cells
        support, n = tp + fn, tn + fp + fn + tp
        def divide(a, b, missing=0.):
            return np.divide(a, b, out=np.full(len(a), missing), where=b > 0)
        values = {"macro_f1": (divide(2 * tn, 2 * tn + fp + fn) + divide(2 * tp, 2 * tp + fp + fn)) / 2,
                  "dangerous_miss_rate": divide(fn, support, np.nan), "review_precision": divide(tp, tp + fp),
                  "review_recall": divide(tp, support), "review_workload": (tp + fp) / n,
                  "skip_fraction": (tn + fn) / n}
        for key in rows[0]["discard"]:
            skipped = np.array([r["discard"][key] for r in rows])
            values[f"workload_{key}"] = (weights @ skipped.astype(np.int64)) / n
            values[f"dangerous_miss_rate_{key}"] = divide(weights @ ((truth == 1) & skipped).astype(np.int64), support, np.nan)
        vectors[name] = values
    reference = vectors[expected["reference"]]
    for name, values in vectors.items():
        result = {k: interval(v) for k, v in values.items()}
        result["paired_delta_vs_reference"] = {k: interval(v - reference[k]) for k, v in values.items()}
        check(result, expected["models"][name], f"bootstrap.{name}")
    for name, contrast in expected["contrasts"].items():
        check({k: interval(v - vectors[contrast["right"]][k]) for k, v in vectors[contrast["left"]].items()},
              contrast["metrics"], f"contrast.{name}")
    check(len(first), expected["documents"], "bootstrap_documents")
    check(len(groups), expected["groups"], "bootstrap_groups")


def verify(directory):
    manifest = json.loads((directory / "package-manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Package file changed: {name}")
    bundle = json.loads(gzip.decompress((directory / "evidence.json.gz").read_bytes()))
    freeze = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    check(hashlib.sha256((directory / "freeze.json").read_bytes()).hexdigest(), manifest["freeze_sha256"], "exact_freeze_bytes")
    check(set(freeze["entries"]), set(bundle["runs"]), "frozen_models")
    check(bundle["task_id"], "binary-v2")
    check(bundle["labels"], list(LABELS))
    check(bundle["freeze_sha256"], manifest["freeze_sha256"])
    check({name: len(rows) for name, rows in bundle["cohorts"].items()},
          {"train": 25000, "selection": 1000, "calibration": 500, "test": 2000}, "cohort_counts")
    groups = []
    for name, rows in bundle["cohorts"].items():
        check(len({r["body_sha256"] for r in rows}), len(rows), f"{name}.unique_bodies")
        group = {r["group_id"] for r in rows}
        if any(group & previous for previous in groups):
            raise ValueError("Related groups cross cohort boundaries")
        groups.append(group)
    subsets = bundle["subsets"]["subsets"]
    check(subsets["16000"][:4000], subsets["4000"], "nested_4k")
    check(subsets["25000"][:16000], subsets["16000"], "nested_16k")
    check(set(bundle["runs"]), set(bundle["results"]["models"]), "complete_models")
    predictions, timing_sample = {}, None
    for name, run in bundle["runs"].items():
        check(run["entry"], freeze["entries"][name], f"{name}.frozen_entry")
        decision = run["decision"]
        if not np.exp(-3) <= decision["temperature"] <= np.exp(3):
            raise ValueError(f"Temperature outside fitting bounds: {name}")
        for cohort in ("selection", "calibration", "test"):
            evaluation = run[cohort]
            rows = evaluation["predictions"]
            ids = {r["body_sha256"] for r in bundle["cohorts"][cohort]}
            check(len(rows), len(ids), f"{name}.{cohort}.count")
            check(set(r["body_sha256"] for r in rows), ids, f"{name}.{cohort}.identity")
            identities = {r["body_sha256"]: r for r in bundle["cohorts"][cohort]}
            for row in rows:
                identity = identities[row["body_sha256"]]
                check({k: row[k] for k in identity if k != "reference_basis"},
                      {k: v for k, v in identity.items() if k != "reference_basis"}, f"{name}.{cohort}.metadata")
            check(breakdown(rows), evaluation["metrics"], f"{name}.{cohort}.metrics")
            for row in rows:
                skipped = {k: bool(row["prediction"] == LABELS[0] and row["probabilities"][0] >= v["threshold"])
                           for k, v in decision["policies"].items()}
                check(skipped, row["discard"], f"{name}.{cohort}.decisions")
            if cohort == "calibration":
                verify_policies(rows, decision)
                verify_temperature(rows, decision["temperature"])
            elif cohort == "test":
                rebuilt = rows_from_logits(rows, run["test_logits"], decision)
                for a, e in zip(rebuilt, rows, strict=True):
                    check(a["probabilities"], e["probabilities"], f"{name}.test.probabilities")
                    check(a["prediction"], e["prediction"], f"{name}.test.prediction")
                    check(a["discard"], e["discard"], f"{name}.test.policy")
                check(breakdown(rows_from_logits(rows, run["test_logits"])), evaluation["raw_metrics"], f"{name}.raw_metrics")
                check(scores(rows), bundle["results"]["models"][name]["metrics"], f"{name}.summary")
                check([r["body_sha256"] for r in rows if r["reference"] == LABELS[1] and r["prediction"] == LABELS[0]],
                      bundle["results"]["models"][name]["default_dangerous_misses"], f"{name}.miss_identity")
                predictions[name] = rows
        performance = run["performance"]
        latencies = np.asarray(performance["latency_seconds"]["measurements"])
        if len(latencies) != performance["documents"] or not np.isfinite(latencies).all() or (latencies < 0).any():
            raise ValueError(f"Invalid timing measurements: {name}")
        for key, value in (("mean", latencies.mean()), ("p50", np.quantile(latencies, .5)), ("p95", np.quantile(latencies, .95))):
            check(float(value), performance["latency_seconds"][key], f"{name}.latency.{key}")
        if timing_sample is None:
            timing_sample = set(performance["sample"])
        check(set(performance["sample"]), timing_sample, f"{name}.timing_sample")
        for value in performance["batch_throughput"].values():
            check(performance["documents"] / value["seconds"], value["documents_per_second"], f"{name}.throughput")
    replay_bootstrap(predictions, bundle["bootstrap"])
    if "descriptive" in bundle:
        diagnostic = bundle["descriptive"]
        boundaries = diagnostic["plan"]["characters_quartiles"]
        identity = {r["body_sha256"]: r for r in bundle["cohorts"]["test"]}
        for name, rows in predictions.items():
            for i in range(4):
                sliced = [r for r in rows if sum(r["characters"] >= boundary for boundary in boundaries) == i]
                check(scores(sliced), diagnostic["models"][name]["by_length"][str(i)], f"{name}.length.{i}")
            for basis, expected in diagnostic["models"][name]["by_reference_basis"].items():
                check(scores([r for r in rows if identity[r["body_sha256"]]["reference_basis"] == basis]),
                      expected, f"{name}.reference_basis.{basis}")
        def missed(name):
            return {r["body_sha256"] for r in predictions[name] if r["reference"] == LABELS[1] and r["prediction"] == LABELS[0]}
        left, right = missed(freeze["primary_decision"]), missed(freeze["primary_linear"])
        check({"both": sorted(left & right), "laya_only": sorted(left - right), "tfidf_only": sorted(right - left)},
              diagnostic["primary_miss_overlap"], "primary_miss_overlap")
    return {"status": "passed", "models": len(predictions), "documents": len(next(iter(predictions.values()))),
            "checks": "Independent binary confusion/F1/FN/workload, probabilities, policies, timing and paired bootstrap"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
