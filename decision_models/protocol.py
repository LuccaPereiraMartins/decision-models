"""Shared research metrics, calibration, discard policies and artifact fingerprints."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from prepare import family, sample
from schemas import BINARY_TASK, LABELS, annotation, body_hash, load_split, save, task_labels


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def targets(records, *, labels=LABELS):
    return np.array([labels.index(annotation(row)["label"]) for row in records])


def probabilities(logits, temperature=1.0, *, labels=LABELS):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != len(labels) or (not np.isfinite(logits).all()):
        raise ValueError(f"Expected finite {len(labels)}-class logits")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    return softmax(logits / temperature, axis=1)


def temperature(logits, truth, *, labels=LABELS):
    """One scalar, selected by calibration-set multiclass negative log likelihood."""

    def objective(log_t):
        p = probabilities(logits, np.exp(log_t), labels=labels)
        return -np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-12)).mean()

    result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    if not result.success:
        raise ValueError("Temperature optimization failed")
    candidates = (0.0, float(result.x), -3.0, 3.0)
    chosen = min(candidates, key=objective)
    return float(np.exp(chosen))


def predicted_labels(probability, *, labels=LABELS):
    p = np.asarray(probability)
    return (p[:, 1] >= 0.5).astype(int)


def discard(probability, threshold, *, labels=LABELS):
    p = np.asarray(probability)
    return (predicted_labels(p, labels=labels) == 0) & (p[:, 0] >= threshold)


def policy_scores(truth, discarded, *, labels=LABELS):
    truth, discarded = (np.asarray(truth), np.asarray(discarded, dtype=bool))
    dangerous = int(((truth == 1) & discarded).sum())
    worthwhile = int((truth == 1).sum())
    n_discard = int(discarded.sum())
    result = {
        "discarded": n_discard,
        "workload_reduction": n_discard / len(truth),
        "review_worthy_support": worthwhile,
        "dangerous_misses": dangerous,
        "dangerous_miss_rate": dangerous / worthwhile if worthwhile else None,
        "discard_contamination": float(((truth != 0) & discarded).sum() / n_discard) if n_discard else None,
    }
    result.update(reviewed=len(truth) - n_discard, review_workload=(len(truth) - n_discard) / len(truth))
    return result


def fit_policies(p, truth, *, labels=LABELS):
    p, truth = (np.asarray(p), np.asarray(truth))
    policies = {}
    for rate in (0.01, 0.05):
        best = {"threshold": 1.000001, **policy_scores(truth, np.zeros(len(truth), dtype=bool), labels=labels)}
        if (truth == 1).any():
            for threshold in sorted(set(p[:, 0]), reverse=True):
                score = policy_scores(truth, discard(p, threshold, labels=labels), labels=labels)
                if score["dangerous_miss_rate"] <= rate and score["discarded"] > best["discarded"]:
                    best = {"threshold": float(threshold), **score}
        policies[str(rate)] = {
            **best,
            "target_miss_rate": rate,
            "criterion": "Empirical calibration miss rate; no population guarantee",
        }
    for fraction in (0.1, 0.2, 0.3):
        best = {"threshold": 1.000001, **policy_scores(truth, np.zeros(len(truth), dtype=bool), labels=labels)}
        for threshold in sorted(set(p[:, 0]), reverse=True):
            score = policy_scores(truth, discard(p, threshold, labels=labels), labels=labels)
            if score["discarded"] <= int(np.floor(fraction * len(truth))) and score["discarded"] > best["discarded"]:
                best = {"threshold": float(threshold), **score}
        policies[f"skip_{fraction}"] = {
            **best,
            "target_skip_fraction": fraction,
            "criterion": "Maximum calibration skip count within budget; whole probability ties; no population guarantee",
        }
    return policies


def prediction_rows(records, logits, decision=None, *, labels=LABELS):
    if decision is not None and task_labels(decision) != tuple(labels):
        raise ValueError("Decision and prediction tasks disagree")
    decision = decision or {"temperature": 1.0, "policies": {}}
    p = probabilities(logits, decision["temperature"], labels=labels)
    predicted = predicted_labels(p, labels=labels)
    rows = []
    for record, vector, guess in zip(records, p, predicted, strict=True):
        row = {
            "body_sha256": body_hash(record),
            "group_id": record.get("group_id", body_hash(record)),
            "date": record["date"],
            "source": record.get("source", "sec"),
            "family": family(record),
            "document_type": record.get("document_type", "8-K"),
            "characters": len(record["body"]),
            "reference": annotation(record)["label"],
            "reference_source": "human" if record["human"]["label"] is not None else "teacher",
            "teacher_uncertainty": record["llm"].get("uncertainty"),
            "prediction": labels[guess],
            "probabilities": vector.tolist(),
            "uncertainty": float(1 - vector.max()),
            "discard": {
                key: bool(guess == 0 and vector[0] >= value["threshold"]) for key, value in decision["policies"].items()
            },
        }
        row["task_id"] = BINARY_TASK
        rows.append(row)
    return rows


def metrics(rows, *, labels=LABELS):
    if not rows:
        return {"count": 0}
    truth = np.array([labels.index(r["reference"]) for r in rows])
    predicted = np.array([labels.index(r["prediction"]) for r in rows])
    p = np.array([r["probabilities"] for r in rows])
    if (
        p.shape != (len(rows), len(labels))
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (not np.allclose(p.sum(1), 1))
    ):
        raise ValueError("Prediction probabilities disagree with the declared task")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(range(len(labels))), zero_division=0
    )
    confidence = p.max(1)
    bins, ece = ([], 0.0)
    for index in range(10):
        mask = (confidence >= index / 10) & (confidence < (index + 1) / 10 if index < 9 else confidence <= 1)
        accuracy = float((predicted[mask] == truth[mask]).mean()) if mask.any() else None
        mean_confidence = float(confidence[mask].mean()) if mask.any() else None
        if mask.any():
            ece += float(mask.mean()) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": int(mask.sum()),
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    default_policy = policy_scores(truth, predicted == 0, labels=labels)
    result = {
        "count": len(rows),
        "macro_f1": float(
            f1_score(truth, predicted, labels=list(range(len(labels))), average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(truth, predicted)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(labels)))).tolist(),
        "default_discard": default_policy,
        "brier": float(((p - np.eye(len(labels))[truth]) ** 2).sum(1).mean()),
        "log_loss": float(-np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-12)).mean()),
        "ece_10_bins": ece,
        "calibration_bins": bins,
        "policies": {
            key: policy_scores(truth, [r["discard"][key] for r in rows], labels=labels)
            for key in rows[0].get("discard", {})
        },
    }
    result.update(
        task_id=BINARY_TASK,
        labels=list(labels),
        review_precision=float(precision[1]),
        review_recall=float(recall[1]),
        false_negatives=default_policy["dangerous_misses"],
        false_negative_rate=default_policy["dangerous_miss_rate"],
        review_workload=default_policy["review_workload"],
        skip_fraction=default_policy["workload_reduction"],
    )
    return result


def breakdown(rows, *, labels=LABELS):
    return {
        "overall": metrics(rows, labels=labels),
        "by_family": {
            key: metrics([r for r in rows if r["family"] == key], labels=labels)
            for key in sorted({r["family"] for r in rows})
        },
        "by_source": {
            key: metrics([r for r in rows if r["source"] == key], labels=labels)
            for key in sorted({r["source"] for r in rows})
        },
        "by_reference_source": {
            key: metrics([r for r in rows if r["reference_source"] == key], labels=labels)
            for key in sorted({r["reference_source"] for r in rows})
        },
    }


def calibrate(records, logits, output, *, labels=LABELS):
    truth = targets(records, labels=labels)
    t = temperature(logits, truth, labels=labels)
    p = probabilities(logits, t, labels=labels)
    decision = {
        "temperature": t,
        "method": "Single temperature on separate calibration groups",
        "policies": fit_policies(p, truth, labels=labels),
        "calibration_documents": len(records),
        "calibration_classes": dict(Counter((labels[i] for i in truth))),
        "calibration_sha256": hashlib.sha256(
            json.dumps([(body_hash(r), annotation(r)["label"]) for r in records]).encode()
        ).hexdigest(),
    }
    decision.update(task_id=BINARY_TASK, labels=list(labels), default_review_threshold=0.5)
    rows = prediction_rows(records, logits, decision, labels=labels)
    save(decision, output / "decision.json")
    save(
        {"metrics": breakdown(rows, labels=labels), "predictions": rows, "evaluation": "Calibration fitting data"},
        output / "calibration.json",
    )
    return decision


def training_subset(splits, number):
    rows = load_split(splits, "train")
    frozen = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    if frozen.get("subsets_sha256") and fingerprint(splits / "subsets.json") != frozen["subsets_sha256"]:
        raise ValueError("Frozen training subsets changed")
    manifest = json.loads((splits / "subsets.json").read_text(encoding="utf-8"))
    number = len(rows) if number is None else number
    identities = manifest["subsets"].get(str(number))
    if identities is None:
        raise ValueError(f"No frozen training subset of {number} documents")
    by_hash = {body_hash(r): r for r in rows}
    return [by_hash[key] for key in identities]


def speed(predict, records, synchronize=lambda: None):
    """Warmed full inference; persist the exact sample, batch definitions and measurements."""
    records = sample(records, 128)
    texts = [r["body"] for r in records]
    predict(texts[:1])
    synchronize()
    latencies = []
    for text in texts:
        synchronize()
        start = time.perf_counter()
        predict([text])
        synchronize()
        latencies.append(time.perf_counter() - start)
    throughput = {}
    for batch in (1, 8):
        synchronize()
        start = time.perf_counter()
        for offset in range(0, len(texts), batch):
            predict(texts[offset : offset + batch])
        synchronize()
        elapsed = time.perf_counter() - start
        throughput[str(batch)] = {"documents_per_second": len(texts) / elapsed, "seconds": elapsed}
    return {
        "documents": len(records),
        "sample": [body_hash(r) for r in records],
        "latency_seconds": {
            "p50": float(np.quantile(latencies, 0.5)),
            "p95": float(np.quantile(latencies, 0.95)),
            "mean": float(np.mean(latencies)),
            "measurements": latencies,
        },
        "batch_throughput": throughput,
        "measurement": "Warm inference incl. text processing; excludes loading; shared local machine",
    }
