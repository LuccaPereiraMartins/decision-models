"""Replay published metrics and paired intervals on CPU, without source bodies or weights."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax


def check(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            check(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            check(a, e)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    else:
        assert actual == expected, (actual, expected)


def main():
    directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(directory.parents[1]))
    from evaluate_study import bootstrap
    from protocol import breakdown
    from schemas import LABELS

    manifest = json.loads((directory / "package-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    with gzip.open(directory / "predictions.json.gz", "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)
    details = json.loads(gzip.decompress((directory / "comparison-details.json.gz").read_bytes()))
    evaluations = json.loads(gzip.decompress((directory / "evaluations.json.gz").read_bytes()))
    results_bytes = gzip.decompress((directory / "results.json.gz").read_bytes())
    assert hashlib.sha256(results_bytes).hexdigest() == manifest["results_sha256"]
    results = json.loads(results_bytes)
    expected_bootstrap = json.loads((directory / "bootstrap.json").read_text())
    assert bundle["freeze_sha256"] == manifest["freeze_sha256"]
    assert len(bundle["cohort"]) == 1800 and len(bundle["models"]) == 64
    assert set(bundle["models"]) == set(details) == set(evaluations)
    predictions = {}
    for name, model in bundle["models"].items():
        decision = details[name]["decision"]
        logits = np.asarray(model["logits"])
        assert logits.shape == (1800, 3) and np.isfinite(logits).all()
        for calibrated in (False, True):
            p = softmax(logits / (decision["temperature"] if calibrated else 1.0), axis=1)
            rows = []
            for identity, vector in zip(bundle["cohort"], p, strict=True):
                rows.append({**identity, "probabilities": vector.tolist(), "prediction": LABELS[vector.argmax()],
                             "uncertainty": float(1 - vector.max()),
                             "discard": {k: bool(vector.argmax() == 0 and vector[0] >= v["threshold"])
                                         for k, v in decision["policies"].items()} if calibrated else {}})
            check(breakdown(rows), evaluations[name]["metrics" if calibrated else "raw_metrics"])
            if calibrated:
                predictions[name] = rows
                check(breakdown(rows)["overall"], results["models"][name]["metrics"])
                check([r["body_sha256"] for r in rows if r["reference"] == "review_worthy" and r["prediction"] == "routine"],
                      results["models"][name]["default_dangerous_misses"])
        np.testing.assert_allclose(sum(model["seconds"]), evaluations[name]["inference_seconds"], rtol=1e-12)
    contrasts = {k: (v["left"], v["right"]) for k, v in expected_bootstrap["contrasts"].items()}
    actual = bootstrap(predictions, expected_bootstrap["reference"], contrasts=contrasts)
    check(actual, expected_bootstrap)
    print("Verified all package hashes, 64 raw/calibrated evaluations and 2000 paired group-bootstrap draws.")


if __name__ == "__main__":
    main()
