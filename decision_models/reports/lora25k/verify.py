"""Independent CPU replay of the additional 25k-model test comparison."""

import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "binary-finbert25k"))
from binary_replay import breakdown, check, rows_from_logits  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    evidence = (HERE / "evidence.json.gz").read_bytes()
    bundle = json.loads(gzip.decompress(evidence))
    original = (HERE.parent / "binary-v2/evidence.json.gz").read_bytes()
    development = (HERE.parent / "binary-finbert25k/evidence.json.gz").read_bytes()
    check(sha(original), bundle["reference_evidence_sha256"], "original evidence")
    check(sha(development), bundle["finbert_development_evidence_sha256"], "selected model evidence")
    reference = json.loads(gzip.decompress(original))
    selected = json.loads(gzip.decompress(development))["runs"]["25000"]
    files = bundle["files"]
    plan, result, raw, complete = [json.loads(files[f"{name}.json"])
                                   for name in ("plan", "test", "raw-logits", "complete")]
    for name, key in (("plan", "plan"), ("test", "test"), ("raw-logits", "raw_logits")):
        check(sha(files[f"{name}.json"].encode()), complete[f"{key}_sha256"], name)
    check(complete["selected_artifact_sha256"], selected["complete"]["artifact_sha256"], "fixed selected weights")
    with zipfile.ZipFile(HERE.parent / "binary-finbert25k/metadata.zip") as metadata:
        check(sha(metadata.read("25000/complete.json")), plan["complete_sha256"], "fixed completion")
    with zipfile.ZipFile(HERE.parent / "binary-finbert25k/source.zip") as source:
        for name, digest in plan["runtime_sha256"].items():
            check(sha(source.read(name)), digest, f"archived runtime {name}")
    check(sha(bundle["evaluator_source"].encode()), plan["evaluator_sha256"], "evaluator source")
    rows = result["predictions"]
    check(len(rows), 2000, "test count")
    fields = ("body_sha256", "group_id", "date", "source", "family", "reference", "reference_source")
    check([[r[k] for k in fields] for r in rows],
          [[r[k] for k in fields] for r in reference["cohorts"]["test"]], "ordered reused cohort")
    check([r["body_sha256"] for r in rows], raw["ordered_body_sha256"], "logit order")
    logits = np.array(raw["logits"])
    if logits.shape != (2000, 2) or not np.isfinite(logits).all():
        raise ValueError("Invalid saved test logits")
    check(rows_from_logits(rows, logits, selected["decision"]), rows, "fixed calibration and routing")
    check(breakdown(rows), result["metrics"], "all test metrics")
    print(json.dumps({"status": "passed", "evidence_sha256": sha(evidence), "test_documents": 2000,
                      "metrics": {k: result["metrics"]["overall"][k]
                                  for k in ("macro_f1", "false_negatives", "false_negative_rate", "skip_fraction")}}, indent=2))


if __name__ == "__main__":
    main()
