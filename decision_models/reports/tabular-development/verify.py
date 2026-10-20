"""Replay exported tabular development metrics on CPU; no bodies, weights or model inference."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "binary-finbert25k"))
from binary_replay import breakdown, check, verify_policies, verify_temperature  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    evidence = (HERE / "evidence.json.gz").read_bytes()
    bundle = json.loads(gzip.decompress(evidence))
    replay = (HERE.parent / "binary-finbert25k/binary_replay.py").read_bytes()
    check(sha(replay), bundle["replay_source_sha256"], "independent arithmetic")
    original = (HERE.parent / "binary-v2/evidence.json.gz").read_bytes()
    check(sha(original), bundle["reference_evidence_sha256"], "reference package")
    reference = json.loads(gzip.decompress(original))
    check(bundle["scope"], "Development only; no test inference or fine-tuning", "scope")
    cohorts = reference["cohorts"]
    training = {r["body_sha256"]: r["reference"] for r in cohorts["train"]}
    for name, scores in bundle["references"].items():
        for cohort, metrics in scores.items():
            check(metrics, reference["runs"][name][cohort]["metrics"]["overall"], "reference metrics")
    summary = {}
    for name, files in bundle["runs"].items():
        run = {key.removesuffix(".json"): json.loads(value) for key, value in files.items()}
        config, decision = run["config"], run["decision"]
        check(config["finetune"]["steps"], 0, "pretrained head")
        check(config["task_id"], "binary-v2", "task")
        for key in ("config", "decision"):
            check(sha(files[f"{key}.json"].encode()), run["complete"][f"{key}_sha256"], key)
        for filename, expected in config["source_sha256"].items():
            check(sha(bundle["source"][filename].encode()), expected, filename)
        subset = reference["subsets"]["subsets"][str(config["samples"])]
        check(len(subset), config["samples"], "training count")
        pairs = [(key, training[key]) for key in subset]
        check(sha(json.dumps(pairs).encode()), config["train_subset_sha256"], "ordered training")
        check(max(run["search"]["trials"], key=lambda r: r["macro_f1"]), config["selected"], "selected width")
        scores = {}
        for cohort in ("selection", "calibration"):
            rows = run[cohort]["predictions"]
            fields = ("body_sha256", "group_id", "date", "source", "family", "reference", "reference_source")
            check([[r[key] for key in fields] for r in rows],
                  [[r[key] for key in fields] for r in cohorts[cohort]], f"ordered {cohort}")
            scores[cohort] = breakdown(rows)
            check(scores[cohort], run[cohort]["metrics"], f"{cohort} metrics")
            p = np.array([r["probabilities"] for r in rows])
            for key, policy in decision["policies"].items():
                discarded = (p[:, 1] < .5) & (p[:, 0] >= policy["threshold"])
                check(discarded.tolist(), [r["discard"][key] for r in rows], f"{cohort} routing")
        verify_policies(run["calibration"]["predictions"], decision)
        verify_temperature(run["calibration"]["predictions"], decision["temperature"])
        check(scores["selection"]["overall"]["macro_f1"], config["selected"]["macro_f1"], "selected F1")
        summary[name] = {cohort: {key: scores[cohort]["overall"][key]
                                for key in ("macro_f1", "false_negatives", "skip_fraction")}
                         for cohort in scores}
    print(json.dumps({"status": "passed", "evidence_sha256": sha(evidence), "runs": summary}, indent=2))


if __name__ == "__main__":
    main()
