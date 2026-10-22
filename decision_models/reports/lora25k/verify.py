"""Independent CPU replay of the additional 25k-model test comparison."""

import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import random
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "binary-finbert25k"))
from binary_replay import (  # noqa: E402
    breakdown, check, replay_bootstrap, rows_from_logits, verify_fitting, verify_policies, verify_temperature,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_laya(bundle, reference):
    extension = bundle["laya"]
    files = extension["files"]
    plan, result, raw, complete = [json.loads(files[f"{name}.json"])
                                   for name in ("plan", "test", "raw-logits", "complete")]
    training = {name.removesuffix(".json"): json.loads(content)
                for name, content in extension["training_files"].items()}
    config, decision = training["config"], training["decision"]
    check(config["samples"], 25000, "Laya training count")
    recipe = bundle["laya_training_recipe"]
    for key in ("adaptation", "lora_rank", "epochs", "seed", "learning_rate", "encoder_learning_rate", "window_batch"):
        check(config[key], recipe[key], f"fixed Laya {key}")
    check(config["max_len"], recipe["context"], "fixed Laya context")
    check(sha(extension["training_files"]["complete.json"].encode()), plan["complete_sha256"], "fixed Laya fit")
    for name in ("config", "decision"):
        check(sha(extension["training_files"][f"{name}.json"].encode()), training["complete"][f"{name}_sha256"], name)
    for name, key in (("plan", "plan"), ("test", "test"), ("raw-logits", "raw_logits")):
        check(sha(files[f"{name}.json"].encode()), complete[f"{key}_sha256"], name)
    check(sha(extension["evaluator_source"].encode()), plan["evaluator_sha256"], "Laya evaluator")
    check(complete["selected_artifact_sha256"], training["complete"]["artifact_sha256"], "selected Laya weights")
    audit = extension["completed_audit"]
    check(audit["checkpoints"]["best.pt"]["sha256"], complete["selected_artifact_sha256"], "audited selected weights")
    train = {r["body_sha256"]: r["reference"] for r in reference["cohorts"]["train"]}
    pairs = [(key, train[key]) for key in reference["subsets"]["subsets"]["25000"]]
    check(sha(json.dumps(pairs).encode()), config["train_subset_sha256"], "ordered Laya training")
    metadata = {r["body_sha256"]: r for r in reference["cohorts"]["train"]}
    check({field: dict(Counter(metadata[key][field] for key, _ in pairs)) for field in ("source", "family", "reference")},
          extension["training_composition"], "training composition")
    with zipfile.ZipFile(HERE.parent / "binary-finbert25k/source.zip") as source:
        for name, digest in {**config["source_sha256"], **plan["runtime_sha256"]}.items():
            check(sha(source.read(name)), digest, f"Laya runtime {name}")
    verify_fitting({"configuration": config, "history": training["history"], "selection": training["selection"],
                    "checkpoint_audit": audit["checkpoints"]})
    check(training["history"]["steps"], 9375, "complete Laya updates")
    fields = ("body_sha256", "group_id", "date", "source", "family", "reference", "reference_source")
    for cohort in ("selection", "calibration"):
        saved = training[cohort]
        expected = list(reference["cohorts"][cohort])
        random.Random(42).shuffle(expected)
        check([[r[k] for k in fields] for r in saved["predictions"]],
              [[r[k] for k in fields] for r in expected], f"ordered Laya {cohort}")
        check(breakdown(saved["predictions"]), saved["metrics"], f"Laya {cohort} metrics")
        replay = extension["development_replay"][cohort]
        check(replay["ordered_body_sha256"], [r["body_sha256"] for r in replay["predictions"]], "replay order")
        check([[r[k] for k in fields] for r in replay["predictions"]],
              [[r[k] for k in fields] for r in reference["cohorts"][cohort]], "replay cohort")
        rebuilt = rows_from_logits(replay["predictions"], replay["logits"], decision)
        check(rebuilt, replay["predictions"], f"Laya {cohort} replay logits")
        check(breakdown(rebuilt), replay["metrics"], f"Laya {cohort} replay metrics")
        lookup = {r["body_sha256"]: r for r in rebuilt}
        ordered = [lookup[r["body_sha256"]] for r in saved["predictions"]]
        np.testing.assert_allclose([r["probabilities"] for r in ordered],
                                   [r["probabilities"] for r in saved["predictions"]], rtol=0, atol=1e-6)
        check([r["discard"] for r in ordered], [r["discard"] for r in saved["predictions"]], "replayed routing")
    verify_policies(training["calibration"]["predictions"], decision)
    verify_temperature(training["calibration"]["predictions"], decision["temperature"])
    epoch = training["history"]["best_epoch"]
    initial = training["epoch-zero-selection"]
    selected = training[f"validation-epoch-{epoch}"]["predictions"] if epoch else initial["predictions"]
    lookup = {r["body_sha256"]: r["logits"] for r in selected}
    rows = training["selection"]["predictions"]
    check(rows_from_logits(rows, [lookup[r["body_sha256"]] for r in rows], decision), rows, "raw selected Laya logits")
    rows = result["predictions"]
    check([[r[k] for k in fields] for r in rows],
          [[r[k] for k in fields] for r in reference["cohorts"]["test"]], "ordered Laya test")
    check([r["body_sha256"] for r in rows], raw["ordered_body_sha256"], "Laya test logit order")
    check(rows_from_logits(rows, raw["logits"], decision), rows, "fixed Laya test decisions")
    check(breakdown(rows), result["metrics"], "Laya test metrics")
    performance = training["performance"]
    samples = np.array(performance["latency_seconds"]["measurements"])
    check(len(samples), 128, "Laya timing count")
    previous = next(run for run in reference["runs"].values() if run["configuration"].get("kind") == "laya"
                    and run["configuration"].get("adaptation") == "lora" and run["configuration"]["samples"] == 16000)
    check(performance["sample"], previous["performance"]["sample"], "common timing documents")
    for key, value in (("mean", samples.mean()), ("p50", np.quantile(samples, .5)), ("p95", np.quantile(samples, .95))):
        check(float(value), performance["latency_seconds"][key], f"Laya timing {key}")
    for batch in performance["batch_throughput"].values():
        check(128 / batch["seconds"], batch["documents_per_second"], "Laya throughput")
    controller = extension["controller_receipt"]
    check(sha(extension["controller_source"].encode()), controller["controller_sha256"], "controller source")
    job = controller["jobs"][-1]
    duration = (datetime.fromisoformat(job["ended_utc"]) - datetime.fromisoformat(job["started_utc"])).total_seconds()
    np.testing.assert_allclose(duration, job["elapsed_seconds"], rtol=0, atol=1)
    if not duration >= training["history"]["seconds"] > 0:
        raise ValueError("Training/command timing is inconsistent")
    observations = extension["telemetry"]
    values = np.array([[float(x) for x in r["gpu_csv"].split(",")[:5]] for r in observations])
    summary = extension["telemetry_summary"]
    check(len(observations), summary["samples"], "telemetry count")
    check(float(values[:, 0].mean()), summary["device_gpu_util_mean_percent"], "sampled GPU mean")
    check(float(np.median(values[:, 0])), summary["device_gpu_util_median_percent"], "sampled GPU median")
    check(float(values[:, 2].max()), summary["device_memory_max_MiB"], "sampled device memory")
    check(float(np.mean([r["worker_cpu_percent"] for r in observations])), summary["worker_cpu_mean_percent"], "sampled CPU mean")
    return result


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
    results = {"finbert": result}
    if "laya" in bundle:
        results["laya"] = verify_laya(bundle, reference)
        predictions = {key: reference["runs"][key]["test"]["predictions"]
                       for key in bundle["bootstrap"]["models"] if key in reference["runs"]}
        predictions.update({f"{key}-lora-n25000-c{512 if key == 'finbert' else 1024}": saved["predictions"]
                            for key, saved in results.items()})
        replay_bootstrap(predictions, bundle["bootstrap"])
    print(json.dumps({"status": "passed", "evidence_sha256": sha(evidence), "test_documents": 2000,
                      "metrics": {name: {k: saved["metrics"]["overall"][k]
                                          for k in ("macro_f1", "false_negatives", "false_negative_rate", "skip_fraction")}
                                  for name, saved in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
