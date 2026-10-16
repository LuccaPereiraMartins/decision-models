"""Publish verified binary results without exporting document bodies or model weights."""

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import zipfile

import numpy as np

from binary_replay import LABELS, check, verify as replay
from evaluate_study import validate_freeze
from prepare import family
from protocol import breakdown, fingerprint
from schemas import HERE, annotation, body_hash, load_split


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def seal_package(output, freeze_sha256):
    files = sorted(p for p in output.iterdir() if p.is_file() and p.name != "package-manifest.json")
    write(output / "package-manifest.json", {
          "task_id": "binary-v2", "freeze_sha256": freeze_sha256,
          "files": {p.name: fingerprint(p) for p in files},
          "limits": "CPU arithmetic replay; exact fitting needs the local frozen bodies, checkpoints and pinned pretrained weights"})


def local_sources(filenames):
    found, pending = set(filenames), list(filenames)
    while pending:
        tree = ast.parse((HERE / pending.pop()).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                filename = module.split(".")[0] + ".py"
                if (HERE / filename).is_file() and filename not in found:
                    found.add(filename)
                    pending.append(filename)
    return sorted(found)


def metadata(row):
    return {"body_sha256": body_hash(row), "group_id": row["group_id"], "date": row["date"],
            "source": row.get("source", "sec"), "family": family(row),
            "document_type": row.get("document_type", "8-K"), "characters": len(row["body"]),
            "reference": annotation(row)["label"],
            "reference_source": "human" if row["human"]["label"] is not None else "teacher",
            "teacher_uncertainty": row["llm"].get("uncertainty"),
            "reference_basis": row["reference_protocol"]["basis"]}


def document_sources(records, manifest):
    registry, path, links = None, None, {}
    for cohort, rows in records.items():
        for row in rows:
            digest = body_hash(row)
            if row.get("document_id") and row.get("url"):
                link = {"document_id": row["document_id"], "url": row["url"], "scope": "captured_source"}
            else:
                if cohort != "train":
                    raise ValueError("Evaluation source identifiers are missing")
                if registry is None:
                    path = HERE / "data/research/legacy-events/events.json"
                    key = manifest["prepared"]["settings"]["legacy_events"]
                    check(fingerprint(path), manifest["prepared"]["inputs"][key], "legacy_registry_identity")
                    value = read(path)
                    check(value["status"], "complete", "legacy_registry_status")
                    registry = {info["legacy_body_sha256"]: (event, info) for event, info in value["events"].items()}
                    check(len(registry), len(value["events"]), "unique_legacy_links")
                event, info = registry[digest]
                match = re.fullmatch(r"Archives-edgar-data-(\d+)-(\d{18})-(\d{10}-\d{2}-\d{6})\.txt-[0-9a-f]+", info["cache_file"])
                if not match:
                    raise ValueError(f"Unrecognized retained submission path: {event}")
                cik, directory, accession = match.groups()
                check(("sec:" + accession, accession.replace("-", ""), info["date"]),
                      (event, directory, row["date"]), "legacy_source_identity")
                link = {"document_id": event, "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{directory}/{accession}.txt",
                        "scope": "legacy_parent_submission"}
            if not link["url"].startswith(("https://", "http://")) or digest in links:
                raise ValueError("Source URL is invalid or a body occurs in multiple cohorts")
            links[digest] = link
    return links, path


def body_free(value):
    if isinstance(value, dict):
        if {"body", "text", "weights", "optimizer", "api_key"} & value.keys():
            raise ValueError("Raw content, weights or secrets do not belong in public evidence")
        for child in value.values():
            body_free(child)
    elif isinstance(value, list):
        for child in value:
            body_free(child)


def controller_receipt(state, name):
    record = {k: state["jobs"][name][k] for k in
              ("status", "elapsed_seconds", "started_utc", "ended_utc") if k in state["jobs"][name]}
    attempts = [row for row in state.get("attempts", []) if row["job"] == name]
    record["attempts"] = attempts
    completed = [row for row in attempts if row.get("exit_code") == 0]
    if completed:
        successful = completed[-1]["elapsed_seconds"]
        prior = record["elapsed_seconds"] - successful
        if prior < -1e-6:
            raise ValueError(f"Attempt exceeds cumulative command time: {name}")
        record.update(successful_attempt_seconds=successful, prior_attempt_seconds=max(0, prior))
    return record


def checkpoint_audit(directory, config):
    import torch
    receipts = {}
    for name in ("epoch-zero.pt", "best.pt", "last.pt"):
        value = torch.load(directory / name, map_location="cpu", weights_only=True)
        ignored = {"git_revision", "gpu"}  # The trainer records these on a source-identical research resume.
        if {k: v for k, v in value["config"].items() if k not in ignored} != {k: v for k, v in config.items() if k not in ignored}:
            raise ValueError(f"Checkpoint configuration differs: {directory / name}")
        weights = value["weights"]
        if not weights or any(not torch.isfinite(v).all().item() for v in weights.values()):
            raise ValueError(f"Nonfinite checkpoint: {directory / name}")
        digest = hashlib.sha256()
        for key, tensor in sorted(weights.items()):
            digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        receipts[name] = {"sha256": fingerprint(directory / name), "weights_sha256": digest.hexdigest(), "tensors": len(weights),
                          "parameters": sum(v.numel() for v in weights.values()),
                          "epoch": value["state"]["epoch"], "best_epoch": value["state"]["best_epoch"],
                          "recorded_git_revision": value["config"].get("git_revision"), "recorded_gpu": value["config"].get("gpu")}
    history = read(directory / "history.json")
    best = max(history["history"], key=lambda row: row["validation"]["macro_f1"])
    check(best["epoch"], history["best_epoch"], "selected_epoch")
    check(receipts["best.pt"]["epoch"], history["best_epoch"], "checkpoint_epoch")
    check(receipts["last.pt"]["epoch"], config["epochs"], "completed_epochs")
    return receipts


def diagnostics(bundle, freeze, plan):
    check(plan["selection_input_sha256"], bundle["benchmark_manifest"]["prepared"]["prepared_input_sha256"]["selection"],
          "diagnostic_selection_identity")
    boundaries = plan["characters_quartiles"]
    cohort = {r["body_sha256"]: r for r in bundle["cohorts"]["test"]}
    models = {}
    for name, run in bundle["runs"].items():
        rows = run["test"]["predictions"]
        models[name] = {"by_length": {str(i): breakdown([r for r in rows if
                            np.searchsorted(boundaries, r["characters"], side="right") == i], labels=LABELS)["overall"]
                            for i in range(4)},
                        "by_reference_basis": {basis: breakdown([r for r in rows if
                            cohort[r["body_sha256"]]["reference_basis"] == basis], labels=LABELS)["overall"]
                            for basis in sorted({r["reference_basis"] for r in cohort.values()})}}
    def misses(key):
        return {r["body_sha256"] for r in bundle["runs"][key]["test"]["predictions"]
                if r["reference"] == LABELS[1] and r["prediction"] == LABELS[0]}
    left, right = misses(freeze["primary_decision"]), misses(freeze["primary_linear"])
    return {"plan": plan, "models": models, "primary_miss_overlap": {
            "both": sorted(left & right), "laya_only": sorted(left - right), "tfidf_only": sorted(right - left)}}


def export(splits, study, output):
    freeze = validate_freeze(splits, study)
    if freeze["manifest"].get("task_id") != "binary-v2" or freeze["manifest"]["labels"] != list(LABELS):
        raise ValueError("Export only the revised binary study")
    results, statistics, state = [read(study / name) for name in ("results.json", "bootstrap.json", "state.json")]
    check(fingerprint(study / "bootstrap.json"), results["bootstrap_sha256"], "bootstrap_hash")
    records = {name: load_split(splits, name) for name in ("train", "selection", "calibration", "test")}
    check({name: len(rows) for name, rows in records.items()},
          {"train": 25000, "selection": 1000, "calibration": 500, "test": 2000}, "counts")
    cohorts = {name: [metadata(r) for r in rows] for name, rows in records.items()}
    links, legacy_path = document_sources(records, freeze["manifest"])
    subsets = read(splits / "subsets.json")
    by_hash = {r["body_sha256"]: r for r in cohorts["train"]}
    runs, local_audits, sources = {}, {}, {}
    for name, entry in freeze["entries"].items():
        directory = study / entry["directory"]
        config, decision = read(directory / "config.json"), read(directory / "decision.json")
        selected, calibration = read(directory / "selection.json"), read(directory / "calibration.json")
        test_path = study / "test" / f"{name}.json"
        check(fingerprint(test_path), results["test_artifacts"][name], f"{name}.test_hash")
        test = read(test_path)
        check(test["freeze_sha256"], fingerprint(study / "freeze.json"), f"{name}.freeze")
        check(read(study / "test" / f"{name}.complete.json"),
              {"freeze_sha256": test["freeze_sha256"], "result_sha256": fingerprint(test_path)}, f"{name}.completion")
        for cohort, evaluation in (("selection", selected), ("calibration", calibration), ("test", test)):
            expected = list(cohorts[cohort])
            if config["family"] == "research_neural" and cohort != "test":
                random.Random(42).shuffle(expected)
            rows = evaluation["predictions"]
            check([r["body_sha256"] for r in rows], [r["body_sha256"] for r in expected], f"{name}.{cohort}.order")
            for row, identity in zip(rows, expected, strict=True):
                check({k: row[k] for k in identity if k != "reference_basis"},
                      {k: v for k, v in identity.items() if k != "reference_basis"}, f"{name}.{cohort}.metadata")
        calibration_identity = [(r["body_sha256"], r["reference"]) for r in calibration["predictions"]]
        check(hashlib.sha256(json.dumps(calibration_identity).encode()).hexdigest(),
              decision["calibration_sha256"], f"{name}.calibration_order_hash")
        if config["family"] in ("research_linear", "research_neural"):
            train = [by_hash[h] for h in subsets["subsets"][str(config["samples"])]]
            identity = [(r["body_sha256"], r["reference"]) for r in train]
            check(hashlib.sha256(json.dumps(identity).encode()).hexdigest(), config["train_subset_sha256"], f"{name}.train_order_hash")
        for filename, digest in config["source_sha256"].items():
            check(fingerprint(HERE / filename), digest, f"source.{filename}")
            sources[filename] = digest
        scored = [json.loads(line) for line in (study / "test" / f"{name}-logits.jsonl").read_text(encoding="utf-8").splitlines()]
        check([r["body_sha256"] for r in scored], [r["body_sha256"] for r in cohorts["test"]], f"{name}.logit_order")
        public_config = {k: v for k, v in config.items() if k != "split_manifest"}
        run = {"entry": entry, "configuration": public_config, "decision": decision,
               "complete": read(directory / "complete.json"), "performance": read(directory / "performance.json"),
               "selection": selected, "calibration": calibration, "test": test,
               "test_logits": [r["logits"] for r in scored],
               "controller": controller_receipt(state, name)}
        for filename in ("history.json", "search.json"):
            if (directory / filename).exists():
                run[filename.removesuffix(".json")] = read(directory / filename)
        if config["family"] == "research_neural":
            run["checkpoint_audit"] = checkpoint_audit(directory, config)
        local_audits[name] = {"exported_json_sha256": {f: fingerprint(directory / f) for f in
                            ("config.json", "decision.json", "calibration.json", "selection.json", "performance.json")},
                            "test_logits_sha256": fingerprint(study / "test" / f"{name}-logits.jsonl")}
        runs[name] = run
    body_free(runs)
    for name, run in runs.items():
        matched = run["entry"].get("matched_zero")
        if matched:
            control = runs[matched]
            owner = runs[control["entry"]["control_of"]]
            check(run["checkpoint_audit"]["epoch-zero.pt"]["weights_sha256"],
                  owner["checkpoint_audit"]["epoch-zero.pt"]["weights_sha256"], f"{name}.matched_initial_weights")
            check(control["complete"]["artifact_sha256"], owner["checkpoint_audit"]["epoch-zero.pt"]["sha256"],
                  f"{name}.zero_artifact")
    bundle = {"task_id": "binary-v2", "labels": list(LABELS), "freeze_sha256": fingerprint(study / "freeze.json"),
              "benchmark_manifest": freeze["manifest"], "cohorts": cohorts, "subsets": subsets,
              "document_sources": links,
              "runs": runs, "bootstrap": statistics, "results": results, "schedule": state.get("laya_16k_gate",{})}
    evidence_receipts = ("cohort-freeze.json", "input-archive.json", "label-source-lineage.json",
                         "audit-membership-correction.json", "sdk-memory-correction.json")
    bundle["provenance_receipts"] = {name: read(output / name) for name in evidence_receipts if (output / name).exists()}
    if (output / "diagnostic-plan.json").exists():
        bundle["descriptive"] = diagnostics(bundle, freeze, read(output / "diagnostic-plan.json"))
    body_free(bundle)
    output.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    (output / "evidence.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    shutil.copyfile(HERE / "binary_replay.py", output / "verify.py")
    shutil.copyfile(study / "freeze.json", output / "freeze.json")
    with zipfile.ZipFile(output / "metadata.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if legacy_path is not None:
            body_free(read(legacy_path))
            item = zipfile.ZipInfo("preparation/legacy-events.json", date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, legacy_path.read_bytes())
        for name, entry in freeze["entries"].items():
            directory = study / entry["directory"]
            for filename in ("config.json", "decision.json", "complete.json", "performance.json", "selection.json", "calibration.json",
                             "history.json", "search.json"):
                if not (directory / filename).exists():
                    continue
                body_free(read(directory / filename))
                item = zipfile.ZipInfo(f"{name}/{filename}", date_time=(1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(item, (directory / filename).read_bytes())
    write(output / "export-audit.json", {"status": "local_frozen_identity_checks_passed", "runs": local_audits,
                                          "source_sha256": sources, "exporter_sha256": fingerprint(Path(__file__))})
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in local_sources({*sources, "export_binary.py", "binary_replay.py", "report_binary.py", "study_binary.py",
                                       "binary_pipeline.py", "prepare_binary.py", "label.py", "evaluate_study.py"}):
            item = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, (HERE / filename).read_bytes())
        lineage = bundle["provenance_receipts"].get("label-source-lineage.json", {})
        for versions in (lineage.get("initial_redecision_loaded_sources", {}), lineage.get("later_worker_sources", {})):
            for filename, digest in versions.items():
                snapshot = HERE / "data/research/binary-v2-labels" / f"{Path(filename).stem}-source-{digest[:12]}.py"
                check(fingerprint(snapshot), digest, f"label_source_snapshot.{filename}")
                item = zipfile.ZipInfo(f"label-history/{snapshot.name}", date_time=(1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(item, snapshot.read_bytes())
    (output / ".gitattributes").write_bytes(b"* -text\n")
    composition = {}
    for size, hashes in subsets["subsets"].items():
        rows = [by_hash[h] for h in hashes]
        composition[size] = {"documents": len(rows), "characters": sum(r["characters"] for r in rows),
                             "sources": dict(Counter(r["source"] for r in rows)),
                             "labels": dict(Counter(r["reference"] for r in rows))}
    write(output / "training-composition.json", composition)
    seal_package(output, bundle["freeze_sha256"])
    receipt = replay(output)
    write(output / "cpu-replay.json", receipt)
    seal_package(output, bundle["freeze_sha256"])
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=HERE / "data/research/binary-v2")
    parser.add_argument("--study", type=Path, default=HERE / "training_runs/research-binary-v2/study")
    parser.add_argument("--output", type=Path, default=HERE / "reports/binary-v2")
    args = parser.parse_args()
    print(json.dumps(export(args.splits, args.study, args.output), indent=2))
