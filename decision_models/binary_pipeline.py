"""Resume the approved label queues, freeze binary data, and launch only the bounded study."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
from filelock import FileLock

from label import BINARY_CEILING_USD, BINARY_RUBRIC, BINARY_RUBRIC_HASH, reservation
from prepare_binary import NAMES, reference_metadata, verify_inputs
from schemas import BINARY_LABELS, BINARY_TASK, HERE, annotation, body_hash, file_hash, read_records, save, write_records


def completed(path):
    rows = read_records(path)
    return bool(rows) and all(r["human"]["label"] is None and annotation(r)["label"] in BINARY_LABELS and
        annotation(r).get("uncertainty") is not None and annotation(r).get("response_id") and
        annotation(r).get("rubric_sha256") == BINARY_RUBRIC_HASH and
        annotation(r).get("task_id") == BINARY_TASK for r in rows)


def active(record):
    """A PID alone is never enough to identify an owned process."""
    try:
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["create_time"]) < .01 and process.cmdline() == record["command"]:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def process_records(value):
    if all(key in value for key in ("pid", "create_time", "command")):
        yield value
    for key in ("stages", "jobs", "descendants"):
        for child in value.get(key, []):
            yield from process_records(child)


def stop_tree(record):
    """Revalidate both recorded survivors and the live parent's current descendants."""
    records = list(process_records(record))
    for owner in list(records):
        process = active(owner)
        if process:
            for child in process.children(recursive=True):
                try:
                    records.append({"pid": child.pid, "create_time": child.create_time(), "command": child.cmdline()})
                except psutil.NoSuchProcess:
                    pass
    stopped = {}
    for owner in records:
        process = active(owner)
        if process and process.pid not in stopped:
            try:
                process.terminate()
                stopped[process.pid] = (process, owner)
            except psutil.NoSuchProcess:
                pass
    _, remaining = psutil.wait_procs([p for p, _ in stopped.values()], timeout=10)
    for process in remaining:
        owner = stopped[process.pid][1]
        verified = active(owner)
        if verified:
            try:
                verified.kill()
            except psutil.NoSuchProcess:
                pass
    _, remaining = psutil.wait_procs(remaining, timeout=10)
    if any(active(stopped[p.pid][1]) for p in remaining):
        raise RuntimeError("Verified pipeline descendants did not stop; inspect the retained receipt")


def preserved(session):
    for name, expected in session["preserved_sha256"].items():
        if hashlib.sha256((HERE.parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original artifact changed: {name}")


def run(splits, output, session_path):
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "pipeline.lock", timeout=0):
        run_locked(splits, output, session_path)


def run_locked(splits, output, session_path):
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if session.get("task_id") != BINARY_TASK or session.get("labels") != list(BINARY_LABELS) or session.get("incremental_budget_usd") != BINARY_CEILING_USD:
        raise ValueError("Pipeline requires the authorized binary task and unchanged $2 incremental cap")
    cutoff = datetime.fromisoformat(session["compute_cutoff_utc"]).timestamp()
    queue = HERE / "data/research/binary-v2-labels"
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "pipeline-receipt.json"
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if any(active(record) for record in process_records(old)):
            raise ValueError("A recorded pipeline or child is still running; do not duplicate it")
    prepared = json.loads((splits / "prepared.json").read_text(encoding="utf-8"))
    settings = {"splits": str(splits), "session_sha256": file_hash(session_path),
                "prepared_sha256": file_hash(splits / "prepared.json")}
    sources = {name: file_hash(HERE / name) for name in
               ("binary_pipeline.py", "prepare_binary.py", "label.py", "schemas.py", "study_binary.py")}
    if receipt.exists() and (old.get("settings") != settings or old.get("source_sha256") != sources):
        raise ValueError("Pipeline inputs or source changed on resume; retain its receipt and use a reviewed new run")
    process = psutil.Process()
    state = {"pid": process.pid, "create_time": process.create_time(), "command": process.cmdline(),
             "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "stages": [],
             "source_sha256": sources, "settings": settings}
    save(state, receipt)

    def verify_sources():
        if file_hash(session_path) != settings["session_sha256"] or file_hash(splits / "prepared.json") != settings["prepared_sha256"] or any(file_hash(HERE / name) != expected for name, expected in sources.items()):
            raise ValueError("Pipeline source/session/preparation changed after launch")

    def execute(name, command):
        verify_sources()
        if time.time() >= cutoff:
            raise TimeoutError("Computation cutoff reached")
        with (output / f"{name}.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=HERE.parent, stdout=log, stderr=subprocess.STDOUT)
            stage = {"name": name, "command": command, "pid": child.pid,
                     "create_time": psutil.Process(child.pid).create_time()}
            state["stages"].append(stage)
            save(state, receipt)
            while True:
                verify_sources()
                remaining = cutoff - time.time()
                if remaining <= 0:
                    stop_tree(stage)
                    stage["stopped_at_cutoff"] = True
                    save(state, receipt)
                    raise TimeoutError("Pipeline stopped at computation cutoff")
                owner = active(stage)
                if owner:
                    descendants = stage.setdefault("descendants", [])
                    known = {(p["pid"], p["create_time"]) for p in descendants}
                    for descendant in owner.children(recursive=True):
                        try:
                            identity = {"pid": descendant.pid, "create_time": descendant.create_time(), "command": descendant.cmdline()}
                            if (identity["pid"], identity["create_time"]) not in known:
                                descendants.append(identity)
                        except psutil.NoSuchProcess:
                            pass
                    save(state, receipt)
                try:
                    stage["exit_code"] = child.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if any(active(record) for record in stage.get("descendants", [])):
                stop_tree(stage)
                raise RuntimeError("Command exited with live descendants; do not advance or duplicate them")
        save(state, receipt)
        if stage["exit_code"]:
            raise RuntimeError(f"{name} failed; inspect retained log and process receipt")

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        preserved(session)
        verify_inputs(prepared, splits)
        initial = queue / "controller-receipt.json"
        while True:
            if time.time() >= cutoff:
                if initial.exists():
                    stop_tree(json.loads(initial.read_text(encoding="utf-8")))
                raise TimeoutError("Initial labels did not finish before cutoff")
            value = json.loads(initial.read_text(encoding="utf-8"))
            if value["status"] == "completed":
                if any(active(record) for record in process_records(value)):
                    time.sleep(min(1, max(0, cutoff - time.time())))
                    continue
                break
            if value["status"] != "running":
                raise RuntimeError("Initial label controller stopped/failed; inspect its receipt")
            if active(value) is None:
                raise ValueError("Initial label controller is missing or its identity changed")
            time.sleep(min(30, max(0, cutoff - time.time())))
        redecision, audit = queue / "redecision.jsonl", queue / "retained-label-audit.jsonl"
        if not completed(redecision) or not completed(audit):
            raise ValueError("Initial label queues are incomplete or belong to another rubric/task")
        verify_sources()
        verify_inputs(prepared, splits)
        known = {body_hash(row): row for row in read_records(redecision)}
        new = []
        for name in NAMES:
            for row in read_records(splits / f"{name}-input.jsonl"):
                basis, key = row["reference_protocol"]["basis"], body_hash(row)
                if basis == "binary_redecision":
                    if key not in known or reference_metadata(row) != reference_metadata(known[key]):
                        raise ValueError("Initial redecision queue changed prepared membership or metadata")
                elif basis == "new_binary_label":
                    if key in known:
                        raise ValueError("A new document unexpectedly appears in the original redecision queue")
                    new.append(row)
                elif basis != "retained_three_class":
                    raise ValueError("Unknown prepared reference basis")
        if len({body_hash(r) for r in new}) != len(new):
            raise ValueError("Prepared new label queue contains duplicate bodies")
        path = queue / "new-documents.jsonl"
        if path.exists():
            existing = read_records(path)
            if [reference_metadata(r) for r in existing] != [reference_metadata(r) for r in new]:
                raise ValueError("New label queue membership or metadata changed on resume")
        else:
            write_records(new, path)
        save({"documents": len(new), "body_order_sha256": hashlib.sha256(json.dumps([body_hash(r) for r in new]).encode()).hexdigest(),
              "maximum_reservation_usd": sum(reservation(r["body"], BINARY_RUBRIC)[1] for r in new)},
             queue / "new-documents-preflight.json")
        for attempt in range(2):
            if completed(path):
                break
            execute(f"new-labels-{attempt + 1}", [sys.executable, "-u", str(HERE / "label.py"), "--dataset", str(path),
                    "--task", BINARY_TASK, "--incremental-budget-usd", "2"])
        if not completed(path):
            raise ValueError("New document references incomplete; do not freeze/train")
        if [reference_metadata(r) for r in read_records(path)] != [reference_metadata(r) for r in new]:
            raise ValueError("Label worker changed prepared document metadata")
        if not (splits / "manifest.json").exists():
            execute("freeze-data", [sys.executable, "-u", str(HERE / "prepare_binary.py"), "--output", str(splits),
                    "--freeze", "--labels", str(redecision), str(path), "--retained-audit", str(audit)])
        manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("prepared") != prepared or manifest.get("label_sources_sha256") != {str(p.resolve()): file_hash(p) for p in (redecision, path)} or manifest.get("retained_label_consistency", {}).get("sha256") != file_hash(audit):
            raise ValueError("Frozen benchmark no longer matches preparation and completed label queues")
        execute("bounded-study", [sys.executable, "-u", str(HERE / "study_binary.py"), "--splits", str(splits),
                "--output", str(output / "study"), "--session", str(session_path), "--stage", "all"])
        preserved(session)
        state["status"] = "models_complete_report_pending"
    except Exception as exc:
        state.update(status="stopped" if isinstance(exc, TimeoutError) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        for stage in reversed(state["stages"]):
            try:
                stop_tree(stage)
            except Exception as stop_error:
                state["stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
        raise
    finally:
        state["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(state, receipt)
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=HERE / "data/research/binary-v2")
    parser.add_argument("--output", type=Path, default=HERE / "training_runs/research-binary-v2")
    parser.add_argument("--session", type=Path, default=HERE / "training_runs/research-binary-v2/session.json")
    args = parser.parse_args()
    run(args.splits.resolve(), args.output.resolve(), args.session.resolve())
