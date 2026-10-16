"""One fixed 25k FinBERT LoRA development extension, with an eight-hour guard."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock
import psutil

from schemas import HERE, save


OUTPUT = HERE / "training_runs/research-binary-finbert25k-20261015"
SPLITS = HERE / "data/research/binary-v2"
RECIPE = OUTPUT / "finbert-lora-n25000-c512"
REFERENCE = HERE / "training_runs/research-binary-v2/study"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(process):
    return {"pid": process.pid, "create_time": process.create_time(), "command": process.cmdline()}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not (OUTPUT / "receipt.json").exists(), "Do not duplicate an existing run"
    assert not RECIPE.exists(), "This fit starts from pretrained weights, not the 16k checkpoint"
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        assert not any(Path(arg).name in {"train.py", "study_binary.py", "research_runtime.py", "fit_baselines.py"}
                       for arg in command), f"Existing GPU work: {process.info}"
    with FileLock(HERE / "data/research/gpu-job.lock", timeout=0):
        pass  # The training worker acquires the same lock for its entire run.
    manifest = read(SPLITS / "manifest.json")
    assert manifest["task_id"] == "binary-v2"
    old_config = read(REFERENCE / "runs/finbert-lora-ce-w0.0-lr0.0001-n16000-c512/config.json")
    for name, value in old_config["source_sha256"].items():
        assert digest(HERE / name) == value, name
    session = read(HERE / "training_runs/research-binary-v2/session.json")
    protected = {HERE.parent / name: value for name, value in session["preserved_sha256"].items()}
    paths = [REFERENCE / name for name in ("plan.json", "state.json", "selection.json", "freeze.json",
                                          "results.json", "bootstrap.json", "test-started.json")]
    paths += [SPLITS / name for name in ("manifest.json", "subsets.json", "train.jsonl", "selection.jsonl", "calibration.jsonl")]
    paths += [path for folder in (HERE / "reports/binary-v2", HERE / "paper-binary-v2")
              for path in folder.rglob("*") if path.is_file() and path.suffix not in {".aux", ".out", ".log"}]
    protected.update({path: digest(path) for path in paths})
    assert all(digest(path) == value for path, value in protected.items())
    command = [sys.executable, "-u", str(HERE / "train.py"), "--model", "finbert", "--adaptation", "lora",
               "--lora-rank", "8", "--samples", "25000", "--context", "512", "--objective", "ce",
               "--class-weight-power", "0", "--learning-rate", "0.0001", "--encoder-learning-rate", "0.0001",
               "--epochs", "3", "--window-batch", "8", "--device", "cuda", "--splits", str(SPLITS),
               "--output", str(RECIPE)]
    started = time.time()
    cutoff = started + 8 * 3600
    receipt = {"status": "starting", "controller": identity(psutil.Process()), "command": command,
               "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
               "compute_cutoff_utc": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
               "scope": "Post-hoc development extension; training/selection/calibration only. No test evaluation.",
               "future_test": "Fresh held-out data is required for independent generalization evidence.",
               "runner_sha256": digest(Path(__file__)), "workers": 1, "api_cost_usd": 0,
               "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "preserved_sha256": {str(path): value for path, value in protected.items()}}
    save(receipt, OUTPUT / "receipt.json")
    awake = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
    awake.argtypes, awake.restype = [ctypes.c_uint32], ctypes.c_uint32
    child = None
    try:
        assert awake(0x80000001), "Cannot prevent idle sleep"
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        with (OUTPUT / "training.log").open("x", encoding="utf-8") as log, (OUTPUT / "telemetry.jsonl").open("x", encoding="utf-8") as telemetry:
            child = subprocess.Popen(command, cwd=HERE.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
            worker = psutil.Process(child.pid)
            worker.cpu_percent()
            receipt.update(status="running", worker=identity(worker))
            save(receipt, OUTPUT / "receipt.json")
            while child.poll() is None:
                try:
                    assert worker.create_time() == receipt["worker"]["create_time"] and worker.cmdline() == receipt["worker"]["command"]
                    sample = {"utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.time() - started,
                              "worker_cpu_percent": worker.cpu_percent(), "worker_rss_bytes": worker.memory_info().rss,
                              "cpu_times_seconds": worker.cpu_times()._asdict()}
                except psutil.NoSuchProcess:
                    if child.poll() is not None:
                        break
                    raise
                try:
                    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,pstate",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                    sample.update(gpu_csv=gpu.stdout.strip(), gpu_columns=["gpu_util_percent", "memory_util_percent", "memory_used_MiB",
                                  "power_W", "temperature_C", "sm_clock_MHz", "pstate"], gpu_returncode=gpu.returncode)
                    if gpu.returncode:
                        sample["gpu_error"] = gpu.stderr.strip()
                except subprocess.TimeoutExpired:
                    sample["gpu_error"] = "nvidia-smi observation timed out; training continues"
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                if time.time() >= cutoff:
                    child.terminate()
                    child.wait(timeout=30)
                    receipt["status"] = "stopped_at_cutoff"
                    break
                time.sleep(min(15, max(0, cutoff - time.time())))
            receipt["exit_code"] = child.wait()
            if receipt["status"] != "stopped_at_cutoff":
                receipt["status"] = "completed_artifacts_pending_independent_review" if child.returncode == 0 and (RECIPE / "complete.json").exists() else "failed"
        assert all(digest(path) == value for path, value in protected.items()), "Protected study/publication changed"
        receipt["preserved_hashes_verified"] = True
        if receipt["status"] == "completed_artifacts_pending_independent_review":
            from research_runtime import verify
            verify(RECIPE, manifest)
            receipt["artifact_source_runtime_verified"] = True
    except Exception as exc:
        receipt.update(status="needs_attention", error=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        awake(0x80000000)
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time() - started,
                       awake_guard_released=True)
        save(receipt, OUTPUT / "receipt.json")


if __name__ == "__main__":
    with FileLock(OUTPUT.parent / "binary-finbert25k-controller.lock", timeout=0):
        run()
