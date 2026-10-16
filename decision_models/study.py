"""Run the bounded binary study, freeze every choice, then evaluate the revised cohort."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from protocol import fingerprint
from research_runtime import verify
from schemas import BINARY_LABELS, BINARY_TASK, HERE, body_hash, load_split, save, source_hashes, task_labels


@contextmanager
def keep_awake():
    """Keep this runner awake on Windows while allowing the display to turn off."""
    setter, previous = None, None
    if sys.platform == "win32":
        import ctypes

        setter = ctypes.windll.kernel32.SetThreadExecutionState
        setter.argtypes, setter.restype = [ctypes.c_uint], ctypes.c_uint
        previous = setter(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, no display requirement.
        if not previous:
            raise OSError("Could not keep the bounded runner awake")
    try:
        yield
    finally:
        if setter:
            setter(previous)


class BinaryStudy:
    optional_laya = "laya-lora-ce-w0.0-lr0.0001-n16000-c1024"

    def __init__(self, splits, output, session):
        self.splits, self.output = splits.resolve(), output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((self.splits / "manifest.json").read_text(encoding="utf-8"))
        if task_labels(self.manifest) != BINARY_LABELS:
            raise ValueError("The bounded runner requires the frozen binary-v2 benchmark")
        self.session = json.loads(session.read_text(encoding="utf-8"))
        self.cutoff = datetime.fromisoformat(self.session["compute_cutoff_utc"]).timestamp()
        self.sizes = [4000, 16000, 25000]
        subsets = json.loads((self.splits / "subsets.json").read_text(encoding="utf-8"))
        if any(str(n) not in subsets["subsets"] for n in self.sizes):
            raise ValueError("The study requires nested 4k/16k/25k training subsets")
        plan = {
            "task_id": BINARY_TASK,
            "labels": list(BINARY_LABELS),
            "seed": 42,
            "manifest_sha256": fingerprint(self.splits / "manifest.json"),
            "subsets_sha256": fingerprint(self.splits / "subsets.json"),
            "session_sha256": fingerprint(session),
            "compute_cutoff_utc": self.session["compute_cutoff_utc"],
            "primary_samples": 4000,
            "sizes": self.sizes,
            "epochs": 3,
            "primary_comparison": "Laya LoRA versus word TF-IDF/logistic, both at 4k",
            "neural_recipes": [
                {"kind": "laya", "adaptation": "head", "sizes": [4000], "context": 1024, "learning_rate": 3e-5},
                {"kind": "laya", "adaptation": "lora", "sizes": [4000, 16000], "context": 1024, "learning_rate": 1e-4},
                {
                    "kind": "finbert",
                    "adaptation": "lora",
                    "sizes": [4000, 16000],
                    "context": 512,
                    "learning_rate": 1e-4,
                },
            ],
            "linear_models": ["tfidf/logistic", "finbert/logistic"],
            "references": ["laya-sdk", "training-majority"],
            "lora_rank": 8,
            "objective": "Unweighted document-level CE; fixed uniform whole-document pooling",
            "checkpoint_selection": "Raw binary selection macro F1; epoch zero eligible",
            "policy_targets": [0.01, 0.05],
            "calibration_skip_budgets": [0.1, 0.2, 0.3],
            "bootstrap_replicates": 2000,
            "test_reuse": "Existing 1800 bodies retained and topped up; references revised; not a fresh confirmation",
            "laya_16k_gate": "Development/runtime only: forecast x1.3 plus 2h evaluation must fit before cutoff",
            "deferred": ["25k LoRA", "extra seeds", "context sweeps", "other encoders", "generative LLMs", "RL/RLCD"],
            "source_sha256": source_hashes("study.py", "train.py"),
        }
        path = self.output / "plan.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Bounded study inputs/source changed; preserve this study and use a new output")
        if not path.exists():
            save(plan, path)
        path = self.output / "state.json"
        self.state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"jobs": {}, "attempts": []}
        self.comparisons = {}

    def ensure_idle(self, record):
        """A stopped controller must not duplicate its still-running child."""
        if not record or "pid" not in record:
            return
        try:
            process = psutil.Process(record["pid"])
            if abs(process.create_time() - record["create_time"]) > 0.001 or not process.is_running():
                return
            if process.cmdline() != record["command"]:
                raise ValueError("Recorded live process command differs from its receipt")
            raise RuntimeError(f"Recorded job is still running: PID {record['pid']}")
        except psutil.NoSuchProcess:
            return

    def job(self, name, script, arguments, role="core", control=None, fallback_of=None):
        directory = self.output / "runs" / name
        command = [
            sys.executable,
            "-u",
            str(HERE / script),
            "--splits",
            str(self.splits),
            "--output",
            str(directory),
            *map(str, arguments),
        ]
        record = {
            "script": script,
            "arguments": list(map(str, arguments)),
            "directory": str(directory.relative_to(self.output)),
            "role": role,
            "control_of": control,
            "fallback_of": fallback_of,
        }
        previous = self.state["jobs"].get(name)
        self.ensure_idle(previous)
        if previous and any(previous[key] != record[key] for key in record):
            raise ValueError(f"Changed job on resume: {name}")
        if (directory / "complete.json").exists():
            verify(directory, self.manifest)
            record = {**(previous or record), "status": "completed"}
            record.pop("excluded_from_freeze", None)
        else:
            deadline = self.cutoff - 7200 if name == self.optional_laya else self.cutoff
            if time.time() >= deadline:
                raise TimeoutError("Computation cutoff reached; existing checkpoints retained")
            directory.mkdir(parents=True, exist_ok=True)
            if script != "research_runtime.py" and any((directory / f).exists() for f in ("last.pt", "config.json")):
                command.append("--resume")
            logs = self.output / "logs"
            logs.mkdir(exist_ok=True)
            print(f"START {name}", flush=True)
            started = time.time()
            with (logs / f"{name}.log").open("a", encoding="utf-8") as log:
                log_offset = log.tell()
                process = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
                record.update(
                    status="running",
                    pid=process.pid,
                    create_time=psutil.Process(process.pid).create_time(),
                    command=command,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                )
                self.state["jobs"][name] = record
                save(self.state, self.output / "state.json")
                try:
                    code = process.wait(timeout=max(1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=30)
                    elapsed = time.time() - started
                    record.update(
                        status="stopped",
                        limitation="Computation cutoff; last resumable checkpoint retained",
                        elapsed_seconds=(previous or {}).get("elapsed_seconds", 0) + elapsed,
                    )
                    self.state["attempts"].append(
                        {"job": name, "exit_code": None, "elapsed_seconds": elapsed, "limitation": record["limitation"]}
                    )
                    save(self.state, self.output / "state.json")
                    raise TimeoutError(record["limitation"]) from None
            elapsed = time.time() - started
            oom = (
                code
                and name == self.optional_laya
                and b"out of memory" in (logs / f"{name}.log").read_bytes()[log_offset:].lower()
            )
            record.update(
                status="completed" if code == 0 else "oom" if oom else "failed",
                exit_code=code,
                elapsed_seconds=(previous or {}).get("elapsed_seconds", 0) + elapsed,
                ended_utc=datetime.now(timezone.utc).isoformat(),
            )
            self.state["attempts"].append({"job": name, "exit_code": code, "elapsed_seconds": elapsed})
            save(self.state, self.output / "state.json")
            if code:
                raise RuntimeError(f"Job failed: {name}; inspect its retained log/checkpoint")
            verify(directory, self.manifest)
        self.state["jobs"][name] = record
        save(self.state, self.output / "state.json")
        self.comparisons[name] = {"directory": record["directory"], "role": role, "control_of": control}
        return name

    def fit(self):
        if (self.output / "freeze.json").exists() or (self.splits / "test-evaluation.json").exists():
            raise ValueError("Study is frozen/evaluated; no more fitting against this cohort")
        for record in self.state["jobs"].values():
            self.ensure_idle(record)
        for name in ("train", "selection", "calibration"):
            load_split(self.splits, name)
        choices = {}
        for kind in ("majority", "tfidf", "finbert"):
            name = f"{kind}-logistic-n4000"
            choices[kind] = self.job(
                name,
                "fit_baselines.py",
                ["--model", kind, "--samples", 4000, "--device", "cuda" if kind == "finbert" else "cpu"],
            )
        self.job("laya-sdk", "decision_base.py", ["--model", "laya"], "reference")
        choices["laya-head"] = self.neural("laya", "head", 4000, 1024, lr=3e-5, role="core")
        choices["laya-lora"] = self.neural("laya", "lora", 4000, 1024, role="core")
        choices["finbert-lora"] = self.neural("finbert", "lora", 4000, 512, role="core")
        for kind in ("tfidf", "finbert"):
            for n in (16000, 25000):
                self.job(
                    f"{kind}-logistic-n{n}",
                    "fit_baselines.py",
                    ["--model", kind, "--samples", n, "--device", "cuda" if kind == "finbert" else "cpu"],
                    "curve",
                )
        self.neural("finbert", "lora", 16000, 512, role="curve")
        # Dataset volume and elapsed fit time inform scheduling; no test data is opened.
        subsets = json.loads((self.splits / "subsets.json").read_text(encoding="utf-8"))["subsets"]
        train = {body_hash(r): r for r in load_split(self.splits, "train")}
        volume = {n: sum(len(train[h]["body"]) for h in subsets[str(n)]) for n in (4000, 16000)}
        record = self.state["jobs"][choices["laya-lora"]]
        history = json.loads((self.output / record["directory"] / "history.json").read_text(encoding="utf-8"))
        elapsed = max(record.get("elapsed_seconds", 0), history["seconds"])
        if elapsed <= 0:
            raise ValueError("Missing measured 4k Laya runtime for the larger-fit gate")
        forecast = elapsed * volume[16000] / volume[4000]
        gate = {
            "forecast_seconds": forecast,
            "margin": 1.3,
            "evaluation_reserve_seconds": 7200,
            "remaining_seconds": self.cutoff - time.time(),
        }
        name = self.optional_laya
        completed = (self.output / "runs" / name / "complete.json").exists()
        previous = self.state["jobs"].get(name, {})
        if not completed and (previous.get("excluded_from_freeze") or previous.get("status") in ("stopped", "oom")):
            previous["excluded_from_freeze"] = True
            status = (
                "infeasible"
                if previous["status"] == "oom"
                else "stopped"
                if previous["status"] == "stopped"
                else self.state["laya_16k_gate"]["status"]
            )
            self.state["laya_16k_gate"] = {
                **gate,
                "status": status,
                "limitation": previous.get("limitation", "Retained excluded partial fit; no automatic retry"),
            }
            save(self.state, self.output / "state.json")
        elif completed or forecast * 1.3 + 7200 <= gate["remaining_seconds"]:
            gate["status"] = "admitted"
            self.state["laya_16k_gate"] = gate
            save(self.state, self.output / "state.json")
            try:
                self.neural("laya", "lora", 16000, 1024, role="curve")
            except (TimeoutError, RuntimeError) as error:
                oom = self.state["jobs"].get(name, {}).get("status") == "oom"
                if not isinstance(error, TimeoutError) and not oom:
                    raise
                self.state["laya_16k_gate"].update(status="infeasible" if oom else "stopped", limitation=str(error))
                if name in self.state["jobs"]:
                    self.state["jobs"][name].update(excluded_from_freeze=True, limitation=str(error))
                self.comparisons.pop(name, None)
                save(self.state, self.output / "state.json")
        else:
            self.state["laya_16k_gate"] = {
                **gate,
                "status": "deferred",
                "reason": "Cannot fit guarded runtime before cutoff",
            }
            if name in self.state["jobs"]:
                if self.state["jobs"][name]["status"] not in ("stopped", "oom"):
                    raise RuntimeError("A failed optional fit needs investigation before freezing")
                self.state["jobs"][name]["excluded_from_freeze"] = True
            save(self.state, self.output / "state.json")
        self.finish_selection(choices)

    def neural(self, kind, adaptation, n, context, objective="ce", weight=0, lr=1e-4, role="core"):
        # Match CLI parsing so integer and float recipes reuse the same job.
        weight = float(weight)
        name = f"{kind}-{adaptation}-{objective}-w{weight}-lr{lr:g}-n{n}-c{context}"
        arguments = [
            "--model",
            kind,
            "--adaptation",
            adaptation,
            "--samples",
            n,
            "--context",
            context,
            "--objective",
            objective,
            "--class-weight-power",
            weight,
            "--learning-rate",
            lr,
            "--epochs",
            3,
            "--window-batch",
            8,
        ]
        if adaptation == "lora":
            arguments += ["--lora-rank", 8]
        return self.job(name, "train.py", arguments, role)

    def recipe(self, name):
        directory = self.output / self.state["jobs"][name]["directory"]
        return json.loads((directory / "config.json").read_text(encoding="utf-8"))

    def finish_selection(self, choices):
        # Preserve matched controls for every final neural context/recipe, once per identical setup.
        controls = {}
        for name, comparison in list(self.comparisons.items()):
            config = self.recipe(name)
            if config["family"] != "research_neural":
                continue
            key = tuple(config[k] for k in ("kind", "adaptation", "max_len", "window_batch"))
            if key not in controls:
                directory = self.output / comparison["directory"]
                controls[key] = self.job(
                    name + "-zero", "research_runtime.py", ["--zero", str(directory)], "control", name
                )
            comparison["matched_zero"] = controls[key]
        save(
            {
                "choices": choices,
                "comparisons": self.comparisons,
                "criterion": "Raw selection macro F1; no test access",
            },
            self.output / "selection.json",
        )

    def freeze(self):
        if (self.output / "freeze.json").exists():
            from evaluate_study import validate_freeze

            validate_freeze(self.splits, self.output)
            return
        selected = json.loads((self.output / "selection.json").read_text(encoding="utf-8"))
        for record in self.state["jobs"].values():
            self.ensure_idle(record)
        if any(
            j["status"] != "completed"
            and not (name == self.optional_laya and j["status"] in ("stopped", "oom") and j.get("excluded_from_freeze"))
            for name, j in self.state["jobs"].items()
        ):
            raise ValueError("Unfinished fit cannot enter the freeze")
        completed = {name for name, record in self.state["jobs"].items() if record["status"] == "completed"}
        if set(selected["comparisons"]) != completed:
            raise ValueError("Selected comparisons must include every completed job and no partial fits")
        if (
            selected["choices"]["tfidf"] != "tfidf-logistic-n4000"
            or selected["choices"]["laya-lora"] != "laya-lora-ce-w0.0-lr0.0001-n4000-c1024"
        ):
            raise ValueError("The primary comparison must remain the fixed 4k TF-IDF/Laya LoRA pair")
        entries, curves = {}, defaultdict(list)
        for name, comparison in selected["comparisons"].items():
            if any(comparison[k] != self.state["jobs"][name][k] for k in ("directory", "role", "control_of")):
                raise ValueError(f"Selected job identity changed: {name}")
            directory = self.output / comparison["directory"]
            config, artifact = verify(directory, self.manifest)
            if self.state["jobs"][name].get("excluded_from_freeze"):
                raise ValueError("An excluded partial fit cannot enter frozen comparisons")
            files = ("config.json", "complete.json", "decision.json", "selection.json", "performance.json", artifact)
            entries[name] = {
                **comparison,
                "sha256": {f: fingerprint(directory / f) for f in files},
                "samples": config.get("samples", 0),
                "kind": config.get("kind", "laya"),
                "family": config["family"],
                "selection_macro_f1": json.loads((directory / "selection.json").read_text())["metrics"]["overall"][
                    "macro_f1"
                ],
            }
            if comparison["role"] in ("core", "curve") and config["family"] in ("research_linear", "research_neural"):
                key = tuple(
                    config.get(k) for k in ("family", "kind", "adaptation", "classifier", "max_len", "loss_kind")
                )
                curves[key].append(name)
        contrasts = {}
        for names in curves.values():
            names.sort(key=lambda n: entries[n]["samples"])
            for smaller, larger in zip(names, names[1:]):
                contrasts[larger + "-sample-increment"] = [larger, smaller]
        save(
            {
                "plan_sha256": fingerprint(self.output / "plan.json"),
                "selection_sha256": fingerprint(self.output / "selection.json"),
                "state_sha256": fingerprint(self.output / "state.json"),
                "manifest": self.manifest,
                "entries": entries,
                "sample_contrasts": contrasts,
                "primary_linear": selected["choices"]["tfidf"],
                "primary_decision": selected["choices"]["laya-lora"],
                "primary_samples": 4000,
                "criterion": "All binary choices fixed before revised-cohort evaluation",
                "utc": datetime.now(timezone.utc).isoformat(),
            },
            self.output / "freeze.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit", "freeze", "test", "all"), default="all")
    args = parser.parse_args()
    with keep_awake():
        study = BinaryStudy(args.splits, args.output, args.session)
        if args.stage in ("fit", "all") and not (args.output / "freeze.json").exists():
            study.fit()
        if args.stage in ("freeze", "all"):
            study.freeze()
        if args.stage in ("test", "all"):
            from evaluate_study import evaluate

            evaluate(args.splits, args.output, deadline=study.cutoff)
