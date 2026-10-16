"""Package the frozen primary binary baseline for serving; never fit or read test data."""

import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_model():
    study = HERE / "training_runs/research-binary-v2/study"
    freeze = json.loads((study / "freeze.json").read_text(encoding="utf-8"))
    name = freeze["primary_linear"]
    entry = freeze["entries"][name]
    run = study / entry["directory"].replace("\\", "/")
    for filename in ("config.json", "decision.json", "complete.json", "model.joblib"):
        if sha(run / filename) != entry["sha256"][filename]:
            raise ValueError(f"Frozen serving input changed: {filename}")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    if (config["task_id"], config["kind"], config["classifier"]) != ("binary-v2", "tfidf", "logistic"):
        raise ValueError("Serving requires the frozen binary TF-IDF/logistic baseline")
    output = HERE / "serving"
    output.mkdir(exist_ok=True)
    shutil.copyfile(run / "model.joblib", output / "model.joblib")
    metadata = {
        "task_id": config["task_id"], "model": name, "labels": config["labels"],
        "training_documents": config["samples"], "temperature": decision["temperature"],
        "policy": {"name": "calibration_miss_1pct", "routine_threshold": decision["policies"]["0.01"]["threshold"],
                   "calibration_miss_target": .01, "population_guarantee": False},
        "versions": config["versions"], "model_sha256": sha(output / "model.joblib"),
        "freeze_sha256": sha(study / "freeze.json"),
        "config_sha256": sha(run / "config.json"), "decision_sha256": sha(run / "decision.json"),
        "references": "Luna teacher labels; not analyst-validated financial materiality",
    }
    (output / "manifest.json").write_bytes((json.dumps(metadata, indent=2) + "\n").encode())
    print(f"Packaged {name}: {(output / 'model.joblib').stat().st_size:,} bytes; no fitting performed")


if __name__ == "__main__":
    build_model()
