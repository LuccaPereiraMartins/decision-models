"""CPU inference for the frozen binary TF-IDF baseline and calibration policy."""

from contextlib import asynccontextmanager
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


Label = Literal["routine", "review_worthy"]
MODEL_DIRECTORY = Path(__file__).resolve().parent / "decision_models/serving"


class Document(BaseModel):
    body: str = Field(min_length=1)


class Prediction(BaseModel):
    label: Label
    probabilities: dict[Label, float]
    action: Literal["review", "skip"]


@asynccontextmanager
async def lifespan(app):
    metadata = json.loads((MODEL_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    artifact = MODEL_DIRECTORY / "model.joblib"
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Serving model checksum mismatch")
    for package, expected in metadata["versions"].items():
        if version(package) != expected:
            raise ValueError(f"Serving runtime differs from the frozen model: {package}")
    model = joblib.load(artifact)
    if set(model.classes_) != {"routine", "review_worthy"} or metadata["task_id"] != "binary-v2":
        raise ValueError("The API requires the binary triage model")
    app.state.model, app.state.metadata = model, metadata
    yield


app = FastAPI(title="Financial document triage", lifespan=lifespan)


@app.get("/health")
def health():
    metadata = app.state.metadata
    return {"status": "ok", "model": metadata["model"], "policy": metadata["policy"]["name"]}


@app.post("/predict", response_model=Prediction)
def predict(document: Document):
    model, metadata = app.state.model, app.state.metadata
    logits = np.log(np.maximum(model.predict_proba([document.body])[0], 1e-12)) / metadata["temperature"]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    probabilities = dict(zip(model.classes_, probabilities.tolist()))
    label = "review_worthy" if probabilities["review_worthy"] >= .5 else "routine"
    skip = label == "routine" and probabilities["routine"] >= metadata["policy"]["routine_threshold"]
    return {"label": label, "probabilities": probabilities, "action": "skip" if skip else "review"}
