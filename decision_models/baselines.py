"""Raw-document baselines; encoder features come from encoders.py, never filing-specific cleaning."""

from contextlib import closing
import hashlib
from importlib.metadata import version
import json
import sqlite3
import time

import numpy as np

from schemas import HERE, body_hash

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FITTED_MODELS = ("majority", "tfidf", "finbert")


def make(kind, C=1.0, class_weight=None, max_iter=2000, classifier="logistic"):
    """Return an unfitted pipeline; FinBERT inputs are document vectors, others are raw bodies."""
    if kind not in FITTED_MODELS:
        raise ValueError(f"Unknown baseline: {kind}")
    if kind == "majority":
        return DummyClassifier(strategy="most_frequent")
    if kind == "tfidf":
        features = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
        )
    else:
        features = StandardScaler()
    if classifier != "logistic":
        raise ValueError("The binary study uses logistic regression")
    return make_pipeline(
        features, LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, random_state=42)
    )


def embeddings(kind, records, split, device, details=None):
    import torch
    from encoders import EncoderModel, MODELS

    torch.set_num_threads(4)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": MODELS[kind],
                "cache_format": "document-f32-v1",
                "raw": True,
                "window": 510,
                "pooling": "mean of all windows, then L2",
                "device": device,
                "versions": {n: version(n) for n in ("torch", "transformers", "numpy")},
                "source_sha256": hashlib.sha256((HERE / "encoders.py").read_bytes()).hexdigest(),
            }
        ).encode()
    ).hexdigest()
    cache = HERE / "data/cache" / f"{kind}-documents-{fingerprint[:12]}.sqlite"
    cache.parent.mkdir(parents=True, exist_ok=True)
    vectors, model = [], None
    stats = {
        "documents": len(records),
        "computed": 0,
        "reused": 0,
        "recorded_compute_seconds": 0.0,
        "peak_vram_bytes": 0,
        "model_loading_seconds": 0.0,
        "cache_fingerprint": fingerprint,
    }
    started = time.perf_counter()
    with closing(sqlite3.connect(cache, timeout=30)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS recipe (fingerprint TEXT PRIMARY KEY, width INTEGER)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS features (body TEXT PRIMARY KEY, vector BLOB, sha TEXT, seconds REAL, vram INTEGER)"
        )
        recipe = db.execute("SELECT fingerprint,width FROM recipe").fetchall()
        if recipe and (len(recipe) != 1 or recipe[0][0] != fingerprint):
            raise ValueError(f"Embedding cache differs: {cache}")
        width = recipe[0][1] if recipe else None
        for row in records:
            digest = body_hash(row)
            saved = db.execute("SELECT vector,sha,seconds,vram FROM features WHERE body=?", (digest,)).fetchone()
            if saved:
                blob, checksum, seconds, vram = saved
                if hashlib.sha256(blob).hexdigest() != checksum:
                    raise ValueError(f"Cached feature checksum differs: {digest}")
                vector = np.frombuffer(blob, dtype="<f4").copy()
                stats["reused"] += 1
            else:
                if model is None:
                    loading = time.perf_counter()
                    model = EncoderModel(device, kind=kind).eval()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    stats["model_loading_seconds"] = time.perf_counter() - loading
                    if width is not None and width != model.encoder.config.hidden_size:
                        raise ValueError("Cached feature width differs from the pinned encoder")
                    width = model.encoder.config.hidden_size
                    db.execute("INSERT OR IGNORE INTO recipe VALUES (?,?)", (fingerprint, width))
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                computing = time.perf_counter()
                with torch.no_grad():
                    vector = model.embed(row["body"]).cpu().numpy().astype("<f4")
                seconds = time.perf_counter() - computing
                vram = torch.cuda.max_memory_allocated() if device == "cuda" else 0
                blob = vector.tobytes()
                if vector.ndim != 1 or vector.size != width or not np.isfinite(vector).all():
                    raise ValueError("Invalid encoder feature")
                db.execute(
                    "INSERT INTO features VALUES (?,?,?,?,?)",
                    (digest, blob, hashlib.sha256(blob).hexdigest(), seconds, vram),
                )
                stats["computed"] += 1
                if stats["computed"] % 250 == 0:
                    db.commit()
                    print(
                        f"{kind} {split}: {len(vectors) + 1}/{len(records)}; {stats['computed']} new features",
                        flush=True,
                    )
            if vector.size != width or not np.isfinite(vector).all() or seconds < 0 or vram < 0:
                raise ValueError("Invalid cached encoder feature")
            vectors.append(vector)
            stats["recorded_compute_seconds"] += seconds
            stats["peak_vram_bytes"] = max(stats["peak_vram_bytes"], vram)
    stats["preparation_seconds"] = time.perf_counter() - started
    if details is not None:
        details.update(stats)
    return np.array(vectors)
