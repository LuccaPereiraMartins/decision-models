"""Tabular foundation-model heads on frozen document features (NVIDIA structured-data-models)."""

import copy
import hashlib
import math

import numpy as np

from schemas import LABELS


# name -> (sdm class, constructor kwargs); weights are pinned by the installed sdm release.
MODELS = {
    "kumo-small": ("KumoTabular", {"size": "small"}),
    "kumo-medium": ("KumoTabular", {"size": "medium"}),
    "kumo-large": ("KumoTabular", {"size": "large"}),
    "tabiclv2": ("TabICLv2", {}),
}
FEATURES = ("finbert", "tfidf")


def reducer(kind, dimensions, seed=42):
    """Fit only on training rows: PCA for dense FinBERT vectors, truncated SVD for sparse TF-IDF."""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "finbert":
        return make_pipeline(StandardScaler(), PCA(dimensions, random_state=seed))
    if kind == "tfidf":
        from baselines import make

        return make_pipeline(make("tfidf")[0], TruncatedSVD(dimensions, random_state=seed), StandardScaler())
    raise ValueError(f"Unknown tabular feature: {kind}")


class FeatureReducer:
    """Concatenate one reduced block per feature kind; each block is fitted on the context rows."""

    def __init__(self, kinds, dimensions, seed=42):
        self.kinds, self.dimensions = tuple(kinds), dimensions
        self.blocks = {kind: reducer(kind, dimensions, seed) for kind in self.kinds}

    def fit_transform(self, inputs):
        return np.hstack([self.blocks[k].fit_transform(inputs[k]) for k in self.kinds]).astype("float32")

    def transform(self, inputs):
        return np.hstack([self.blocks[k].transform(inputs[k]) for k in self.kinds]).astype("float32")


def _table(features, labels=None, device="cpu"):
    """Feature table, or (features, target) tables when labels are supplied."""
    import pandas as pd
    import sdm

    frame = pd.DataFrame(features, columns=[f"x{i}" for i in range(features.shape[1])])
    overrides = {}
    if labels is not None:
        frame["target"], overrides["target"] = [str(label) for label in labels], "categorical"
    table = sdm.TableTensor.from_pandas(frame, stypes=sdm.infer_stypes(frame, overrides=overrides), device=device)
    return table if labels is None else (table.drop_columns("target"), table[:, "target"])


class TabularModel:
    """In-context classifier; context rows play the role of training data, nothing is fitted unless fine-tuned."""

    def __init__(self, name, device="cpu", labels=LABELS):
        import sdm
        import torch

        if name.startswith("kumo") and device == "mps":
            # Measured on sdm 0.2.1: pretrained KumoTabular returns constant probabilities on MPS (float64 recipe).
            raise ValueError("KumoTabular is unreliable on MPS; use --device cpu (or cuda)")
        cls, kwargs = MODELS[name]
        self.name, self.labels, self.device = name, tuple(labels), torch.device(device)
        self.model = getattr(sdm.models, cls)(task="classification", device=device, **kwargs).eval()
        digest = hashlib.sha256()
        for key, value in sorted(self.model.state_dict().items()):
            digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.detach().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        self.pretrained_sha256 = digest.hexdigest()

    def autocast(self):
        import torch

        return torch.amp.autocast(self.device.type, torch.float16, enabled=self.device.type == "cuda")

    def _probabilities(self, output):
        frame = output.to_pandas()
        frame.columns = [str(c) for c in frame.columns]
        # Read by class name: sdm orders output columns independently of the task's label order.
        if set(frame.columns) != set(self.labels):
            raise ValueError("Tabular model returned different task classes")
        p = np.column_stack([frame[c].to_numpy() for c in self.labels])
        if not np.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any():
            raise ValueError("Tabular model returned invalid class probabilities")
        return p / p.sum(1, keepdims=True)

    def contexts(self, n, context_size, bags, seed):
        """Disjoint-as-possible random context subsets; one bag uses every row when it fits."""
        rng = np.random.default_rng(seed)
        if n <= context_size:
            return [np.arange(n)] * (bags or 1)
        bags = bags or math.ceil(n / context_size)
        order = np.concatenate([rng.permutation(n) for _ in range(math.ceil(bags * context_size / n))])
        return [np.sort(order[b * context_size : (b + 1) * context_size]) for b in range(bags)]

    def log_probabilities(
        self, x_context, y_context, x_query, *, context_size=4096, bags=None, num_estimators=4, batch=512, seed=42
    ):
        """Average class probabilities over context bags, return clipped log-probabilities as logits."""
        import torch

        self.model.eval()
        y_context, total = np.asarray(y_context), np.zeros((len(x_query), len(self.labels)))
        subsets = self.contexts(len(x_context), context_size, bags, seed)
        for bag, rows in enumerate(subsets):
            print(f"Context {bag + 1}/{len(subsets)}: {len(rows)} training rows, {len(x_query)} queries", flush=True)
            generator = torch.Generator(device=self.device).manual_seed(seed + bag)
            with self.autocast():
                x, y = _table(x_context[rows], y_context[rows], self.device)
                self.model.fit(
                    x=x,
                    y=y,
                    num_estimators=num_estimators,
                    generator=generator,
                )
                for start in range(0, len(x_query), batch):
                    output = self.model.predict(_table(x_query[start : start + batch], device=self.device))
                    total[start : start + batch] += self._probabilities(output)
            self.model.clear()
        return np.log(np.maximum(total / len(subsets), 1e-12))

    def finetune(self, x, y, evaluate, *, steps, every, context_size=512, query_size=256, lr=1e-5, seed=42,
                 max_seconds=None, checkpoint=None):
        """Full fine-tuning on episodes sampled from training rows; keep the best state by `evaluate()`.

        Step 0 is the pretrained state, so selection can always fall back to no fine-tuning.
        """
        import sdm
        import sdm.processing as sp
        import time
        import torch
        import torch.nn.functional as F

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        recipe = self.model.default_recipe()
        recipe.target = sp.StypeDispatch(
            categorical=[sp.AlignCategories(), sp.ShuffleCategories(method="shift")], numerical=sp.Standardize()
        )
        recipe.output = sp.Identity()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        y = np.asarray(y)
        started = time.perf_counter()
        history = [{"step": 0, "score": evaluate()}]
        evaluation_seconds = time.perf_counter() - started
        best, best_state = history[0]["score"], copy.deepcopy(self.model.state_dict())
        losses = []

        def preserve(step):
            if checkpoint:
                checkpoint({"step": step, "weights": self.model.state_dict(), "best_weights": best_state,
                            "optimizer": optimizer.state_dict(), "history": history, "losses": losses,
                            "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []})

        preserve(0)
        training_seconds = 0.
        for step in range(1, steps + 1):
            training_started = time.perf_counter()
            self.model.train()
            index = rng.permutation(len(x))[: context_size + query_size]
            context, query = index[:context_size], index[context_size:]
            x_context, y_context = _table(x[context], y[context], self.device)
            x_query, target = _table(x[query], y[query], self.device)
            optimizer.zero_grad()
            with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"):
                output = self.model(x_context=x_context, y_context=y_context, x_query=x_query, recipe=recipe)[0]
                logits, truth = sdm.evaluation.to_class_indices(output, target, missing_score=-torch.inf)
                loss = F.cross_entropy(logits.float(), truth)
            if not torch.isfinite(loss):
                preserve(step - 1)
                raise ValueError("Non-finite tabular fine-tuning loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            training_seconds += time.perf_counter() - training_started
            if max_seconds is not None:
                elapsed = time.perf_counter() - started
                remaining = training_seconds / step * (steps - step) + math.ceil((steps - step) / every) * evaluation_seconds
                if step == min(5, steps):
                    print(f"Fine-tuning forecast: {elapsed + remaining:.1f}s for {steps} steps; budget {max_seconds:g}s", flush=True)
                if elapsed >= max_seconds or (step == min(5, steps) and elapsed + remaining > max_seconds):
                    preserve(step)
                    raise TimeoutError(f"Fine-tuning exceeds {max_seconds:g}s budget; checkpoint retained at step {step}")
            if step % every == 0 or step == steps:
                score = evaluate()
                history.append({"step": step, "score": score, "train_loss": float(np.mean(losses))})
                print(history[-1], flush=True)
                losses = []
                if score > best:
                    best, best_state = score, copy.deepcopy(self.model.state_dict())
                preserve(step)
        self.model.load_state_dict(best_state)
        self.model.eval()
        return history, max(history, key=lambda h: (h["score"], -h["step"]))["step"]
