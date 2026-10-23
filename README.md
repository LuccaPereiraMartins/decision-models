# Financial document triage

Research on binary financial-document triage using Laya, FinBERT and TF-IDF,
with reproducible evidence and a LaTeX paper. The corpus includes filings,
corporate releases and public announcements, with SEC documents predominant.

Start with the [research guide](decision_models/README.md) or
[paper](decision_models/paper/paper.pdf).

The main result is a tradeoff: at 4k training documents, Laya LoRA's macro F1
advantage over TF-IDF/logistic is uncertain, while it skips more documents,
misses more review references and takes longer to run. The
[critical review](decision_models/REVIEW.md) separates verified findings,
implementation fixes and questions that need new evidence.

## API

The CPU image serves the frozen **binary TF-IDF/logistic baseline** and its
calibration-fitted routing policy. It needs no keys, GPU or private corpus:

```bash
docker compose up --build
```

Open http://localhost:8000/docs. `GET /health` identifies the model and policy;
`POST /predict` accepts:

```json
{"body": "The company entered a definitive merger agreement."}
```

The response includes `label` (`routine` or `review_worthy`), calibrated class
`probabilities`, and an `action` (`review` or `skip`). A routine prediction can
still go to review: skipping requires the saved threshold for the empirical
1% calibration miss target. That fitting target is not a population guarantee.

The image copies the approximately 4.9 MB classifier from the completed 4k fit;
it never retrains during a build. The [serving manifest](decision_models/serving/manifest.json)
binds the artifact, calibration and dependency versions. The Docker context is an
allowlist of inference code, the model/manifest and dependency files. It excludes
corpora, checkpoints, research scripts, papers, credentials and Git history.

## Local setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/); Python 3.13 matches
the serving image. Create an isolated environment from the committed lockfile:

```bash
uv sync --frozen --python 3.13
uv run --no-sync uvicorn app:app --reload
```

This also supports metrics and the TF-IDF/majority fitting CLI without PyTorch,
model downloads or collection packages. For collection, labeling and neural experiments:

```bash
uv sync --frozen --python 3.13 --group models --group encoders --group laya
```

Set `SEC_USER_AGENT` and `OPENAI_API_KEY` in a local `.env` for collection and
labelling only; copy `env.example` if needed. GPU experiments need CUDA-enabled
PyTorch. Never commit `.env`. Compose does not load research credentials.

## Research

See the [study guide](decision_models/README.md) for comparisons and
[reproduction instructions](decision_models/README.md#reproduce-the-published-evidence) for CPU evidence replay.
References are LLM labels; reported agreement does not establish investment usefulness.

## Development

Both workflows run on every branch push and pull-request update:

- `ci.yml`: run Ruff and independently replay both published binary evidence packages on CPU.
- `docker-build.yml`: build the inference image, verify its exact application-file allowlist and lack of neural/API dependencies, then check binary predictions, probabilities and routing.

Update and commit `uv.lock` with dependency changes.
