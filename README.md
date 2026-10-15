# Financial document triage

Research on binary financial-document triage using Laya, FinBERT and TF-IDF,
with reproducible evidence and a LaTeX paper. The corpus includes filings,
corporate releases and public announcements, with SEC documents predominant.

Start with the [research guide](decision_models/README.md) or
[paper](decision_models/paper-binary-revised/paper.pdf).

## API

The demonstration API serves the original three-class TF-IDF model trained on
SEC 8-Ks. It is separate from the completed binary research study.

```bash
cp env.example .env  # Only if missing; Windows: copy env.example .env
docker compose up --build
```

Open http://localhost:8000/docs. `GET /health` reports readiness; `POST /predict` accepts:

```json
{"body": "The company entered a definitive merger agreement."}
```

It returns `label` (`routine`, `review_worthy`, `unclear`) and class `probabilities`. It does not extract entities or trim filing sections; callers choose what text to submit.

The image builds a ~6 MB balanced word TF-IDF model from the frozen 8,000-row training split. It uses the completed baseline's validation-selected decision offsets; returned probabilities are uncalibrated and the selected label need not have the highest raw probability. No keys or GPU are needed for this API.

## Local setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Install into your existing Python:

```bash
uv export --frozen --no-dev --no-emit-project | uv pip install --system -r -
python -m decision_models.build_model
uvicorn app:app --reload
```

For collection, labeling and neural experiments:

```bash
uv pip install --system --group models --group encoders --group laya
```

Set `SEC_USER_AGENT` and `OPENAI_API_KEY` in `.env` for collection and labeling. GPU experiments need CUDA-enabled PyTorch. Never commit `.env`. Compose loads it for local convenience; the API itself needs no credentials.

## Research

See the [study guide](decision_models/README.md) for comparisons and
[reproduction instructions](decision_models/reproduce.md) for CPU evidence replay.
References are LLM labels; reported agreement does not establish investment usefulness.

## Development

Both workflows run on every branch push and pull-request update:

- `ci.yml`: check out the repository, install uv, install only the locked lint dependencies, then run Ruff.
- `docker-build.yml`: build the model/API image, start a container, then retry one prediction request until startup succeeds. Endpoint verification runs after the image build.

There is no separate test suite. Update and commit `uv.lock` with dependency changes.
