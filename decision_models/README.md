# Financial-document triage

Binary routing of financial documents to `routine` or `review_worthy`, comparing
Laya, FinBERT and a lexical classifier. Luna supplies the reference labels: these
scores measure teacher agreement, not analyst-validated financial materiality.

- [Paper](paper/paper.pdf): six main pages, references and one supporting page.
- [Core evidence](reports/binary-v2/README.md): all 16 frozen comparisons, predictions,
  fitting records, measurements and independent CPU replay.
- [FinBERT 25k development](reports/binary-finbert25k/README.md): the original
  fitting evidence, preserved separately.
- [25k LoRA comparisons](reports/lora25k/README.md): completed FinBERT and
  Laya reused-test evaluations, paired intervals and independent CPU replay.
- [Sources](sources.md): acquisition, reuse conditions and provenance limitations.

## Reading the implementation

Start with `baselines.make` and `fit_baselines.fit`: fit features on training
documents, choose a logistic classifier on selection, then calibrate separately.
`protocol.py` defines the common measurements and routing policy. Read
`encoders.py` and `train.py` next for neural windows and document-level fitting.
`evaluate_study.py` applies the frozen choices and computes paired intervals.
Collection, historical identity recovery and scheduling are supporting machinery;
they are not needed to understand the classifier or replay the published arithmetic.

For each document, the algorithm is:

1. Produce two logits, ordered as routine and review-worthy.
2. Divide by the calibration-fitted temperature and apply softmax.
3. Predict review-worthy at probability 0.5 or above, including ties.
4. Skip only a routine prediction whose routine probability reaches the saved threshold.

`fit_policies` checks whole probability ties and selects the largest feasible
calibration skip count. Miss targets divide by review-worthy references;
skip budgets divide by all documents. These are document counts, not measured
reading time, economic losses or guarantees about future data.

## Design and interpretation

The frozen corpus has 25,000 training, 1,000 selection, 500 calibration and 2,000
test documents. Training ends in March 2026, development covers April-June and test
July-September. Related-document groups cannot cross cohorts. SEC supplies
1,870/2,000 test documents; the other public sources provide limited coverage.

Previously unclear references received binary decisions; definite references were
retained. A balanced 200-document training audit found 93% teacher consistency.
The revised test reuses 1,800 previously inspected bodies and adds 200, so it is
not an independent confirmation on fresh documents. The paper states these limits.

| System | Training documents |
| --- | --- |
| Laya native SDK | No task fitting |
| Laya head | 4,000 |
| Laya LoRA | 4,000, 16,000 and 25,000 |
| Frozen FinBERT + logistic | 4,000, 16,000 and 25,000 |
| FinBERT LoRA | 4,000, 16,000 and 25,000 |
| Word TF-IDF + logistic | 4,000, 16,000 and 25,000 |
| Majority | Counts the shared 4,000 labels; no optimization |

Three matched pre-update controls complete the 16 comparisons. The primary pair
is Laya LoRA versus TF-IDF/logistic at 4k. Macro F1 is primary; false negatives,
review recall, skipped reading and processing time quantify the routing tradeoff.
Selection chooses recipes/checkpoints; calibration fits temperatures and thresholds.
The 25k LoRA runs were planned but completed after the initial report because of
training time. Their recipes were fixed in advance without using prior test results.
All sizes reuse the same frozen test cohort. Bootstrap intervals condition on one
fitted run.
The two planned 25k LoRA fits bring the complete study to 18 scored configurations;
their separate evidence package preserves the original freeze unchanged.

## Reproduce the published evidence

From the repository root, using an isolated Python environment with only NumPy:

```sh
uv run --isolated --no-project --python 3.13 --with numpy==2.5.3 python decision_models/reports/binary-v2/verify.py
uv run --isolated --no-project --python 3.13 --with numpy==2.5.3 python decision_models/reports/binary-finbert25k/verify.py
uv run --isolated --no-project --python 3.13 --with numpy==2.5.3 python decision_models/reports/lora25k/verify.py
```

These independent verifiers check saved predictions, metrics, checkpoint selection,
calibration, policies, resource arithmetic and package hashes. They use no bodies,
model weights, GPU, network or API. They do not rerun model inference.
The first `uv` invocation may download Python or NumPy; the verifier itself is offline.
If NumPy is already installed, invoking any script directly with `python` works too.

To regenerate the paper's numerical tables and figures, then compile its source:

```sh
uv sync --frozen --python 3.13 --group reports
uv run --no-sync python decision_models/report.py
uv run --no-sync python decision_models/build_paper.py --engine tectonic
```

Rendering needs the `reports` dependency group; Tectonic 0.17.0 is the tested engine.
The renderer preserves editorial prose. The builder checks protected original inputs
and reconstructs numerical tables/figures from verified original and additional
evidence, compiles in a clean directory and produces
[arxiv-source.zip](paper/arxiv-source.zip). Inspect rendered pages after editing.
See [paper/README.md](paper/README.md) for a standalone archive build.

## Models, inputs and exact fitting replay

Public evidence excludes source bodies and neural weights. The local frozen inputs
are `data/research/binary-v2/{train,selection,calibration,test}.jsonl`, `manifest.json`
and `subsets.json`. The six-file `benchmark-inputs.zip` has an exact
[receipt](reports/binary-v2/input-archive.json); acquiring today's pages is not an
exact replacement for these snapshots. Preserve attribution with transferred inputs.

Selected `best.pt`, classifier `model.joblib` and native `base.json` artifacts remain
in `training_runs/`. Redundant `last.pt` and `epoch-zero.pt` were removed only from
completed runs after checking selected-artifact hashes and finite weights. Unfinished
runs retain resumable checkpoints. Pre-update controls retain their selected models.
Historical audits in the evidence describe the checkpoints available at audit time;
they are records, not a claim that deleted intermediate files can still be reopened.

Each evidence package's `source.zip` preserves the original producer/runtime code.
For exact original fitting or inference, extract it to a separate directory within this Git checkout,
use the recorded dependency versions and pinned pretrained revisions, and supply the
retained inputs/models. Original configurations bind source hashes; the refactored
active code deliberately fails their source checks. Never rewrite those bindings or
clear test ownership markers to reuse an evaluated cohort. New comparisons require
fresh held-out data and new run directories. Recreated fits need not reproduce exact
floating-point weights across hardware or software changes.

## Active code

| Responsibility | Files |
| --- | --- |
| Acquisition and legacy identity recovery | `collect_sources.py`, `legacy_events.py` |
| Binary preparation, grouping and labels | `prepare.py`, `label.py`, `schemas.py` |
| Classical models and frozen features | `baselines.py`, `fit_baselines.py` |
| Laya/FinBERT models and fitting | `encoders.py`, `train.py`, `decision_base.py` |
| Metrics, calibration and verified inference | `protocol.py`, `research_runtime.py` |
| Bounded study and frozen evaluation | `study.py`, `evaluate_study.py` |
| Paper rendering and packaging | `report.py`, `build_paper.py` |
| Frozen CPU serving package | `build_model.py`, `serving/` |
| Exploratory tabular foundation-model heads | `tabular.py`, `fit_tabular.py`, `tabular_smoke.py` |

Collection, labelling and fitting are separate from evidence replay. Their CLIs
explain required paths with `--help`. `study.py` requires a session JSON containing
`compute_cutoff_utc`; it serializes jobs and reserves evaluation time. Fitting never
opens test. `evaluate_study.py` validates the freeze and persistent cohort ownership
before evaluation. Source closures are captured for each new run.

The binary rubric and durable Luna ledger retain a $2 incremental cap; restarting
cannot reset it. No replay command calls Luna. Setup and the CPU API are documented
in the [repository README](../README.md).

Superseded three-class code, execution diaries, old paper editions and their data
are available in [Git history](https://github.com/LuccaPereiraMartins/systematic-trading/tree/cbe5c5dd0424e71ee9a6199e311adc659885b86c/decision_models).
The current tree keeps one paper, the original binary evidence and its two extensions.

## Exploratory: tabular foundation-model heads

`fit_tabular.py` replaces the logistic head of the frozen baselines with a pretrained
in-context tabular model from NVIDIA [structured-data-models](https://github.com/NVIDIA/structured-data-models)
(`kumo-small|medium|large`, open `nvidia/Kumo-Tabular` weights, or `tabiclv2`). Document
features are cached FinBERT vectors (PCA) and/or word TF-IDF (truncated SVD), each reduced
on training rows only. Training rows are the model's context, averaged over context bags
when they exceed `--context-size`. Selection chooses the reduced width and any fine-tuning
checkpoint, while calibration fits temperature and policies as in `protocol.py`.

These runs are **development evidence only**: the script never loads test, and a
generalization claim needs fresh held-out data. The development references are frozen
FinBERT/logistic selection macro F1 0.7580/0.7869/0.8002 and TF-IDF/logistic
0.7825/0.8152/0.8313 at 4k/16k/25k. Selection also chooses the TFM width, so compare
the calibration-set F1 in `calibration.json` as well.
Verified CUDA results and CPU arithmetic replay are in
[tabular development evidence](reports/tabular-development/README.md).

```sh
uv sync --frozen --python 3.13 --group models --group encoders --group tabular
# Laptop smoke test on synthetic, non-evidential documents (CPU or Apple MPS):
uv run --no-sync python decision_models/tabular_smoke.py --output /tmp/tabular-smoke
uv run --no-sync python decision_models/fit_tabular.py --splits /tmp/tabular-smoke --output /tmp/tabular-run \
  --samples 300 --model tabiclv2 --device mps --dimensions 16 32 --context-size 512 --num-estimators 2
# Frozen corpus on CUDA; repeat for --samples 16000/25000, --features tfidf or "finbert tfidf":
uv run --no-sync python decision_models/fit_tabular.py --splits decision_models/data/research/binary-v2 \
  --output decision_models/training_runs/tabular/kumo-large-finbert-n4000 \
  --samples 4000 --model kumo-large --features finbert --device cuda \
  --dimensions 32 64 96 --context-size 8192 --num-estimators 8
```

Add `--finetune-steps 300 --finetune-every 50` for full fine-tuning on context/query
episodes sampled from training rows. Its default learning rate is 1e-5, following the
sdm example. Step 0 is the pretrained model, so selection can keep it. Each run saves
`reducer.joblib`, `context.npz` (reduced training rows and labels), optional `finetuned.pt`,
and the usual config, search, decision, calibration, selection and performance files.
Fine-tuning uses bf16 CUDA, finite-loss/gradient checks and gradient clipping at 1.
`--finetune-max-seconds 900` bounds the fit and rejects an excessive five-step
forecast. `finetune-last.pt` preserves model/best weights, optimizer and RNG state
at selection checkpoints or a budget stop; automatic resume is not implemented.

There are two hardware limits. With sdm 0.2.1, pretrained KumoTabular returns constant
probabilities on MPS, so Kumo runs on CPU on a Mac; TabICLv2 works on MPS. FinBERT
vectors for the long real documents are the expensive step: the sqlite cache shares them
across runs, so compute them once with `--encoder-device cuda`. The hosted Kumo Tabular
NIM, `nvcr.io/nim/nvidia/kumo-tabular`, needs a Linux NVIDIA GPU and is not wired in here.
