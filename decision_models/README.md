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

## Design and interpretation

The frozen corpus has 25,000 training, 1,000 selection, 500 calibration and 2,000
test documents. Training ends in March 2026, development covers April–June and test
July–September. Related-document groups cannot cross cohorts. SEC supplies
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

## Reproduce the published evidence

From the repository root, with Python and NumPy installed:

```sh
python decision_models/reports/binary-v2/verify.py
python decision_models/reports/binary-finbert25k/verify.py
python decision_models/reports/lora25k/verify.py
```

These independent verifiers check saved predictions, metrics, checkpoint selection,
calibration, policies, resource arithmetic and package hashes. They use no bodies,
model weights, GPU, network or API. They do not rerun model inference.

To regenerate the paper's numerical tables and figures, then compile its source:

```sh
python decision_models/report.py
python decision_models/build_paper.py --engine tectonic
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
