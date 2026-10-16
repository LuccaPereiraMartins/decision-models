# Financial-document triage research

This study compares decision models, a finance encoder and a lexical classifier
for binary document triage: `routine` or `review_worthy`. Labels come from Luna;
scores measure teacher agreement, not analyst-validated financial materiality.

## Start here

- [Paper](paper-binary-revised/paper.pdf): the concise editorial edition. Read the
  results for the system comparison, missed references and review workload.
- [Binary evidence](reports/binary-v2/README.md): predictions, fitting records,
  resource measurements and independent CPU replay for all 16 frozen comparisons.
- [FinBERT 25k extension](reports/binary-finbert25k/README.md): subsequent
  development-only results, separate from the paper's test comparisons.
- [Reproduction](reproduce.md): metric replay, paper compilation and the additional
  inputs required to reproduce model fitting.

## Study design

The corpus contains 25,000 training, 1,000 selection, 500 calibration and 2,000
test documents. Training ends in March 2026; development covers April–June and
test July–September. Related-document groups cannot cross cohorts. Sources include
SEC filings and release exhibits, central-bank releases, GOV.UK and Wikinews;
SEC supplies 1,870 of the 2,000 test documents.

Previously unclear references received new binary decisions. Definite references
were retained, with a separate 200-document training-only consistency audit.
The test retains 1,800 previously inspected bodies and adds 200; it is not a fresh
independent confirmation. See the [protocol](binary-study.md) and
[source audit](sources.md) for provenance and limitations.

| System | Training documents |
| --- | --- |
| Laya native SDK | No task fitting |
| Laya head | 4,000 |
| Laya LoRA | 4,000 and 16,000 |
| Frozen FinBERT + logistic regression | 4,000, 16,000 and 25,000 |
| FinBERT LoRA | 4,000 and 16,000 |
| Word TF-IDF + logistic regression | 4,000, 16,000 and 25,000 |
| Majority | Counts the shared 4,000 training labels; no optimization |

The primary comparison uses the same 4,000 training documents for Laya LoRA and
TF-IDF/logistic. Larger subsets measure sample efficiency. Three matched initial
states support adaptation comparisons. Selection chooses recipes/checkpoints;
calibration fits temperatures and routing thresholds; all choices precede test.

Macro F1 is the primary metric. False negatives, review recall, skipped documents
and processing time describe the operational tradeoff. Bootstrap intervals are
conditional on one fitted run. The subsequent FinBERT 25k fit uses development
cohorts only because it was chosen after the test results were inspected.

## Code map

| Stage | Files |
| --- | --- |
| Acquisition and provenance | `collect_sources.py`, `legacy_events.py` |
| Grouping, binary preparation and labels | `prepare.py`, `prepare_binary.py`, `label.py`, `schemas.py` |
| Models and fitting | `baselines.py`, `fit_baselines.py`, `encoders.py`, `train.py`, `decision_base.py` |
| Metrics, calibration and inference | `protocol.py`, `research_runtime.py` |
| Bounded execution and frozen evaluation | `binary_pipeline.py`, `study_binary.py`, `study.py`, `evaluate_study.py` |
| Evidence and original paper generation | `export_binary.py`, `binary_replay.py`, `report_binary.py` |
| Separate 25k development extension | `run_finbert25k.py`, `finbert25k_evidence.py` |
| Editorial paper packaging | `build_paper.py` |

The original producer/evaluator sources and published packages are byte-bound
to completed runs. Keep their snapshots for exact replay. Do not remove ownership
markers or rewrite run metadata to make a changed implementation pass verification.
The public verifiers replay saved evidence without training or opening local test
bodies. Collection, labelling and training entry points are not replay commands.

## Historical work

The [three-class guide](historical-guide.md) documents the original API benchmark
and expanded experiment grid. The [core report](reports/core-v1/README.md),
[original binary paper](paper-binary-v2/README.md), [execution checkpoints](binary-checkpoint.md)
and [planning log](research.md) preserve earlier stages. Their dated plans and
runtime estimates are historical; the completed binary study is the current result.

Retired tools (`collect.py`, `split.py`, `evaluate.py`, `research_pipeline.py`,
`review.py` and the toy `laya_infer.py`) are available in the
[last revision before removal](https://github.com/LuccaPereiraMartins/systematic-trading/tree/7b36eadb79daeaa6583643792a4e29afa957cefc/decision_models).
They are superseded by the mixed-source collector, binary preparation and bounded
study, or apply only to the old three-class pilot. Shared training/runtime code
and the independent verifiers remain required even after a study has completed.

The repository's [API](../README.md#api) serves the frozen primary binary TF-IDF
baseline with its calibration-fitted routing policy. Its separate
[serving package](serving/README.md) contains the classifier required by the image.
