# Reproducing the bounded study

This procedure concerns the frozen mixed-source benchmark and the [bounded core study](core-study.md). The older 10k archive and its previously opened test are a separate experiment. Reproduction needs preserved artifacts; a fresh live collection cannot reconstruct the exact historical corpus.

The [published core package](reports/core-v1/README.md) supplies all held-out logits, teacher references, fixed decisions,
metrics, shared group intervals and execution/source evidence. `python decision_models/reports/core-v1/verify.py`
replays the metrics/statistics, exploratory appendix and separate development evidence on CPU, without API calls, weights or document bodies. Full training/inference replay
still needs the inputs below. Published-file byte hashes are preserved across Windows/Linux by the package's attributes.

## Required inputs

| Artifact | Purpose |
| --- | --- |
| Frozen benchmark directory | Train, selection, calibration and test JSONL; manifest; ordered nested subsets |
| `benchmark-inputs.zip` | Exact frozen input files for a clean benchmark installation; no execution-state markers |
| Original `research-study` directory | Completed baseline/head/reference/1k LoRA runs, their metadata and weights |
| `research-core-v1` directory | Three new LoRA fits, controls, plan/state, eventual freeze/test/report evidence |
| Core `source-snapshot.zip` | Exact bytes of 27 source/dependency files plus the initial core plan |
| Original source archive | Exact original source/dependency bytes and original plan/state, preserved before updating the checkout |
| Pinned pretrained weights and Python environment | Model revisions and package versions recorded in each run's `config.json` |
| Report `report-source.py` and manifest | Exact renderer bytes and output hashes, including any later presentation fixes |
| Report `diagnostic-inputs.json` and `diagnostics-source.py` | Optional recorded post-hoc scope and calculation source for the exploratory appendix |
| Report `development-evidence.json.gz` and `development-source.py` | Optional separate FinBERT 4k/16k development inputs and descriptive calculation source |
| Report `compact-evidence.json.gz` and `compact-source.py` | Optional matched-4k BGE/FinBERT development inputs and descriptive calculation source |

Keep the two study directories together. Core comparison paths reference original run directories rather than duplicate large weights. The core plan binds the original plan/state hashes; completed models also bind source bytes, versions, training-subset identities, labels and the split manifest. Each neural run retains its own source snapshots and pre-update/best/final checkpoints.

Local collection/label artifacts, credentials and checkpoints are ignored by Git. A Git clone alone is therefore not an exact-data reproduction package. Source rights/attribution are in [sources.md](sources.md); preserve that information with any corpus transfer. No credentials are required to replay local model inference from downloaded artifacts.

## Exact replay and independent rerun

For an exact replay in the original paths and pinned environment, use the unchanged core command from [README.md](README.md#bounded-core-study). It reuses verified completed jobs. Once frozen, `--stage all` skips fitting and resumes test/report against the identical freeze. Completed test-result hashes are checked before reuse. `--stage report` rebuilds presentation from verified predictions, without changing model choices or dispatching paid calls.

To include the exploratory appendix, copy the published `reports/core-v1/diagnostic-inputs.json` into the preserved core run's `report/` directory before that report command. The renderer checks its freeze and primary/reference identities and its ordered selection character counts against the original selection file. Preserve the published renderer and `diagnostics.py` bytes for exact analysis-source replay. Without this optional input, only the original core sections are generated. Diagnostics use existing predictions and policies; they cannot select a new model or threshold.

To also include the separate FinBERT development section, preserve `reports/core-v1/development-evidence.json.gz`
in the core run's `report/` directory and the published `development.py` source. The renderer checks the original
freeze identity, matching recipes, nested training identities and shared development cohorts. This probability-based
report replay does not evaluate a new checkpoint on test. For exact 16k training replay, preserve the extension
run/controller/source/configuration artifacts and the existing benchmark inputs. Train only this declared recipe
in a separate development output; selection/calibration are fitting cohorts, and any new held-out evaluation
requires fresh data. Do not reopen the completed 64-comparison runner or reset ownership markers.

For the compact section, also preserve `reports/core-v1/compact-evidence.json.gz` in the report directory
and its recorded `compact.py` source. The renderer checks the shared 4k training identities, matching
neural recipes and canonical development cohorts; original prediction ordering remains inspectable.
Public CPU replay additionally checks every metric, probability/temperature identity, calibrated policy
and original calibration cohort hash. Full BGE replay requires the compact run's pinned weights/configuration
and local corpus. Best and matched epoch-zero checkpoint replay are restricted to development cohorts.

For a relocated replay, the constructor's recorded absolute reuse path needs the original directory layout. Do not edit frozen plan/state/checkpoint metadata to make a moved archive pass. An independent reproduction needs a clean installation of the preserved benchmark inputs, a fresh core output, and the relocated completed original study passed via `--reuse`, with exact source/runtime bytes. This is a new fit, not continuation of the evaluated study; reproduce the declared methods and keep the result separate. Do not use reported test outcomes to choose further recipes. The original benchmark's ownership marker stays preserved and rejects further fitting there; never delete it to continue experimentation.

Sources use recorded byte hashes, including line endings. A default checkout on another OS may change those bytes. Restore the preserved source files into an isolated replay checkout and check the recorded hashes. Use the recorded CUDA-enabled Torch build; installing the PyPI version alone previously selected CPU-only Torch on this machine. Do not replace dependencies in an active research environment. Model revisions are pinned, not the latest Hub revision.

## Evidence to check

1. Benchmark manifest and nested subset hashes match the plan. Train ends March 2026; selection/calibration are April–June; the held-out cohort is July–September. Related events and verified near duplicates do not cross partitions.
2. All completed imported/new runs pass `research_runtime.verify`. New recipes are exactly Laya/ModernBERT/FinBERT LoRA at 4,000 documents, with seed 42, rank 8 and at most three epochs. No larger or generative adaptation is silently included.
3. The original 4k head and the new 4k adaptations use the identical ordered training subset. Selection has 837 documents; calibration has 363. Check best-epoch choice, finite checkpoint tensors, preserved epoch-zero weights, confusion matrices, F1 arithmetic and critical-miss counts.
4. The freeze binds the plan, selection, state, model artifacts and decisions before any held-out inference. Both primary methods and main trained-model tables use 4,000 documents. Larger curves are supplementary. The benchmark's holdout ownership marker matches that freeze.
5. Every test result binds the same freeze and ordered body/group/reference cohort. Raw and calibrated metrics share documents. Group bootstrap uses shared draws; it measures cohort uncertainty conditional on one trained seed, not training variability or multiplicity-corrected significance.
6. Report output hashes and report-source/package versions match its manifest. Inspect every rendered PDF page, all figures, missing/undefined values, table support and source citations. Human review is deferred; conclusions remain teacher-agreement findings.

Training clocks include recorded training/validation work but exclude startup and final calibration/performance work. Warm inference timings include preprocessing/all source windows, exclude model loading, and use the common 128-document selection sample on a shared RTX 3060/CPU machine. They are measurements of this environment, not guaranteed production or cloud speed.
