# LoRA improves the head, but lexical triage still leads this pilot

**Historical validation checkpoint.** The bounded continuation has since completed held-out evaluation.
Start with the [current results checkpoint](core-study.md) and [final report](reports/core-v1/report.md).

Review checkpoint, 12 October 2026. Both jobs agreed for tonight are complete and local research
compute is stopped. Findings below measure **validation-set agreement with Luna's labels**;
they are not held-out performance or independent analyst judgments. The five PRs remain draft.

## The measured comparisons

Every row uses the same 837 selection documents, including 552 labeled review-worthy.
Critical misses are review-worthy documents predicted routine. Mean latency uses the same
128-document warm sample, includes text processing and excludes loading. TF-IDF runs on CPU;
Laya runs on the local RTX 3060. These are shared-machine measurements.

| Training documents | Completed recipe | Macro F1 | Critical misses / 552 | Mean latency, ms |
| ---: | --- | ---: | ---: | ---: |
| 1,000 | Word TF-IDF + linear SVM | 0.675 | 99 | 4.4 |
| 1,000 | Laya LoRA | 0.656 | 104 | 266.5 |
| 1,000 | Laya head | 0.544 | 67 | 252.3 |
| 16,000 | Combined TF-IDF + linear SVM | 0.766 | 67 | 11.3 |
| 16,000 | Laya head | 0.661 | 131 | 251.9 |

The LoRA recipe improves F1 by 0.112 over the matched head while increasing critical misses
by 37. Their pre-update predictions are identical; their adaptation-specific learning rates
differ. This single recipe comparison does not establish a causal LoRA effect or a learning curve.
Stated training sizes exclude additional recipe/context selection supervision. No significance
claim or training-seed variability estimate is made.

The lexical recipes currently offer stronger teacher agreement and much lower measured latency.
The worthwhile remaining question is whether more LoRA data and a matched backbone control
change that comparison enough to justify their cost.

## Calibration did not protect the LoRA pilot at its selected targets

Policies were chosen on the separate 363-document calibration cohort, with 252 review-worthy
references. They meet the empirical calibration miss constraints. They were then applied to
selection without retuning:

| Calibration miss target | Selection documents discarded / 837 | Workload reduction | Critical misses / 552 | Achieved miss rate |
| --- | ---: | ---: | ---: | ---: |
| 1% | 28 | 3.35% | 7 | 1.27% |
| 5% | 91 | 10.87% | 36 | 6.52% |

Both miss targets fail on selection. No unclear references are discarded by these two LoRA
policies. Empirical calibration feasibility is not a population safety guarantee; the current
evidence does not support unattended inbox discard.

## Review the stack in order

| PR | Review focus | What is measured so far |
| --- | --- | --- |
| [#4 Data](https://github.com/LuccaPereiraMartins/systematic-trading/pull/4) | Source rights/provenance, date and event separation, duplicate quarantine, labeling cap | Frozen 25,000 train / 1,200 validation / 1,800 test benchmark; test inference untouched |
| [#5 Baselines](https://github.com/LuccaPereiraMartins/systematic-trading/pull/5) | Converged validation-selected classifiers, probabilities, critical misses, discard policies, common timing | All 45 baseline fits across five nested sizes |
| [#6 Decision models](https://github.com/LuccaPereiraMartins/systematic-trading/pull/6) | Matched pre-update controls, raw-F1 selection, full-source pooling, context and adaptation recipes | Four head loss/weight screens, four head contexts, four head curve sizes and one LoRA pilot |
| [#7 Language models](https://github.com/LuccaPereiraMartins/systematic-trading/pull/7) | Fresh triage heads, conditional A/B/C probabilities, clipping, pinned weights and QLoRA implementation | Frozen FinBERT/BGE baselines and unadapted Laya/Qwen references; language-model adaptation is unrun |
| [#8 Study/report](https://github.com/LuccaPereiraMartins/systematic-trading/pull/8) | Immutable freeze/test guards, bootstrap interpretation, report provenance, sample composition and cloud plan | Verified progress notes and this checkpoint; full report awaits final test evidence |

The independent fresh review pack has not been returned. Existing rubric feedback is qualified
non-expert feedback, not gold labels. The journalistic cohort has no Q3 coverage, and official
announcements do not establish generalization to commercial news.

## Replan before resuming compute

The local LoRA training clock is 1.09 hours and its training peak is 3.18 GiB; startup, final
calibration/selection and resource measurement are additional. The [cloud proposal](cloud-compute.md)
records primary-source reference prices, exact-byte transfer checks and a small trial before
booking the remaining matrix. Cloud speedup and full-matrix cost are not measured.

The larger LoRA curves, ModernBERT architecture control, FinBERT/BGE/Qwen adaptation, final
model/policy freeze, one held-out evaluation, fresh human audit and publication report/PDF remain
unfinished. Agree the retained comparisons and compute cap first. Preserve the original plans,
source snapshots and checkpoints; do not restart the unrestricted `all` stage.

Detailed metrics, policy counts, recipe limits and verification summaries are in the dated
[research milestones](research.md). Full ignored artifacts and independent receipts remain in
`training_runs/research-study/` and `data/research/checks/` in the local checkout. RL/RLCD remains
optional after the core research.
