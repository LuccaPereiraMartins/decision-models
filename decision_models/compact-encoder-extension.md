# Compact encoder development extension, 13 October 2026

The user authorized another useful experiment before the 6pm London checkpoint. This one runs
BGE-small rank-8 LoRA on the same 4,000-document training subset as the completed FinBERT LoRA
recipe. It asks whether a compact general-purpose encoder gives a useful quality/resource tradeoff
and compares adaptation with the completed frozen BGE/logistic baseline. Architecture, pretrained
objectives, head parameter counts and classifier recipes differ; this is a system comparison,
not a causal test of finance pretraining or adapter updates alone.

Use pinned `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, context
512, uniform whole-document pooling, rank 8, alpha 16, dropout 0.05, unweighted cross-entropy,
both learning rates 0.0001, seed 42 and three epochs. Raw F1 on the existing 837-document
selection cohort chooses the checkpoint, including epoch zero. The separate 363-document
calibration cohort fits temperature and the existing 1%/5% miss-target rules. No new recipe
or seed sweep runs. The provisional budget is 30–60 minutes, with computation stopped at
17:00 London and the remaining hour reserved for verification/reporting by 18:00.

This experiment was chosen after inspecting the original held-out results and the FinBERT
development extension. Selection/calibration scores are fitting evidence. It does not add a
65th held-out result, choose a replacement test winner or change the original study's conclusions.
The original freeze, results, state and ownership marker, plus the completed 16k FinBERT
artifacts, remain unchanged. New generalization evidence requires fresh held-out data.

Execution is standalone and bounded, in `training_runs/research-compact-extension-20261013/`.
The controller runs one GPU job, temporarily prevents idle sleep and releases that guard on
exit. It preserves the latest resumable checkpoint if stopped at the cutoff. No cloud jobs,
paid calls or restart of the full matrix are included. Independent verification and a separate
development section will record completed results or the actual limitation.

## Verified result

The controller finished normally at 14:58 UTC (15:58 London), after 21.8 minutes overall, and
released its awake guard. Recorded training/validation took 21.0 minutes; raw selection F1
selects epoch two. Selection F1 is **0.6586**, compared with **0.6715** for 4k FinBERT LoRA and
**0.6459** for frozen BGE/logistic. BGE's matched epoch-zero F1 is 0.3120; its gain includes
training both a fresh classifier head and encoder adapters, rather than isolating LoRA's effect.

BGE LoRA misses **84 / 552** teacher review-worthy references versus FinBERT's **132 / 552**,
while sending 67 additional selection documents for review. Unclear discards are seven versus
five among 93. Compared with frozen BGE's 109 critical misses, it sends 23 more documents for
review. These different operating points do not establish operational superiority.

Warm mean is **20.23 ms** versus FinBERT's **33.94 ms** on the identical 128-document sample.
Training peak is 0.32 versus 0.78 GiB, inference peak 0.19 versus 0.52 GiB, and trainable
parameters are 149,379 versus 298,755. Frozen BGE/logistic's warm mean is 20.40 ms; its feature
preparation and classifier fit have separate accounting. These are local system measurements.

BGE's calibration-selected 1%/5% policies discard 41/91 selection documents with twelve/36
critical misses (2.17%/6.52%) and no unclear discards. All three compared systems exceed both
empirical calibration targets on selection. Target satisfaction while fitting is not a population bound.

Independent verification checked source/runtime/artifact identities, exact shared 4k subset,
original ordered development cohorts, finite zero/best/final checkpoints and epoch-two selection,
raw/calibrated metrics, temperature/maximal feasible policies and resource clocks/sample. The
best and matched epoch-zero checkpoints were replayed on development data only. All original
core and completed FinBERT hashes remain unchanged. The report's separate section on pages
19–20 and exported development inputs/audit receipts retain the fitting-evidence limitations.
