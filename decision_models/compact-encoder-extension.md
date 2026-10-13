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
