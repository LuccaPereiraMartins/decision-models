# FinBERT sample-efficiency extension, 13 October 2026

The user requested one worthwhile job from the deferred grid before the 6pm London checkpoint.
This separate development experiment compares the fixed FinBERT LoRA recipe at 4,000 and 16,000
training documents. The completed 4k training/validation clock was 32.6 minutes. The 16k subset
has 4.64 times its raw character volume; the provisional budget is 2–3 hours plus verification,
with a hard compute cutoff of 17:00 London and the remaining hour reserved for reporting.

The new run starts from the same pinned pretrained weights: context 512, rank 8, alpha 16,
dropout 0.05, unweighted cross-entropy, both learning rates 0.0001, seed 42 and three epochs.
Raw macro F1 on the existing 837-document selection cohort chooses the checkpoint, including
epoch zero. The separate 363-document calibration cohort supplies temperature and existing
1%/5% discard-target rules. No additional recipe or seed sweep runs.

This choice was made after observing the primary test results. New results will therefore be
reported as development/selection evidence, with training composition, critical misses,
calibration and measured runtime/memory. Changing class/source composition and the extra
validation labels qualify sample-efficiency claims. Epoch selection means the selection scores
are optimistic fitting evidence, not independent generalization estimates.

The original 64-comparison study, freeze, results and holdout ownership remain unchanged. This
standalone fit does not resume the full study runner, clear any marker or evaluate the July–September
test cohort. A new held-out comparison requires fresh data. It is not a 65th result in the published
primary study. No cloud jobs or paid API calls are included.

Local execution artifacts are in `training_runs/research-extension-20261013/`: the bounded runner,
receipt, training log and `finbert-lora-n16000-c512/` checkpoint/configuration/source snapshots.
The controller temporarily prevents idle sleep, runs one GPU job and releases the guard on exit.
At the cutoff it preserves the latest resumable checkpoint rather than claiming a completed fit.
