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

## Completed development result

The controller completed normally at 13:37 UTC (14:37 London), with its awake guard released.
Both fits select epoch three. Selection macro F1 rises from **0.6715 to 0.7636**, with teacher
review-worthy misses falling from **132 to 67 among 552**. Default discards fall from 286 to 242
of 837, while discarded unclear references rise from five to seventeen among 93. This is
selection evidence after checkpoint choice; the original held-out conclusion is unchanged.

Recorded training/validation increases from 32.6 to 143.1 minutes, a 4.38-fold increase. The
new controller's overall wall time is 2.41 hours. Warm mean latency is 33.94 versus 34.02 ms
on the identical 128-document sample. Training peak is 0.78/0.79 GiB and inference peak is
0.52 GiB for both. More samples, more optimizer updates and changed source/class proportions
vary together; two points do not identify a minimum sample requirement or the cause of the gain.

The 16k calibration-selected 1%/5% policies discard 63/133 selection documents, with eight/21
critical misses (1.45%/3.80%) and zero/four unclear references discarded. The 1% calibration
target still fails on selection. Calibration fitting scores are not independent evidence.

Independent verification checked source/runtime and artifact hashes, ordered train/selection/
calibration cohorts, nested training identities, finite zero/best/final checkpoints and matched
initialization, checkpoint choice, raw/calibrated metrics, temperature and maximal feasible
policies, and measured resources. The 16k checkpoint was replayed on development cohorts only.
All original study/benchmark ownership hashes remain unchanged. Audits and exact development
inputs are published with the report; full training replay still needs local corpus/checkpoints.
