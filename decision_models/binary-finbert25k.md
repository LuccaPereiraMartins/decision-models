# FinBERT LoRA 25k development extension, 15 October 2026

Completed and independently verified. Read the separate
[development evidence and CPU replay instructions](reports/binary-finbert25k/README.md).
The fit took **4h21m**, selecting epoch three after 9,375 optimizer steps. Selection
macro F1 is **0.8522** versus **0.8464** at 16k, while selection misses increase
from **86 to 93 of 717** review references and default skipping rises from 32.5%
to 34.2%. This is a small fitting gain with a changed routing tradeoff. Independent
GPU replay of all selection/calibration logits and CPU metric/policy replay passed.

The user authorized one 25,000-document FinBERT LoRA fit after reviewing the
completed binary study. This is a standalone development extension. The completed
16-comparison freeze, results and publication packages remain unchanged; the new
job uses training, selection and calibration only. Its scores cannot supply fresh
independent generalization evidence after the earlier test results were inspected.

The run starts from the same pinned pretrained weights and seed as the 4k/16k
recipe: context 512, rank eight, alpha 16, dropout 0.05, uniform whole-document
pooling, unweighted cross-entropy, head/encoder rates 0.0001, window batch eight,
document batch eight, three epochs, raw selection F1 checkpoint choice with epoch
zero eligible. It uses the existing nested 25k training subset and unchanged
1,000 selection / 500 calibration cohorts. No relabelling or paid calls are needed.

Run `python decision_models/run_finbert25k.py` once. It refuses duplicate output,
checks for competing jobs and uses the shared GPU lock. It records process PID,
creation time and command, prevents idle sleep, preserves checkpoints at the
eight-hour cutoff, and verifies protected hashes after exit. Artifacts, logs and
the receipt are under `training_runs/research-binary-finbert25k-20261015/`.
Raw completion is distinct from the subsequent independent scientific checks
recorded in the separate evidence package.

Every 15 seconds the controller records worker CPU time/memory and device-wide
GPU utilisation, memory utilisation, memory use, power, temperature, SM clock and
power state. These samples describe the device, including desktop activity, and
are not kernel-level timings. An observation timeout does not restart training.

## Runtime estimate and the previous Laya run

The 25k subset contains 278,402,752 source characters versus 155,013,200 at 16k:
**1.796 times the text**, rather than only 1.5625 times the documents. The completed
16k commands took 2.51 hours for FinBERT LoRA and 20.66 hours for Laya LoRA.
Character-volume scaling gives approximately **4.51 hours** and **37.10 hours**.
Use **4–5 hours for FinBERT** and **35–40 hours for Laya** as provisional estimates
with the same implementation. Padding, token/window distribution and fixed
development overhead prevent character count from being an exact cost model.
No 25k Laya training job is authorized or launched by this extension.

Laya's three epoch intervals were each approximately 6 hours 46 minutes, based on
preserved initial/epoch validation timestamps. This gives no indication of a
large overnight pause. Its recorded training routine took 20.45 hours of the
20.66-hour command, but that routine includes preparation, validation, checkpoint
I/O and CPU work. CUDA configuration and peak allocation establish GPU use;
they do not establish continuous GPU saturation. There is no continuous
utilisation series for that run. The document-serial training loop and gradient
checkpoint recomputation are plausible throughput constraints that would need
profiling before promising a faster Laya schedule.

## Measured FinBERT GPU activity

The fitting/epoch-validation phase has 1,025 successful device samples: mean GPU
utilisation **92.83%**, median **97%**, P10 **77%**. Only one sample is at or below
5%. Mean device power is 147 W, with maximum sampled temperature 75 C. Startup
and initial selection averaged 25.55% GPU utilisation; final calibration,
selection rescoring and timing averaged 92.0%. Phase boundaries use preserved
file modification times. Worker CPU averaged 89.8% of one logical CPU during
fitting. These sampled device-wide observations establish sustained activity
for this FinBERT run, and do not recover historical Laya utilisation.
