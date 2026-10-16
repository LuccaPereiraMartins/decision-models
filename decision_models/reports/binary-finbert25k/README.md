# FinBERT LoRA 25k: development evidence

This separate package compares the fixed FinBERT LoRA recipe at 4k, 16k and 25k
training documents on the unchanged 1,000 selection and 500 calibration documents.
It contains no test evaluation. The 25k experiment was chosen after earlier test
outcomes were inspected; these are fitting results, not fresh generalization evidence.
The original 16-comparison freeze and scientific results are unchanged. Fresh held-out data is required for a new generalization comparison.

| Training documents | Selection macro F1 | Selection misses / 717 | Default documents skipped |
| --- | ---: | ---: | ---: |
| 4,000 | 0.7512 | 175 | 40.8% |
| 16,000 | 0.8464 | 86 | 32.5% |
| 25,000 | 0.8522 | 93 | 34.2% |

The 25k fit selected epoch three after 9,375 optimizer steps. The selection F1 gain
over 16k is 0.0058, while missed review references increase by seven. This is a small
fitting gain with a changed miss/workload tradeoff, rather than evidence of uniformly
better routing. Calibration F1 is 0.8340 versus 0.8149 at 16k. The 1% calibration
miss-target policy achieves nine selection misses and 10.3% skipped reading at 25k;
its achieved selection miss rate is not a population guarantee.

The bounded command completed in 4h21m (15,685.8 seconds). Recorded training routine
time is 15,570.1 seconds, including preparation and epoch validation. Mean/P95 warm
inference on the common 128-document sample is 34.90/116.52 ms. Training peak allocated
GPU memory is 0.789 GiB; device-wide memory readings additionally include context,
reserved memory and desktop use.

## Replay and contents

```sh
python decision_models/reports/binary-finbert25k/verify.py
```

The published verifier requires Python and NumPy only. It checks exact cohort and
training order, matched initialization, fitting/checkpoint choice, confusion/F1/miss
and workload arithmetic, independently replayed development logits and raw/calibrated
metrics, calibration temperature and maximal whole-tie policies, timing and telemetry.
It verifies every package file against package-manifest.json. It does not rerun models.

- evidence.json.gz: body-free cohort metadata, ordered development predictions,
  independent raw logits, 4k/16k/25k fitting histories, checkpoint audit receipts,
  training composition, resources and phase summaries.
- metadata.zip: exact original development configurations, histories, predictions,
  decisions, completion records, telemetry-controller receipt and 25k training log.
- source.zip: exact producer/runtime/exporter/replay source closure.
- telemetry.jsonl.gz: original device/worker observations at approximately 15-second
  intervals, including observation failures if any.
- local-audit.json and cpu-replay.json: local artifact/finite-weight and independent
  arithmetic checks. Weights and source document bodies are excluded.

The local audit checked finite initial/selected/final checkpoints and source/runtime
identity for all three fitting recipes. Initial weights match exactly. A separate
GPU inference replay of the completed 25k selected checkpoint on selection/calibration
checks the saved probabilities and decisions. This replay neither fits a model nor
accesses test documents. Exact fitting replay additionally needs the retained local
text snapshots and pinned pretrained weights; their identities accompany
the evidence and the binary input archive receipt. Selected models remain local;
completed-run intermediate checkpoints were removed after the recorded audits.

## GPU observations

| Recorded phase | Samples | Mean GPU utilisation | Median |
| --- | ---: | ---: | ---: |
| Startup and initial selection | 11 | 25.55% | 0% |
| Fitting and epoch validation | 1,025 | 92.83% | 97% |
| Final calibration, selection and timing | 5 | 92.00% | 90% |

Only one fitting sample is at or below 5% utilisation. Mean fitting-phase device
power is 147 W; maximum sampled temperature is 75 C. The fitted checkpoint's
independent development replay took a further 88.6 seconds, outside training timing.

Phase boundaries use saved initial/final validation file modification timestamps.
The fitting phase includes epoch validation, checkpoint I/O and CPU work; the final
phase includes calibration, selection rescoring and timing. Raw samples and boundaries
permit CPU replay of the phase statistics. Device-wide utilisation includes desktop
activity, and sampled utilisation is not continuous kernel busy time. Worker CPU
percent uses 100% for one logical CPU, not the whole machine. These observations
measure this FinBERT job and do not establish historical Laya utilisation.
