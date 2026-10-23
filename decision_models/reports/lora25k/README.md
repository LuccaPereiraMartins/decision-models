# 25k LoRA comparisons

The completed FinBERT and Laya 25k checkpoints were evaluated on the existing
2,000-document test cohort, using their fixed selected checkpoints, temperatures
and routing thresholds. No fitting was performed during test evaluation.

| System | Training documents | Test macro F1 | Misses / 1,543 | Default skipped reading |
| --- | ---: | ---: | ---: | ---: |
| FinBERT LoRA | 16,000 | .8140 | 202 | 28.8% |
| FinBERT LoRA | 25,000 | .8288 | 193 | 29.0% |
| Laya LoRA | 16,000 | .8424 | 171 | 28.0% |
| Laya LoRA | 25,000 | .8503 | 165 | 28.0% |

FinBERT's 16k-to-25k macro-F1 increment is .0148 (paired 95% interval
[.0019, .0284]); Laya's is .0079 ([-.0039, .0204]), including zero. Both reduce
misses slightly at nearly unchanged skipped reading. Intervals use 2,000 paired
whole-group bootstrap draws over 1,967 groups and condition on one fitted run.
They do not measure training variability or correct for multiple comparisons.

On the shared Laya selection set, 25k F1 is higher at epochs one and two (.8269
vs .8112; .8540 vs .8381), then nearly matches the 16k score at epoch three
(.86159 vs .86164). Their final class decisions differ on 56 of 1,000 documents.
The aggregate score converges even though the errors change.

Labels remain teacher judgments. Both 25k runs were part of the planned comparisons
but completed after the initial report because of training time. Their recipes were
fixed in advance and not adjusted in response to earlier results. All training sizes
use the same frozen test cohort; fresh documents are still needed to estimate
generalization to a new sample. The original evidence files remain unchanged.

```sh
python decision_models/reports/lora25k/verify.py
```

Python and NumPy independently replay exact cohort order, fixed model/decision
identities, raw-to-calibrated predictions, reported metrics, routing decisions and
paired intervals. The evidence contains body-free logits/predictions, fitting
histories, evaluation metadata, producer sources, telemetry and local audit
receipts. It refers to the unchanged original study and FinBERT development
packages. It contains no weights or document bodies. Finite-weight checks are
recorded local audits; CPU replay does not reopen model tensors or rerun inference.

## Fitting and resources

The inherited Laya recipe was fixed before the new FinBERT evaluation: 25,000 training
documents, context 1,024, uniform document pooling, rank-8 LoRA, three epochs,
unweighted cross-entropy, both learning rates 0.0001 and seed 42. Raw selection F1
chooses the earliest best epoch. The producer sources exactly match the 16k fit.

Both fits completed 9,375 optimizer steps and selected epoch three. Laya finished
on 22 October at 11:01 UTC, within its 48-hour bound. Independent selected-checkpoint
inference exactly reproduced all 1,000 selection and 500 calibration probabilities
before the single test pass. Audits verified source/runtime hashes, exact ordered
cohorts, the shared 4k/16k subsets, matched initial states and finite checkpoints.

| Measurement | FinBERT 25k | Laya 25k |
| --- | ---: | ---: |
| Fitting command time (including preparation/development) | 4.36 h | 37.17 h |
| Corresponding 16k command time | 2.51 h | 20.66 h |
| Warm mean / P95 document inference | 34.90 / 116.52 ms | 291.46 / 1,190.55 ms |

Inference uses the same 128 selection documents on an RTX 3060 (12 GiB). Laya's
training peak allocated GPU memory was 3.24 GiB. Its 8,890 device/worker samples,
approximately 15 seconds apart, show mean device-wide GPU utilization of 99.2%
and mean worker CPU activity of 51.9%. These observations include preparation and
development and do not measure continuous kernel busy time.

At the calibration-fitted 1% miss target, each 25k model misses 9/1,543 test review
references; achieved skipped reading is 6.2% for FinBERT and 7.1% for Laya. At the
20% calibration skip budget, the respective test outcomes are 65 misses/17.3%
skipped and 62 misses/17.5% skipped. Fitted calibration targets are not guarantees
on test or future documents.

The optional 4k Kumo-large full fine-tune failed with CUDA out-of-memory during its
first training episode (1,024 context rows, 256 query rows, 64 reduced features).
Its initial checkpoint is retained. Smaller episodes were deferred to prioritize Laya.
