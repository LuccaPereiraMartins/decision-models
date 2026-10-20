# Additional 25k LoRA comparisons

The completed FinBERT 25k checkpoint was evaluated on the existing 2,000-document
test cohort, using its original selection checkpoint, temperature and routing
thresholds. No fitting was performed during evaluation.

| System | Training documents | Test macro F1 | Misses / 1,543 | Default skipped reading |
| --- | ---: | ---: | ---: | ---: |
| FinBERT LoRA | 16,000 | .8140 | 202 | 28.8% |
| FinBERT LoRA | 25,000 | .8288 | 193 | 29.0% |

The additional 25k fit improves agreement with the reference labels by 0.0148
macro F1, with nine fewer review-worthy misses and nearly unchanged skipped
reading. Labels remain teacher judgments. The 25k run was added after the original
test results were available; this is an additional comparison on that reused cohort.
The original frozen study and its evidence files remain separate.

```sh
python decision_models/reports/lora25k/verify.py
```

Python and NumPy independently replay exact cohort order, fixed model/decision
identities, raw-to-calibrated predictions, every reported metric and routing decision.
The evidence contains body-free predictions and logits, exact evaluation metadata,
source and local audit receipts. It refers to the unchanged original study and
FinBERT development packages. It contains no weights or document bodies.

## Laya 25k in progress

The inherited Laya recipe was fixed before this FinBERT evaluation: 25,000 training
documents, context 1,024, uniform document pooling, rank-8 LoRA, three epochs,
unweighted cross-entropy, both learning rates 0.0001 and seed 42. Raw selection F1
chooses the earliest best epoch. The producer sources exactly match the 16k fit.

The run launched on 20 October 2026 at 21:50 UTC with resumable checkpoints,
15-second device/worker observations and a 48-hour computation bound. A provisional
30–35-hour estimate scales the measured 16k run; an estimate from the new run's
checkpoints will replace it. Laya 25k results are pending.

The optional 4k Kumo-large full fine-tune failed with CUDA out-of-memory during its
first training episode (1,024 context rows, 256 query rows, 64 reduced features).
Its initial checkpoint is retained. Smaller episodes were deferred to prioritize Laya.
