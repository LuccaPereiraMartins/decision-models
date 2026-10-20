# Pretrained tabular heads: development results

Kumo-large was run on the existing nested training subsets, with frozen FinBERT
features standardized and reduced by training-only PCA. Selection chose among
32/64/96 features. Each context used at most 8,192 training rows and eight recipe
estimators; predictions were averaged over one bag at 4k and two bags at 16k.
All training rows are covered; additional context slots repeat rows.

| Training documents | System | Selection F1 | Calibration F1 | Calibration misses / 365 | Calibration skipped |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4,000 | FinBERT/PCA + Kumo-large | .7926 | .7447 | 77 | 35.8% |
| 4,000 | FinBERT + logistic | .7580 | .7643 | 71 | 35.2% |
| 4,000 | Word TF-IDF + logistic | .7825 | .7629 | 54 | 29.4% |
| 16,000 | FinBERT/PCA + Kumo-large | .8226 | .8055 | 57 | 33.4% |
| 16,000 | FinBERT + logistic | .7869 | .7952 | 58 | 33.0% |
| 16,000 | Word TF-IDF + logistic | .8152 | .8053 | 46 | 29.6% |

F1 is macro F1. Misses and skipped reading use the default binary decision.
The selected PCA widths were 64 and 96. The stronger selection scores do not
establish a routing improvement: calibration performance at 16k is similar to
TF-IDF/logistic, with more misses and more skipped reading. These are system
comparisons; dimensionality reduction also differs from the logistic references.

The jobs were chosen after the original study's test results were available.
Selection fits the width; calibration fits temperature and routing thresholds.
Both cohorts supply development evidence, without an independent generalization
claim. These tabular runs did not evaluate test. Further pretrained expansion
was deferred to prioritize the authorized 25k LoRA work.

## Compute and replay

On the RTX 3060 12 GiB, recorded fitting-routine times were 247.3 seconds at 4k
and 682.2 seconds at 16k. Frozen feature preparation accounted for 198.2 and
476.7 seconds respectively; the second job reused the first 4k vectors. Peak
allocated VRAM was 1.79/2.01 GiB. Final development inference, including context
fitting, took 17.7/93.8 seconds for 1,500 documents. These batch timings do not
measure warm serial latency and are not comparable to the paper's latency table.

```sh
python decision_models/reports/tabular-development/verify.py
```

Python and NumPy replay all saved metrics, ordered cohort/training identities,
selected widths, temperature optimality and maximal whole-tie calibration policies.
The verifier reuses the unchanged independent arithmetic in
`reports/binary-finbert25k/binary_replay.py` and checks its hash. It performs no model
inference. `evidence.json.gz` contains exact source/metadata, body-free predictions,
local context/artifact audit receipts and sampled resource summaries. Bodies,
reduced training matrices and model weights are retained locally, outside Git.

The producing source is commit `cec902279c80df7ac0c69a8cd3da055aae6c5dd3` and is
also embedded byte-for-byte in the evidence. Use the README's CUDA command at
that revision for fitting replay with the frozen local inputs. The 16k controller
was stopped to remove its queued 25k job while its worker finished normally;
the receipt distinguishes verified completion from its unavailable exit code.
