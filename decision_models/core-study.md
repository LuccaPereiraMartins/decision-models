# Bounded end-to-end study: results checkpoint

All three new LoRA fits, seven matched pre-update controls and 64 frozen held-out evaluations are complete. Start with [the report](reports/core-v1/report.md) or [the 19-page PDF](reports/core-v1/report.pdf). Its first page explains the findings; later sections give source/class support, critical misses, policies, learning/context comparisons and resources.

The main comparison uses the identical 4,000 training documents and 1,800 held-out documents, with 1,292 teacher review-worthy references. These are agreement results against Luna labels. Independent human review remains deferred.

| Method | Test macro F1 | Critical misses | Warm mean ms |
| --- | ---: | ---: | ---: |
| Laya LoRA | 0.7121 | 177 / 1292 | 267.8 |
| Word TF-IDF logistic | 0.7011 | 187 / 1292 | 4.5 |
| ModernBERT-large LoRA | 0.6892 | 267 / 1292 | 147.4 |
| Frozen FinBERT/logistic | 0.6661 | 236 / 1292 | 34.1 |
| FinBERT LoRA | 0.6419 | 306 / 1292 | 33.9 |

Laya's observed F1 advantage is 0.0110, with paired group-bootstrap interval [-0.0133, 0.0358]. This does not resolve an advantage over the selected lexical baseline; its measured warm latency is 59.2 times greater. Critical misses are teacher review-worthy documents predicted routine. Unclear predictions escalate and are not counted as this critical error.

Increasing Laya LoRA from 1k to 4k raises test F1 by 0.071, with paired interval [0.049, 0.093]. Changing source/class allocation and extra selection/calibration labels qualify sample-efficiency claims. FinBERT LoRA's point estimates are worse than its trained frozen-feature classifier under the declared recipes; this is not an exhaustive tuning comparison. Wider head context does not improve F1 monotonically.

Laya's frozen 1%/5% discard policies reduce workload by 5.39%/14.72%, while missing 1.32%/6.58% of teacher review-worthy cases. Calibration targets do not transfer as deployment bounds. Observed misses, support and uncertainty matter alongside F1.

## Scope and execution

| New run | Training documents | Context | Recipe | Recorded training/validation time |
| --- | ---: | ---: | --- | ---: |
| Laya LoRA | 4,000 | 1,024 | CE, unweighted, LR 0.0001 | 3.88 h |
| ModernBERT-large LoRA | 4,000 | 1,024 | CE, unweighted, LR 0.0001 | 2.17 h |
| FinBERT LoRA | 4,000 | 512 | CE, unweighted, LR 0.0001 | 32.6 min |

All use pinned weights, seed 42, rank 8, alpha 16, dropout 0.05 and at most three epochs; raw selection F1 chooses the epoch. Equal rank does not equalize updated parameters: Laya trains 27.166M including its native head, ModernBERT-large 0.923M and FinBERT 0.299M. Heads, tokenization and window overlap differ; these are system comparisons. FinBERT is an established finance encoder, not a claim of current state-of-the-art triage accuracy.

The local RTX 3060 queue ran serially, starting 12 October at 21:41 London. All comparisons froze at 05:21 on 13 October; held-out inference ran from 05:24 to approximately 10:12. The agreed 20-hour checkpoint was 17:41. Model/epoch/calibration/threshold selection is closed for this holdout. No cloud jobs or further paid API calls were dispatched.

The study reuses 54 completed baseline/reference/head/context/1k-LoRA comparisons. Main trained-model tables use 4k; larger completed curves are supplementary. No 16k LoRA, larger new head training, generative LLM adaptation, BGE adaptation, seed sweep or RL/RLCD is included. Human review is deferred; analyst-validated materiality and trading profitability are unestablished.

## Review and reproduction

All 64 results passed independent ordered-cohort, probability, confusion/F1/miss, calibration/policy, timing and hash checks. The independent audit reproduces every paired interval/contrast using 2,000 shared draws over 1,770 event/near-duplicate groups. Every final PDF page and all six figures were inspected; unchanged pages 1–16 match the previously inspected renders exactly.

Pages 15–17 contain a post-hoc appendix, with separate arithmetic verification. Approximately 68.3% of the small observed F1 difference comes from the unclear class; every class-contribution interval includes zero. Collapsing the existing actions into discard versus review also leaves the F1 difference unresolved. Both models miss 105 teacher review-worthy documents, with 72 additional Laya-only and 82 TF-IDF-only misses. Source differences vary and all four length-band F1 intervals include zero. These exploratory, uncorrected diagnostics and the original policy intervals change no primary conclusion or operating rule.

Pages 17–19 contain the separate [16k FinBERT development extension](development-extension.md), chosen after seeing the primary test results. Selection F1 improves from 0.6715 to 0.7636 against the fixed 4k recipe, with critical misses 132 to 67 among 552 and unclear discards five to seventeen. Training/validation takes 4.38 times longer, while warm latency stays approximately unchanged. This is checkpoint-selected development evidence, not a 65th held-out result; the original comparison and conclusions remain frozen.

[The release package](reports/core-v1/README.md) contains exact logits, references, fixed decisions, metric/statistics evidence, renderer/source archives and a CPU replay command. Metric/statistics replay needs no weights or source bodies. Full training/inference replay needs preserved local corpus/checkpoints and pinned dependencies; [reproduce.md](reproduce.md) explains access and rights limitations.

PRs #4–#8 remain open and draft in stack order. The [older checkpoint](review-checkpoint.md) is historical; [research.md](research.md) retains the detailed evidence log. Focus review on the report's first page, main comparison, discard policies and reproducibility limits before planning another training phase. Further model/policy development needs fresh held-out data.
