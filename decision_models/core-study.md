# End-to-end study: bounded local continuation

Agreed 12 October 2026: finish an end-to-end teacher-agreement study before expanding training. The local RTX 3060 runs one GPU job at a time. The next checkpoint is after approximately 20 hours; report and review can follow later if needed. All three adaptations remain required.

| New run | Training documents | Context | Recipe |
| --- | ---: | ---: | --- |
| Laya LoRA | 4,000 | 1,024 | CE, unweighted, LR 0.0001 |
| ModernBERT LoRA | 4,000 | 1,024 | CE, unweighted, LR 0.0001 |
| FinBERT LoRA | 4,000 | 512 | CE, unweighted, LR 0.0001 |

Each starts from its pinned pretrained weights: seed 42, rank 8, alpha 16, dropout 0.05, at most three epochs. Epoch selection uses raw selection macro F1. FinBERT is an established finance encoder comparison; its sentiment results are not a claim of current state-of-the-art triage accuracy. ModernBERT is a system comparison with a different classification head, not an isolated causal test of backbone architecture.

1. Reuse verified completed lexical/frozen-encoder baselines, Laya SDK, head curves/context ablations and the 1k Laya LoRA point. Preserve the original study and source snapshots.
2. Train the three recipes above, serially, with resumable checkpoints. No fresh recipe sweep, 16k LoRA, larger head training, generative LLM adaptation, BGE adaptation or RL/RLCD.
3. Freeze all model/epoch/calibration/policy choices. The primary validation-selected comparison uses **4,000 documents for both Laya and the linear baseline**. Larger imported curves are supplementary. Matched pre-update controls isolate adaptation gains for each neural recipe.
4. Evaluate the unchanged held-out cohort once. Report macro F1, per-class results, review-worthy-to-routine misses, calibration/discard tradeoffs, paired group intervals, source support and measured latency/resources. A benchmark-level marker prevents a new output directory from silently reopening the same test set for further selection.
5. Build the actual report/PDF, inspect every PDF page, review code and reproduction evidence, and push reviewable milestones to origin. Keep PRs #4–#8 open in stack order while work proceeds.

Independent human review is deferred by agreement. Findings therefore concern agreement with Luna's rubric labels, with their limitations stated plainly. There are no analyst-validated materiality or trading-profitability claims. No cloud jobs or additional paid API calls are part of this continuation.

The preset command is in [README.md](README.md#bounded-core-study), with artifact and evidence requirements in [reproduce.md](reproduce.md). The previous completed results remain in [review-checkpoint.md](review-checkpoint.md); [research.md](research.md) retains the detailed historical evidence log.

Execution started on 12 October at 21:41 London time. The progress checkpoint is 13 October around 17:41. The original source bytes and frozen benchmark inputs are archived with verified hashes. Report construction and seven-page rendering checks use an explicitly synthetic fixture; held-out evaluation and final report inspection remain pending.

As of 13 October at 01:48 London time, Laya's 4k LoRA fit is complete and independently verified. ModernBERT is running; FinBERT follows. Both rows below use the same 4,000 training documents and 837-document selection cohort, with 552 review-worthy references.

| Method | Selection macro F1 | Critical misses | Warm mean ms |
| --- | ---: | ---: | ---: |
| Laya LoRA | 0.7272 | 81 / 552 | 267.8 |
| Word TF-IDF logistic | 0.7258 | 84 / 552 | 4.5 |

Laya is close to the selected lexical baseline on these fitting scores, at roughly 59 times its warm latency. Its selected LoRA recipe improves F1 over the completed 4k head recipe (0.5825; 105 critical misses). The 1k-to-4k LoRA F1 increment is 0.0713, subject to the documented changing subset composition. No held-out advantage or significance is established yet.

The 1%/5% calibration-selected discard rules reach 2.17%/7.97% critical misses on selection, discarding 8.00%/19.35% of documents; neither transfers its target there. They also discard one/four unclear references. The training clock is 13,963 seconds (3.88 hours), including epoch validation and excluding startup/final calibration/performance work. Training peak is 3.18 GiB. Warm means use the same 128 documents, CPU for TF-IDF and RTX 3060 for Laya, with processing included and loading excluded.

Artifact/source/runtime/cohort checks, raw metric arithmetic, best epoch, finite checkpoints, policies and timing arithmetic passed. The 4k pre-update weights and selection probabilities exactly match the imported 1k LoRA control, validating its reuse. Both remaining adaptations, final freeze/test inference and real report/reproduction review remain required.
