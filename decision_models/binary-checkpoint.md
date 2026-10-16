# Binary study execution checkpoints

## 15 October, 11:03 UTC (12:03 London): completed study and paper

All **16 frozen comparisons** are complete: seven shared-4k core systems, the
cheap 4k/16k/25k classifier curves, both 16k LoRA fits and three matched initial
controls. FinBERT LoRA 16k took **2.51 hours**; Laya LoRA 16k took **20.66 hours**.
All training has stopped, ahead of the bounded cutoff. Exact cohorts remain
25,000/1,000/500/2,000, and all 77 original artifact seals match.

Read [the compiled paper](paper-binary-v2/paper.pdf), especially **Section 5 and
Tables 2–4**. It has nine pages: main text through page seven, references and one
supporting appendix page. [LaTeX source](paper-binary-v2/arxiv-source.zip) and
[CPU replay instructions](reports/binary-v2/README.md) accompany it. Every page
and both figures were visually inspected; public evidence regenerates all source
fragments and figures without model runtime or network, and two fresh archive
compilations produce pixel-identical pages. Independent CPU replay passed all
16 comparisons, including fitting choices, calibration, policies and bootstrap.

The prespecified 4k Laya LoRA minus TF-IDF macro-F1 difference is **0.0199**, with
paired 95% interval **[-0.0022, 0.0423]**. Laya skips more reading but misses
**268/1,543** review references versus **188/1,543**, at **64.9 times** the mean
warm latency. At 16k, their F1 point estimates are 0.8424 and 0.8245. These are
teacher-agreement findings on a SEC-dominated cohort that reuses earlier test
bodies; they do not establish expert materiality or fresh generalization.

Labelling finished at **$0.4430207 of the $2 cap**: 4,302 completed requests,
118 uncharged capacity rejections and no unresolved requests. Publication hashes
and closure checks are in [the paper manifest](paper-binary-v2/paper-manifest.json)
and [the verification receipt](reports/binary-v2/publication-checks.json).
Draft PR [#11](https://github.com/LuccaPereiraMartins/systematic-trading/pull/11)
is the paper/evidence review destination; the data and experiment PRs remain
separate. No further experiments or paid calls are scheduled.

## 14 October, 21:44 UTC (22:44 London): second 12-hour review

The seven core development runs, both cheap classifier curves and FinBERT LoRA
16k are complete: **12 completed runs**. Laya LoRA 16k completed its first epoch
in **6.91 hours** and has reached **8,200/16,000** documents in its second pass.
The pipeline, study controller and training process identities are verified;
bound sources and runtime versions match. Its initial weights and predictions
exactly match the 4k recipe. The first-epoch selection score is 0.8112, with
62/717 review-reference misses; this is partial fitting evidence, not a result
from the completed fit or test cohort.

The current estimate is training completion on **15 October, 09:00–10:30 London**,
controls and frozen evaluation later that morning, and a verified paper/evidence
handoff around **16:00–18:00**. These remain estimates, especially the paper review.
The compute cutoff and final handoff deadline remain 16 October as stated below.

Labelling remains **$0.4430207 of the $2 cap**, with 4,302 completed requests,
118 uncharged capacity rejections and no unresolved requests. The binary manifest
still has its frozen identity, and all 77 original artifact hashes match again.
The global model freeze and revised-cohort evaluation are still pending. Next:
finish the admitted fit, run the preserved initial-state controls, freeze choices,
evaluate once, and generate/inspect the concise LaTeX/PDF and CPU replay package.

## 14 October, 14:50 UTC (15:50 London)

FinBERT LoRA 16k completed in **2.51 hours**, selecting epoch three. Independent
development checks passed for artifact/source/runtime identity, ordered cohorts,
finite initial/selected/final checkpoints, metrics, calibration and policies. Its
selection macro F1 is **0.8464**, with **86/717** review-reference false negatives.
Mean/P95 warm latency is **34.92/117.06 ms** on the common 128-document sample.
These are fitting outcomes; the revised test has not been evaluated.

The controller admitted Laya LoRA 16k at 11:22 UTC. Its 21.71-hour forecast plus
30% margin and two-hour evaluation reserve required 30.23 hours; 46.08 hours
remained before the cutoff. Its process identity matches the controller receipt,
and the first pass has reached 8,000/16,000 documents without an error. Completion
on Thursday morning, roughly 09:00–11:00 London, remains provisional.

There are now 12 completed fits/inference jobs. Labelling remains **$0.4430207**,
with no outstanding requests, and all 77 original artifact hashes were checked
again and match. Next are the remaining Laya fit, matched initial-state controls,
the global freeze and one revised-cohort evaluation, followed by the verified
LaTeX/PDF handoff. No additional experiments or paid calls are scheduled.

## First 12-hour review

Verified 14 October 2026, 09:48 UTC (10:48 London). This is an execution checkpoint;
selection and calibration outcomes are fitting evidence, not held-out findings.

The binary dataset is frozen at 25,000 training, 1,000 selection, 500 calibration
and 2,000 test documents. Luna's conservative incremental charge is **$0.4430 of
the $2 cap**: 4,302 completed requests, 118 uncharged capacity rejections and no
unresolved reservations. The original 77 artifact seals still match.

All seven core systems have completed their development fits. Word TF-IDF/logistic
and frozen FinBERT/logistic also completed their 16k and 25k fits. Completed
artifact/source/runtime identities, neural zero/best/final checkpoints, selected
epochs, development metrics, calibration policies and common timing samples have
passed the relevant independent checks. The new test cohort has not been scored.

FinBERT LoRA 16k is training normally after its first epoch. Its initial trainable
weights exactly match the 4k recipe's preserved initial state. Based on this first
pass, completion around 12:20–13:00 London is provisional.

Laya LoRA 4k took 4.73 hours. The 16k subset has 4.59 times its character volume,
giving a 21.71-hour extrapolation. Its gate requires **30.23 hours**, including the
30% margin and two-hour control/evaluation reserve. It currently appears feasible;
the controller will decide after FinBERT 16k finishes. No 25k LoRA is scheduled.

Next: finish FinBERT 16k, apply the Laya 16k gate, score matched pre-update controls,
freeze all choices, and run the revised-cohort evaluation once. Then publish CPU
replay evidence and the concise LaTeX/PDF paper, with every final page inspected.
The original study and paper remain unchanged; the revised test reuses earlier
bodies and will not establish fresh independent confirmation.

PRs #9–#11 remain open/draft. Report/provenance changes are pushed through
`d984377`; that head's CI is green. Public source links now cover captured document
identifiers and resolve the 350 older training rows through the frozen SEC registry.
The paper remains an incomplete methods draft until verified results are generated.

Stop training and model evaluation by **16 October, 10:27 London**; finish the
verified handoff by **16 October, 22:27 London**. Continue within scope without
routine approval requests.
