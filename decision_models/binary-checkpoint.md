# Binary study execution checkpoints

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
