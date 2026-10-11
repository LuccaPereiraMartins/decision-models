# Financial-document triage research programme

Approved scope: source-content-only triage for analyst inboxes, written for AI/ML engineers in finance. Labels remain routine, review_worthy and unclear; unclear routes to review. Macro F1 is primary. Review-worthy -> routine is the critical miss. Trading strategies and historical-context reasoning are outside this study.

## Five dependent PRs

1. **Data**: corporate releases, 6-Ks, Fed, ECB and general news alongside 8-Ks; source census and rights register; provenance; staged Luna labels; temporal/event/near-duplicate separation; blinded human-review pack.
2. **Baselines**: word/character/combined TF-IDF, logistic regression/SVM, frozen encoders; validation-only selection; probability calibration and conservative discard policies; common quality/cost/throughput records.
3. **Decision models**: Laya base/head/LoRA; cross-entropy versus Brier loss; 512/1024/2048/4096 context; fixed existing pooling; sample efficiency; ModernBERT architecture control.
4. **Alternatives**: FinBERT/BGE adaptation, Qwen3-1.7B/4B prompt classification and bounded QLoRA; explicit truncation and matched-coverage comparisons; a cost proposal for the GPT-6.1 Sol reference (not executed).
5. **Release**: source-backed report and PDF, figures, error analysis, paired group bootstrap, reproducibility audit and a small serial experiment runner.

## Frozen defaults

- One seed: 42. No repeated-seed training; evaluation bootstrap does not measure training variability.
- Collect up to 100,000 candidate documents from 2019-01-01 through 2026-09-30. Primary 8-Ks at most 40% of each retained train/validation/test split, including after budget-limited labeling; no pretending that source shortfalls were filled by more 8-Ks. Raw acquisition counts remain separate from benchmark support.
- News coverage includes reusable Wikinews and HM Treasury official announcements, with separate provider counts and metrics. The journalistic cohort has no Q3 2026 coverage; official announcements cannot establish that generalization.
- Corporate releases are a text-query sample of SEC EX-99 exhibits from 8-K filings. The saved search responses verify 244,685 candidates and paginated ranges cover all approved dates. Twelve exhibits from interrupted monthly searches were absent from completed smaller-range retries and remain traceable to their original responses. Discovery does not establish an exhaustive release census.
- Offline re-extraction checks source text, publication dates and response provenance. Three retained Wikinews pilot records had their dates and licence fields repaired from preserved responses; their bodies, identities and training-period membership are unchanged. Original files and verification receipts are retained.
- Train through 2026-03-31, validation 2026-04-01..2026-06-30, test 2026-07-01..2026-09-30. Previously experimented-on bodies and near duplicates cannot enter new validation/test. Related groups crossing boundaries are quarantined. Issuer-disjoint evaluation is outside the core scope.
- Up to 1,200 validation and 1,800 test rows; validation partitions 70% selection/30% calibration and policy fitting. No test inference until all core selections are frozen.
- Nested training subsets: 250, 1000, 4000, 16000 and all available, using deterministic round-robin family/label ordering and stable body hashes. Class/source proportions can change as smaller strata exhaust; account for this allocation when interpreting sample efficiency. Recipe selection also uses additional screening and validation labels. Start each neural run from its original pretrained checkpoint.
- GPU jobs run serially on the local RTX 3060 12 GB. Preserve checkpoints, source fingerprints and failed configurations. Local API cost is not the same as zero compute cost.
- Only Luna Flex labeling is currently authorised for paid execution: a shared **$5 maximum**, preserving at least $2 of the stated $7 credits. Keep future relabeling possible. Cloud and paid reference benchmarks need concrete quotes before execution.
- Luna cost accounting includes ordinary input, cache reads, cache writes and output, including long-context premiums. One-use labeling requests use explicit caching without breakpoints; byte-bound reservations still cover cache-write charges. Rates and usage travel with each new annotation. The paid-reference quote assumes all input incurs cache-write pricing.
- Supervised head/LoRA objectives are core. RL/RLCD is a possible follow-on after these five PRs.
- Human review: 12 rubric examples, 72 fresh test documents, 12 overlapping second reviews; three hours total. Prepare now, review later. Unresolved disagreements stay unresolved. Independent quality claims wait for review.

The first overnight/daytime phase covers collection pilots, staged labeling, splits, baselines and initial adaptation screening. The complete experiment matrix is a multi-day programme. Publication claims must distinguish teacher agreement, non-expert human review, missing paid benchmarks and training-seed limitations.

## Progress snapshot: 2026-10-11, 13:15 UTC

Collection and labeling are complete. The raw acquisition contains 99,006 documents, including the
10,000-document historical corpus. The fresh benchmark retains 25,000 training, 1,200 validation and
1,800 test documents across six document families. Selection uses 837 validation documents; calibration
and discard-policy fitting use the other 363. All five nested training sizes are available.

The frozen manifest SHA-256 is `336c29fc36c4f7678c6bae4a785d12ee4d6c6985fc8aba8d4a53efe172be6852`.
Offline checks verified retained source responses, body/event separation, dates, subset membership,
labels and input hashes. The Luna Flex ledger records $4.886856 charged or conservatively reserved
against the approved $5 ceiling, including three unresolved original-request reservations. This is
local ledger accounting, not an account balance. No further paid calls are running.

All 45 baseline fits are complete: nine methods at 250, 1,000, 4,000, 16,000 and 25,000 training
documents. Their artifacts, converged-candidate selection, reference cohorts, probability/metric
calculations and separate calibration records passed verification. Laya SDK and both prompt-only
Qwen references are also complete and verified. Every completed run records a common 128-document
warm latency sample; final resource and performance comparisons still await the complete study.

The table below shows **selection-set teacher agreement**, used for recipe selection. It is preliminary
and is not held-out test performance or analyst accuracy. Critical misses are review-worthy documents
predicted routine; the common selection cohort contains 552 review-worthy references. Baselines use
25,000 training labels, whereas the three references are unadapted, so this table cannot establish
which model family adapts best. Zero critical misses alone does not imply useful triage: the majority
method sends every document to review.

| Completed method | Selection macro F1 | Critical misses / 552 |
| --- | ---: | ---: |
| Majority | 0.265 | 0 |
| Word TF-IDF + logistic regression | 0.779 | 58 |
| Word TF-IDF + linear SVM | 0.779 | 53 |
| Character TF-IDF + logistic regression | 0.763 | 42 |
| Character TF-IDF + linear SVM | 0.761 | 38 |
| Combined TF-IDF + logistic regression | 0.778 | 44 |
| Combined TF-IDF + linear SVM | 0.783 | 54 |
| Frozen FinBERT + logistic regression | 0.721 | 48 |
| Frozen BGE + logistic regression | 0.699 | 55 |
| Laya native SDK, unadapted | 0.387 | 130 |
| Qwen3-1.7B prompt, unadapted | 0.305 | 92 |
| Qwen3-4B prompt, unadapted | 0.343 | 4 |

Qwen uses a 4,096-token prompt budget with 3,905 source tokens, retaining the beginning and end when
necessary. Laya and the frozen encoders process document windows. Coverage-matched comparisons remain
required in the final report.

The first Laya head-tuning screen is active on 1,000 documents with 1,024-token context and cross-entropy.
At this snapshot, 48 jobs are complete, one is running and no job is failed. The remaining adaptation,
context and sample-efficiency matrix runs serially under the GPU lock. No fresh test predictions have
been made; model/policy freeze precedes that evaluation. The blinded non-expert human audit, actual
report/PDF inspection and publication/reproducibility review remain outstanding. All five PRs stay draft.

Full local evidence is under `decision_models/training_runs/research-study/` and
`decision_models/data/research/checks/`; generated artifacts remain ignored. Source, runtime, manifest
and artifact fingerprints accompany completed runs. This committed snapshot lets reviewers follow
progress without distributing source bodies or treating unfinished experiments as findings.

## First adaptation screen completed: 2026-10-11, 13:51 UTC

The first Laya head screen completed on the frozen 1,000-document subset, with cross-entropy,
unweighted classes, learning rate 0.00003 and 1,024-token context. Raw selection macro F1 selected
epoch three. The best, pre-update and resume checkpoints, training-subset identity, source/runtime
fingerprints, complete selection/calibration cohorts and independently recomputed metrics passed
verification. Neural prediction files preserve a deterministic seed-42 shuffle of those cohorts.

| Epoch | Selection macro F1 | Critical misses / 552 | Predicted unclear |
| --- | ---: | ---: | ---: |
| Pre-update | 0.362 | 76 | 0 |
| 1 | 0.363 | 80 | 0 |
| 2 | 0.379 | 87 | 0 |
| 3, selected | 0.539 | 62 | 117 |

The selection gain is 0.177 macro F1 against this screen's own pre-update model. Calibration-set
agreement is 0.619 macro F1 with 23 critical misses among 252 review-worthy references. These are
fitting-cohort results, with no independent significance or held-out performance claim. At the same
1,000 training labels, the highest-F1 linear baseline is word TF-IDF/SVM: 0.675 selection macro F1
and 99 critical misses. Default classifier misses and achieved calibrated discard-policy rates are
separate comparisons; the remaining recipe and learning-curve experiments are still required.

On the common 128-document warm sample, this screen's selected model measured 252 ms mean latency
and 1,058 ms P95 on the shared local machine. The recorded training phase, including epoch validation,
took 2,445 seconds; post-training calibration, evaluation and performance measurement are additional.
The next class-weighted cross-entropy screen is running under the same serial GPU lock. There are now
49 completed jobs, one running and no failed jobs. Test evaluation and the human audit remain pending.

## Weighting and discard policies: 2026-10-11, 14:39 UTC

Both cross-entropy head screens are complete and verified on the identical 1,000-document training
subset and 1,024-token context. Both select epoch three from the same pre-update model. Weighting
reduces default review-to-routine misses here, while their selection macro F1 scores are close.
These are single-seed fitting results; no significance or generalization claim follows from this screen.

| Cross-entropy recipe | Selection macro F1 | Selection critical misses / 552 | Calibration macro F1 | Calibration critical misses / 252 |
| --- | ---: | ---: | ---: | ---: |
| Unweighted | 0.539 | 62 | 0.619 | 23 |
| Class-weighted | 0.535 | 37 | 0.552 | 14 |

Separate calibration chooses the largest discard count satisfying each empirical review-worthy miss
budget. The 1% target permits two calibration misses in both runs: five documents discarded unweighted,
four weighted. The 5% target permits twelve misses: 33 discarded unweighted, 37 weighted. Independent
checks verified saved policy flags, every count/rate and maximum feasible calibration discard counts.

Applying those fixed thresholds to the selection cohort gives:

| Recipe | Calibration miss target | Selection discarded / 837 | Workload reduction | Selection critical misses / 552 | Selection miss rate | Unclear references discarded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unweighted | 1% | 19 | 2.27% | 1 | 0.18% | 9 |
| Class-weighted | 1% | 15 | 1.79% | 2 | 0.36% | 6 |
| Unweighted | 5% | 101 | 12.07% | 27 | 4.89% | 32 |
| Class-weighted | 5% | 104 | 12.43% | 33 | 5.98% | 25 |

The weighted 5% calibration target exceeds 5% on selection; the fitted target is not a population
guarantee. Both 1% policies filter little work. Only predicted routine documents are eligible for
discard, but mistaken unclear references can still be discarded, as counted above. These selection
results are descriptive checks, not a new recipe-selection criterion or held-out test evaluation.

The weighted model measured 251 ms mean and 1,056 ms P95 on the same 128-document warm sample;
its training phase including epoch validation took 2,394 seconds, excluding subsequent calibration,
prediction and timing passes. The first unweighted Brier-loss screen is now running. There are 50
completed jobs and no failures; the remaining loss, context and learning-curve matrix is unchanged.

## Four head recipes completed: 2026-10-11, 16:19 UTC

All four Laya head screens completed on the same frozen 1,000-document subset at context 1,024,
learning rate 0.00003 and seed 42. Each selected epoch three. Complete prediction cohorts, checkpoint
state, source/runtime fingerprints, training identity and calibrated discard counts passed verification.

| Head recipe | Selection macro F1 | Critical misses / 552 | Predicted unclear / 837 |
| --- | ---: | ---: | ---: |
| Cross-entropy, unweighted | 0.539 | 62 | 117 |
| Cross-entropy, class-weighted | 0.535 | 37 | 180 |
| Brier, unweighted | 0.472 | 85 | 32 |
| Brier, class-weighted | 0.493 | 21 | 266 |

Unweighted cross-entropy advances under the prespecified raw selection macro F1 criterion.
An independent check confirmed that the running 4,000-document, 512-token context job uses that
recipe. The remaining context candidates are 1,024, 2,048 and 4,096 tokens; these runs start from
the pretrained model on the common 4,000-document subset.

Weighted Brier has fewer default critical misses, but predicts substantially more unclear labels.
Its separately fitted 1% and 5% calibration policies discard 13 and 64 selection documents
(1.55% and 7.65%), with two and 21 review-worthy misses (0.36% and 3.80%). They also discard two
and eleven unclear references. On the calibration cohort they discard four and 24 documents,
with two and twelve review-worthy misses among 252 references. Fewer default misses alone
does not establish a better workload/safety operating point.

These are teacher-agreement fitting results, with no significance or held-out performance claim.
There are 52 completed jobs, one context job running and no failed jobs at this snapshot.
LoRA screens, context and sample-efficiency curves, model/policy freeze, held-out test, the
qualified human audit and final report/reproducibility review remain pending.

## First context run completed: 2026-10-11, 18:43 UTC

The 4,000-document Laya head run at context 512 completed with the selected unweighted
cross-entropy recipe. Raw selection macro F1 selects epoch two; retaining the final epoch would
have reduced agreement and increased critical misses. Complete cohorts, checkpoint state,
source/runtime fingerprints, training identity, timing summaries and discard policies passed
independent verification. The next context-1,024 run uses the identical 4,000-document subset.

| Epoch | Selection macro F1 | Critical misses / 552 | Predicted unclear / 837 |
| --- | ---: | ---: | ---: |
| Pre-update | 0.392 | 111 | 0 |
| 1 | 0.534 | 180 | 195 |
| 2, selected | 0.560 | 177 | 188 |
| 3 | 0.494 | 258 | 216 |

The selected model gains 0.168 macro F1 against its own pre-update control, but has more default
critical misses. Calibration agreement is 0.572 macro F1 with 77 misses among 252 review-worthy
references. At the same 4,000 training labels, the highest-F1 linear baseline is word TF-IDF/logistic:
0.726 selection macro F1 and 84 critical misses. This is a fitting-cohort comparison; the remaining
contexts and adaptation methods are required before any model-family conclusion.

| Calibration miss target | Selection discarded / 837 | Workload reduction | Selection critical misses / 552 | Selection miss rate | Unclear references discarded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 14 | 1.67% | 4 | 0.72% | 1 |
| 5% | 46 | 5.50% | 21 | 3.80% | 1 |

The empirical calibration policies discard five and nineteen calibration documents respectively,
with two and twelve review-worthy misses. Each maximizes calibration discard count within its
fitted budget; the target is not a population guarantee. Classifier errors and achieved discard
rates remain separate measures.

The recorded training phase, including epoch validation, took 8,195 seconds; subsequent calibration,
prediction and timing passes are additional. Warm inference on the common 128-document sample
measured 291 ms mean and 1,189 ms P95, including text processing and excluding model loading.
Training peak allocated VRAM was 2.10 GiB on the shared local machine.

There are 53 completed jobs, one context job running and no failures at this snapshot. These are
single-seed teacher-agreement fitting results. The remaining context/adaptation/sample curves,
model/policy freeze, held-out test, qualified human audit and final report remain pending.
