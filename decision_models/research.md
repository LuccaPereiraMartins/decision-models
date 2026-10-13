# Financial-document triage research programme

Approved scope: source-content-only triage for analyst inboxes, written for AI/ML engineers in finance. Labels remain routine, review_worthy and unclear; unclear routes to review. Macro F1 is primary. Review-worthy -> routine is the critical miss. Trading strategies and historical-context reasoning are outside this study.

The [bounded core study](core-study.md) is the current agreed plan: three 4k LoRA runs including FinBERT,
then frozen held-out evaluation and the report. Human review and larger/generative adaptation are deferred.
The [12 October review handoff](review-checkpoint.md) records the completed earlier checkpoint.

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

The first overnight/daytime phase covers collection pilots, staged labeling, splits, baselines and initial adaptation screening. The complete experiment matrix requires a separately agreed compute plan. Publication claims must distinguish teacher agreement, non-expert human review, missing paid benchmarks and training-seed limitations. Dated snapshots below retain their original state; the latest execution boundary takes precedence.

## Review checkpoint agreed: 2026-10-12, 7pm London

The full local matrix was taking substantially longer than the original overnight/daytime estimate.
The user set a review deadline of **2026-10-12 at 19:00 Europe/London**. The unrestricted pipeline
has therefore been stopped, preserving its original plans, source hashes and resumable checkpoints.
The current 16,000-document head run resumes with its unchanged recipe from epoch two, document
15,600. Its finite weights, optimizer/RNG state and configuration were checked before resuming;
all 27 workflow source/dependency fingerprints still match. This is a continuation of the same
fit, not a new training-seed experiment.

Only two jobs are permitted before the review handoff:

1. Finish the current Laya head learning-curve point at 16,000 training documents.
2. Run one Laya LoRA pilot on the frozen 1,000-document subset: cross-entropy, unweighted classes,
   1,024-token context, rank 8, learning rate 0.0001 and at most three epochs. It uses the same
   selection/calibration cohorts and full-document pooling as the completed head curve. The
   adaptation-specific learning rates differ, so this is a recipe comparison rather than an
   isolated causal estimate of the effect of LoRA. The designated 1,000-document head curve and
   the matched lexical baselines provide the comparisons; no extra seed or LoRA tuning sweep runs.

The bounded local runner stops compute at **18:00 London**, reserving the final hour for artifact
verification, origin pushes and a review handoff. A job that reaches the cutoff retains its last
checkpoint and is reported as interrupted, never as a completed result. No 25,000-document head
fit or further model/size/context sweep starts automatically. The runner and transition receipt
are retained under the ignored local checks directory; the pinned training source stays unchanged.

Tonight's deliverable is the five draft PRs, verified fitting results and a clear list of outstanding
work. It is not the completed publication study: model/policy freeze, held-out test, the fresh
qualified non-expert audit and the actual research report/PDF remain pending. The test firewall
remains intact while the experiment scope is replanned. Future larger LoRA curves, architecture
controls and FinBERT/BGE/Qwen adaptation require a compute proposal based on measured pilot time,
cloud pricing and checkpoint portability. No cloud job or additional paid model call is dispatched.
The [cloud compute proposal](cloud-compute.md) records public reference prices, the transfer checks
and a suggested order for the replanned experiments; it does not authorise paid execution.

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

## Second context run completed: 2026-10-11, 20:49 UTC

The context-1,024 Laya head run completed on the same 4,000-document subset and unweighted
cross-entropy recipe as the context-512 run. Each selects epoch two under raw selection macro F1.
Saved checkpoints, complete prediction cohorts, training identities, source/runtime fingerprints,
timing summaries and calibrated discard policies passed independent verification.

| Context | Selection macro F1 | Critical misses / 552 | Predicted unclear / 837 | Calibration macro F1 | Calibration critical misses / 252 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.560 | 177 | 188 | 0.572 | 77 |
| 1,024 | 0.582 | 105 | 195 | 0.571 | 45 |

The longer context improves selection agreement and reduces default critical misses in these two
completed runs. Both remain below the matched word TF-IDF/logistic baseline: 0.726 selection
macro F1 and 84 critical misses. These are single-seed fitting comparisons against teacher labels;
the remaining contexts, adaptations and held-out evaluation are needed before broader conclusions.
The context-1,024 model gains 0.220 macro F1 against its own pre-update control (0.362), while
default critical misses increase from 76 to 105. Its third epoch scores 0.557 with 127 misses,
below the selected second epoch.

| Calibration miss target | Selection discarded / 837 | Workload reduction | Selection critical misses / 552 | Selection miss rate | Unclear references discarded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 18 | 2.15% | 4 | 0.72% | 1 |
| 5% | 51 | 6.09% | 23 | 4.17% | 3 |

The context-1,024 policies discard three and fifteen calibration documents respectively, with two
and twelve review-worthy misses and no unclear references discarded. Each maximizes calibration
discard count within its fitted budget. These empirical targets do not guarantee population risk;
default classifier misses and policy discard outcomes remain separate measures.

The recorded training phase, including epoch validation, took 7,160 seconds. Subsequent calibration,
prediction and timing passes are additional. Warm inference on the common 128-document sample
measured 254 ms mean and 1,065 ms P95, including text processing and excluding model loading.
Training peak allocated VRAM was 2.35 GiB. Timing is descriptive on the shared local machine.

There are 54 completed jobs and no failed jobs at this snapshot. The context-2,048 run is active on
the identical subset. Context 4,096, the full adaptation/sample-efficiency matrix, model/policy
freeze, held-out test, qualified human audit and final report/reproducibility review remain pending.

## Third context run completed: 2026-10-11, 22:55 UTC

The context-2,048 Laya head run completed on the same 4,000-document subset, unweighted
cross-entropy recipe and fixed whole-document pooling. Complete checkpoint/cohort identities,
source/runtime fingerprints, probability metrics, timing summaries and calibrated discard
policies passed independent verification.

| Context | Selected epoch | Selection macro F1 | Critical misses / 552 | Predicted unclear / 837 | Calibration macro F1 | Calibration critical misses / 252 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2 | 0.560 | 177 | 188 | 0.572 | 77 |
| 1,024 | 2 | 0.582 | 105 | 195 | 0.571 | 45 |
| 2,048 | 3 | 0.556 | 111 | 260 | 0.512 | 47 |

Among the three completed contexts, 1,024 has the highest selection F1. Increasing context
does not improve it monotonically in these runs; no significance or model-family claim follows
from this fitting comparison. The matched word TF-IDF/logistic baseline remains at 0.726 F1
with 84 critical misses. The fourth context and remaining adaptation/sample curves are pending.

For context 2,048, the pre-update control scores 0.381 F1 with 87 critical misses. Epochs one,
two and three score 0.497, 0.540 and 0.556 with 59, 91 and 111 misses respectively. Raw selection
F1 selects epoch three, a gain of 0.175 over its own control, while default critical misses rise.
These classifier counts are distinct from outcomes under the calibrated discard policies below.

| Calibration miss target | Selection discarded / 837 | Workload reduction | Selection critical misses / 552 | Selection miss rate | Unclear references discarded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 31 | 3.70% | 10 | 1.81% | 1 |
| 5% | 64 | 7.65% | 31 | 5.62% | 1 |

Both policies exceed their calibration miss targets on selection. On calibration they discard
three and fifteen documents, with two and twelve review-worthy misses among 252 references and
no unclear references discarded. Verification confirms maximal feasible calibration discard
counts; this transfer failure illustrates why an empirical fitting constraint is not a population
risk guarantee. It does not trigger selection-cohort retuning of the prescribed policies.

The recorded training phase, including epoch validation, took 7,110 seconds; subsequent calibration,
prediction and timing passes are additional. The common 128-document warm sample measured
250 ms mean and 1,171 ms P95, including text processing and excluding model loading. Training
peak allocated VRAM was 2.84 GiB. These timings describe the shared local machine.

There are 55 completed jobs and no failed jobs at this snapshot. The context-4,096 run is active
on the identical subset. The full adaptation/sample-efficiency matrix, model/policy freeze,
held-out test, qualified human audit and final report/reproducibility review remain pending.

## Head context ablation completed: 2026-10-12, 01:21 UTC

All four Laya head context runs completed on the identical 4,000-document subset with unweighted
cross-entropy, learning rate 0.00003, seed 42 and fixed whole-document pooling. Checkpoints, complete
cohorts, source/runtime fingerprints, probability metrics, timing and discard policies passed
independent verification. A separate check recomputed all four selection scores on matching row
identities and confirmed that the new 250-document curve run uses the selected recipe and context.

| Context | Selected epoch | Selection macro F1 | Critical misses / 552 | Predicted unclear / 837 | Calibration macro F1 | Calibration critical misses / 252 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2 | 0.560 | 177 | 188 | 0.572 | 77 |
| 1,024 | 2 | 0.582 | 105 | 195 | 0.571 | 45 |
| 2,048 | 3 | 0.556 | 111 | 260 | 0.512 | 47 |
| 4,096 | 3 | 0.572 | 121 | 229 | 0.544 | 54 |

Context 1,024 advances under the prespecified raw selection macro F1 criterion. Larger contexts
do not improve agreement monotonically in this head ablation. These are single-seed fitting results
against Luna references, with no significance or held-out performance claim. The matched word
TF-IDF/logistic baseline scores 0.726 F1 with 84 critical misses; the full learning curves and
remaining adaptation methods are needed before broader model-family conclusions.

At context 4,096, the pre-update control scores 0.379 F1 with 91 critical misses. Epochs one,
two and three score 0.534, 0.569 and 0.572 with 84, 107 and 121 misses respectively. Raw F1
selects epoch three, a gain of 0.194 against its own control, while default critical misses rise.

| Calibration miss target | Selection discarded / 837 | Workload reduction | Selection critical misses / 552 | Selection miss rate | Unclear references discarded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 42 | 5.02% | 15 | 2.72% | 1 |
| 5% | 76 | 9.08% | 36 | 6.52% | 1 |

Both context-4,096 policies exceed their calibration miss targets on selection. Calibration
discards four and nineteen documents respectively, with two and twelve review-worthy misses among
252 references and no unclear references discarded. Each maximizes calibration discard count
within its fitted budget. Empirical fitting constraints do not guarantee population risk, and
these transfer checks do not retune the prescribed policies.

The context-4,096 recorded training phase, including epoch validation, took 8,043 seconds; subsequent
calibration, prediction and timing passes are additional. The common 128-document warm sample
measured 263 ms mean and 1,268 ms P95, including text processing and excluding loading. Training
peak allocated VRAM was 3.83 GiB. Timing describes the shared local machine.

There are 56 completed jobs and no failures at this snapshot. The selected context-1,024 head recipe
is now running the 250-document point of its nested learning curve; sizes 1,000/4,000/16,000/25,000
follow, reusing the completed selected 4,000-document context run. Recipe/context selection used
additional 1,000/4,000-document supervision and validation labels, so the smallest curve point is
not a claim of only 250 total supervised labels. LoRA and language-model adaptation, the full
sample-efficiency matrix, model/policy freeze, held-out test, qualified human audit and final
report/reproducibility review remain pending.

## First head learning-curve points verified: 2026-10-12, 03:01 UTC

The selected context-1,024, unweighted cross-entropy head recipe now has verified points at
250, 1,000 and 4,000 training documents. The 4,000-document point reuses its completed context
run. Checkpoints, exact nested identities, full selection/calibration cohorts, source/runtime
fingerprints, probability metrics, discard policies and timing summaries passed verification.

| Training documents | Selected epoch | Head selection F1 | Head critical misses / 552 | Predicted unclear / 837 | Highest-F1 matched baseline | Baseline selection F1 | Baseline critical misses / 552 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 250 | 2 | 0.364 | 76 | 0 | Word TF-IDF/SVM | 0.678 | 82 |
| 1,000 | 3 | 0.544 | 67 | 110 | Word TF-IDF/SVM | 0.675 | 99 |
| 4,000 | 2 | 0.582 | 105 | 195 | Word TF-IDF/logistic | 0.726 | 84 |

The common pre-update control at context 1,024 scores 0.362 F1 with 76 critical misses and no
unclear predictions. The selected head gains are 0.002, 0.182 and 0.220 respectively. At 250
documents, none of the three updated epochs predicts unclear; this point gives little observed
agreement gain. Default critical misses do not improve monotonically with sample size, despite
the head F1 increases. Baseline F1 is also not monotonic between 250 and 1,000 documents.
These are descriptive single-seed fitting comparisons, not significance or held-out claims.
Changing class/source proportions and additional recipe/context supervision remain explicit
parts of the learning-curve protocol.

| Training documents | Calibration miss target | Selection discarded / 837 | Workload reduction | Critical misses / 552 | Miss rate | Unclear references discarded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 1% | 21 | 2.51% | 3 | 0.54% | 11 |
| 250 | 5% | 87 | 10.39% | 15 | 2.72% | 46 |
| 1,000 | 1% | 20 | 2.39% | 2 | 0.36% | 8 |
| 1,000 | 5% | 110 | 13.14% | 30 | 5.43% | 32 |
| 4,000 | 1% | 18 | 2.15% | 4 | 0.72% | 1 |
| 4,000 | 5% | 51 | 6.09% | 23 | 4.17% | 3 |

The 1,000-document 5% target is exceeded on selection. The 250-document policies discard many
unclear references despite meeting the review-worthy miss targets on this fitting cohort.
Only predicted routine rows are eligible, so true unclear references can still be discarded
through classification errors. Both outcomes are reported; fitting targets do not establish
population risk guarantees or justify selection-cohort threshold retuning.

The 250/1,000-document training phases, including epoch validation, took 1,424/2,406 seconds
over 96/375 optimizer steps. Subsequent calibration, prediction and timing are additional.
On the common 128-document warm sample they measured 253/252 ms mean latency and
1,062/1,062 ms P95, including text processing and excluding loading. Peak allocated training
VRAM was 2.34 GiB for each. These timings describe the shared local machine.

A job-identifier inconsistency treated weight 0 and 0.0 differently, unnecessarily repeating
the 1,000-document recipe screen as a curve fit. The screen scored 0.539 and the designated
curve scored 0.544 despite identical captured configurations, training identities, source/version
fingerprints, pre-update weights, Torch/CUDA RNG states and pre-update probabilities. Both fits
are retained; the curve uses its designated run, without selecting the better duplicate. The
specific cause of the fit difference has not been isolated, and this pair does not estimate
training variance. A fixed seed alone does not enable all deterministic algorithms; see the
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
Paired bootstrap intervals remain conditional on the frozen fitted models.

Future studies now normalize weight to the CLI's float type before forming the job identifier
and arguments. Offline checks confirm identical integer/float identities across screen/curve
roles for head/LoRA, with no GPU job launched; Ruff and compilation pass. This launcher-only
fix remains deferred in the review worktree until the live model freeze and test finish.
Existing study identifiers, snapshots and artifacts are retained under their original pins.

There are 58 completed jobs and no failures at this snapshot. The 16,000-document head point
is running on the checked nested prefix with the same selected recipe; 25,000 follows.
LoRA and language-model adaptation, the remaining sample-efficiency matrix, model/policy
freeze, held-out test, qualified human audit and final report/reproducibility review remain pending.

## Larger head curve verified: 2026-10-12, 11:36 UTC

The bounded runner completed the 16,000-document Laya head point and started the single planned
1,000-document LoRA pilot. The head retained the selected 1,024-token context, unweighted
cross-entropy, learning rate 0.00003 and seed 42. Raw selection F1 selected epoch three.
The unchanged training prefix hash is
`4d41eb45714cd91b7c6a0d940d1d8f9eaf3a19a259bacdb085127d4f255ec97a`.
Completion hashes, finite saved weights, optimizer/RNG history, all 6,000 optimizer steps,
source/runtime pins, ordered fitting cohorts and independently recomputed metrics passed.
The deliberate checkpoint continuation is recorded once in the training history.

| Completed 16,000-document recipe | Selection macro F1 | Review-worthy -> routine / 552 |
| --- | ---: | ---: |
| Laya head, selected epoch three | 0.661 | 131 |
| Combined TF-IDF + linear SVM | 0.766 | 67 |
| Word TF-IDF + logistic regression | 0.759 | 61 |
| Frozen FinBERT + logistic regression | 0.716 | 84 |
| Frozen BGE + logistic regression | 0.691 | 77 |

All nine matched baseline artifacts, training hashes and fitting-cohort metrics were checked.
The head gains 0.299 macro F1 over its own matched pre-update control (0.362), but the lexical
baseline remains higher. Across the completed 250/1,000/4,000/16,000 head points, F1 rises from
0.364/0.544/0.582 to 0.661; critical misses are 76/67/105/131 and do not improve monotonically.
These are descriptive validation-set teacher-agreement results, with additional recipe/context
supervision and changing training composition; they are not held-out or significance claims.

The separate calibration cohort has macro F1 0.684 and 57 critical misses among 252 review-worthy
references. Its empirical 1%/5% miss-target policies discard four/30 calibration documents,
with two/twelve critical misses and zero/one unclear references discarded. On selection they
discard 27/84 of 837 documents (3.23%/10.04% workload reduction), with four/24 critical misses
(0.72%/4.35% of 552) and two/three unclear references discarded. Thresholds were not retuned on
selection, and these fitting-cohort constraints do not establish population safety.

The recorded training phase took 31,591 seconds (8.78 hours), including epoch validation and the
checkpoint continuation but excluding final calibration/selection and performance evaluation.
Training peak allocation was 2.35 GiB. On the common 128-document warm inference sample, mean
latency was 251.9 ms, P50 136.4 ms and P95 1,058.4 ms; inference peak allocation was 2.09 GiB.
The raw timings, quantiles, throughput arithmetic and common sample identities were verified.
Timings include text processing, exclude model loading, and come from the shared local machine.

The 25,000-document head run remains deferred. Only the LoRA pilot is active; compute still stops
at 18:00 London for the 19:00 review handoff. No model freeze or held-out test has started.

## LoRA pilot verified and compute stopped: 2026-10-12, 17:00 UTC

Both bounded review jobs completed; the runner stopped at 12:47 UTC (13:47 London). At this
verification checkpoint, all 60 recorded jobs are complete, no research training processes remain,
the GPU lock is available, and no model freeze or test evaluation has started. No larger run starts.

The Laya LoRA pilot used 1,000 training documents, 1,024-token whole-document pooling,
unweighted cross-entropy, rank 8, learning rate 0.0001 and seed 42. Epoch three was selected.
Completion hashes, finite weights, optimizer/RNG checkpoints, all 375 steps, source/runtime pins,
canonical training identities and fitting-cohort predictions passed independent checks. Its
pre-update probabilities exactly match the designated head curve on all 837 selection documents.

| Matched 1,000-document recipe | Selection macro F1 | Critical misses / 552 |
| --- | ---: | ---: |
| Word TF-IDF + linear SVM | 0.675 | 99 |
| Laya LoRA, selected epoch three | 0.656 | 104 |
| Laya head, selected epoch three | 0.544 | 67 |

LoRA gains 0.112 macro F1 over the matched head, but adds 37 critical misses and remains below
the lexical baseline. Both adaptations start from identical predictions; their learning rates
differ, so this is a recipe comparison, not an isolated causal LoRA effect. The training size
does not include additional labels used for recipe/context selection. This single pilot cannot
establish a LoRA learning curve or a model-family ranking on held-out financial documents.

The separate calibration result is F1 0.660 with 52 critical misses among 252 review-worthy
references. The 1%/5% policies discard eight/33 calibration documents with two/twelve critical
misses and no unclear references discarded. On selection they discard 28/91 documents
(3.35%/10.87% workload reduction), with seven/36 critical misses among 552 (1.27%/6.52%).
Neither achieves its miss target on selection. No unclear references are discarded there.
The thresholds remain calibration-selected and were not retuned after observing this failure.

The training clock is 3,929 seconds (1.09 hours), including epoch validation and excluding final
calibration/selection and performance measurement. Training peak allocation is 3.18 GiB.
On the common 128-document warm sample, mean inference is 266.5 ms, P50 143.0 ms and P95
1,123.7 ms; inference peak allocation is 2.10 GiB. The matched word/SVM baseline averages 4.4 ms
on the same sample. Both timings include text processing and exclude loading; TF-IDF uses CPU
and Laya the local RTX 3060, with shared-machine measurements. All raw timing calculations passed.

The [review handoff](review-checkpoint.md) brings the current findings and five-PR review order
together. The larger matrix, final selection/freeze, held-out evaluation, fresh qualified human
audit, actual publication report/PDF and remaining reproduction review await replanning.

## 13 October: bounded 4k Laya LoRA complete

The [core study](core-study.md) supersedes the earlier unrestricted matrix. It adds three predeclared
4k LoRA recipes and defers larger/generative adaptation and independent human review. The first fit
completed at approximately 01:43 London time; ModernBERT started immediately afterward, with FinBERT next.

Laya's 4k LoRA selects epoch three with selection macro F1 **0.7272074842**, 81 critical misses among
552 review-worthy references, and calibration F1 **0.7303494269**, 29 misses among 252. Word TF-IDF/logistic
at the identical 4k subset scores **0.7257928066**, with 84 selection misses. The LoRA recipe's gain over
the designated 4k head recipe is 0.1447129308; its 1k-to-4k increment is 0.0713337415. These are fitting
teacher-agreement scores, not held-out or significance findings. Sampling composition and additional
tuning labels qualify sample-efficiency interpretation.

The 1%/5% rules discard 17/49 calibration documents with two/twelve critical misses (0.79%/4.76%).
On selection they discard 67/162 (8.00%/19.35%), with twelve/44 critical misses (2.17%/7.97%) and
one/four unclear references discarded. Both selection miss rates exceed their calibration targets;
thresholds remain unchanged. Training takes 13,962.542 recorded seconds (3.88 hours), with 1,500 steps
and 3.18 GiB peak allocation. Warm mean is 267.8 ms versus 4.52 ms for word/logistic on the identical
128-document cohort; P50/P95 are 144.0/1,127.9 ms, with 2.11 GiB inference peak.

Completion/source/runtime checks, ordered cohorts, probabilities, F1/confusion/miss arithmetic,
calibration identity, raw best-epoch selection, finite best/zero/final checkpoints, discard rules and
common timing arithmetic all passed CPU checks. The 4k and original 1k LoRA epoch-zero trainable tensors
are identical, and their ordered selection probabilities have maximum difference zero. This proves the
deduplicated matched control is appropriate. The receipt and artifact hashes are retained locally in
`checks/core-laya-lora4000-verification-20261013.json`. No real test inference was performed. The remaining
two fits, freeze/test evaluation, actual report/PDF review and final reproduction audit remain pending.

## 13 October: bounded ModernBERT-large LoRA complete

The pinned `answerdotai/ModernBERT-large` run at 4k selects epoch three with selection macro F1
**0.7009685480**, 123 critical misses among 552, and calibration F1 **0.6752113930**, 60 misses among 252.
The 1%/5% policies discard 19/53 calibration documents, with two/twelve critical misses; on selection
they discard 45/135 (5.37%/16.13%), with seven/35 critical misses (1.27%/6.34%) and zero/one unclear
references discarded. Both selection rates exceed their targets. These remain fitting teacher-agreement
results under the predeclared recipe; no held-out architecture advantage is inferred.

Training/validation takes 7,813.667 seconds (2.17 hours), 1,500 steps, with 2.88 GiB training peak.
The common warm sample averages 147.4 ms, P50/P95 72.2/525.4 ms, and 1.81 GiB inference peak.
All source/runtime/cohort, raw metric, best/zero/final checkpoint, policy and timing checks passed;
the receipt is `checks/core-modernbert-lora4000-verification-20261013.json`.

Source review confirms that the native SDK supplies an actual three-choice distribution. Adapted Laya
averages logits from 50% overlapping windows; ModernBERT/FinBERT average unpadded token states within
non-overlapping windows and uniformly average their window vectors before a new triage classifier.
The frozen FinBERT/BGE logistic baselines normalize mean window vectors; BGE uses CLS. Laya normalizes
literal reserved mask tokens before encoding. These representation/window/head differences make this a
system comparison. Trainable counts also differ: Laya LoRA 27,165,697 versus ModernBERT-large LoRA
922,627, including heads. The report now records exact model IDs/revisions and neural parameter budgets.
FinBERT started next at approximately 03:58 London time. Its fit, matched controls, freeze/test, actual
report/PDF inspection and final reproduction audit remain required. No additional paid/cloud work ran.

## 13 October: bounded FinBERT LoRA complete; training matrix finished

The third predeclared 4k LoRA selects epoch three: selection macro F1 **0.6715172242** with 132
critical misses among 552, and calibration F1 **0.6541647895** with 58 among 252. Frozen FinBERT/logistic
on the identical training subset has F1 **0.6701110583** and 103 selection misses. A near-equal fitting
F1 therefore does not establish an operational improvement. Different heads and updating budgets qualify
the system comparison; no settings were changed after observing this result.

The 1%/5% policies discard 10/42 calibration documents with two/twelve critical misses. Selection
discards 45/120 (5.37%/14.34%), with twelve/44 critical misses (2.17%/7.97%) and zero/one unclear
references discarded. Both rates exceed their targets. Training/validation takes 1,958.567 seconds
(32.6 minutes), 1,500 steps, with 0.78 GiB training peak and 298,755 trainable parameters. Warm mean
is 33.9 ms, P50/P95 19.6/117.6 ms, with 0.52 GiB inference peak on the common 128-document sample.

All three fits passed source/runtime/artifact, ordered cohort/probability/raw metric, calibrated policy,
best/zero/final checkpoint and timing checks. The FinBERT receipt is
`checks/core-finbert-lora4000-verification-20261013.json`. Matched pre-update controls now run serially;
their verification, final freeze, one held-out evaluation and actual report/PDF/reproduction review remain.
The agreed 20-hour checkpoint is approximately 17:41 London time today; no cloud or extra API calls ran.

## 13 October: controls verified, study frozen, held-out inference started

All seven controls completed without OOM/failures; the fit worker exited and its temporary awake guard
released. CPU verification checked every control's exact copied checkpoint, finite weights, source/runtime,
ordered selection/calibration cohort, probability/F1/confusion/critical-miss arithmetic, calibration identity
and original epoch-zero probabilities after temperature scaling. Parameter digests match every one of the
eleven adapted/control pairings, including reused head curves and the shared 1k/4k LoRA control. All seven
sets of calibrated discard rules passed independent arithmetic/maximal feasible threshold checks.

The final freeze contains 64 comparisons, with validation-selected primary **4k Laya LoRA versus 4k word
TF-IDF/logistic**. All artifacts and plan/selection/state/data/source/version hashes were checked before
test access. The freeze is `0be83dda3a2936694d3768c1a15723fbd99b6cb895299d5458771ec32d6c664a`, timestamp
04:21:49 UTC. The tracked [declaration](reports/core-v1/frozen-comparisons.json) contains no held-out outcomes.
Receipts are `checks/core-controls-verification-20261013.json` and `checks/core-freeze-verification-20261013.json`.

Held-out inference began at 04:24 UTC, with a new temporary awake guard. The benchmark ownership and local
test markers both match the freeze. All choices remain locked; resumable prediction work is serial. Actual
test/statistics verification, report/PDF every-page review and final reproduction/code review remain pending.

## 13 October: verified held-out results and final report

All 64 frozen comparisons completed the single held-out pass at approximately 09:12 UTC. Independent audit receipt
checks/core-heldout-verification-20261013.json binds result SHA-256
330cc6dfb84c0067ffd910cc0f40713d2b8da4ad190b4aca9186303365df23d0 to the existing freeze. It checks identical
1,800-document/1,770-group cohorts, raw/calibrated probabilities and every source/family/provenance metric, policies,
confusions/critical misses/support, cache/timing arithmetic, completion hashes and all 2,000-draw intervals/contrasts.
Freeze/model/data/source/version guards still pass; the GPU worker and its awake guard are terminal.

Primary 4k Laya LoRA F1 is 0.712061 versus word TF-IDF/logistic 0.701065. Paired delta is 0.010996 with interval
[-0.013308, 0.035814], so this cohort does not resolve an F1 advantage. Critical misses are 177 versus 187 among
1,292 teacher review-worthy references. Laya's 1%/5% rules discard 5.39%/14.72% with 1.32%/6.58% miss rates.
ModernBERT-large LoRA F1 is 0.689185/267 misses; FinBERT LoRA is 0.641920/306 versus frozen FinBERT/logistic
0.666079/236. Laya LoRA's 1k-to-4k gain is 0.071 with paired interval [0.049, 0.093]. No choices were retuned.

The actual 15-page report and all four figures passed every-page visual inspection, including repeated table headers.
Report-only repairs handle the native SDK's missing kind field, label the majority dummy correctly, distinguish
curve colors and add an evidence-based interpretation. Frozen inference/training sources and choices are unchanged.
The public package preserves exact logits/references, decisions, warm measurements, metric/statistics evidence,
source/execution archives and a CPU replay script; body text/checkpoints remain local, with data-access limits explicit.
Human review, larger/generative adaptation and RL/RLCD remain deferred. Review starts at core-study.md or reports/core-v1/report.md.

## 13 October: exploratory appendix after the primary results

The requested remaining-time extension uses saved predictions only: no fitting, API calls, threshold sweeps or new
selection. Recorded scope and selection-character quartiles are in reports/core-v1/diagnostic-inputs.json. The original
freeze/result hashes and primary conclusion are unchanged. diagnostics.py reconstructs class contributions, binary
review routing, critical-miss overlap, source/length contrasts and the ten original frozen-policy points.

The unclear class contributes 68.3% of the observed 0.010996 macro-F1 difference; all contribution intervals include
zero. Derived routing F1 is 0.7730 versus 0.7676, paired interval [-0.0176, 0.0281]. Laya discards 191 required-review
references versus 212 for TF-IDF, while sending 28 additional documents for review. Critical misses overlap in 105
documents; 72 are Laya-only and 82 TF-IDF-only. This is not measured ensemble performance. All four selection-defined
length-band intervals include zero. Source contrasts vary; absent unclear references limit four families' fixed-three-label
macro-F1 ceiling to 2/3. Source/length results are post-hoc, uncorrected for multiplicity and confounded.

Independent sklearn/row-weighted arithmetic verified all 2,000 draws per slice and class/routing intervals; the public
diagnostics-audit.zip preserves its source and passing receipt. The report now has 17 pages and six figures. Pages 1–14
are pixel-identical to the previously inspected renders; revised pages 15–17 and both new figures passed visual inspection.
The appendix interprets frozen evidence and proposes no new operating rule. Human review and fresh evaluation data remain
necessary for further model/policy development.

## 13 October: fixed-recipe 16k FinBERT development extension completed

The user authorized one additional job from the deferred grid. The standalone controller ran FinBERT LoRA at
16,000 documents from 11:12 to 13:37 UTC, finishing normally and releasing its awake guard. It used the unchanged
4k recipe: context 512, rank 8, seed 42, unweighted CE, both learning rates 0.0001 and three epochs. Both sizes
select epoch three. It did not restart the full study, clear ownership markers or evaluate the original holdout.

Selection macro F1 rises 0.671517 to 0.763617 (+0.092100); review-worthy-to-routine misses fall 132 to 67 among
552, while unclear-to-routine errors rise five to seventeen among 93. Default discards fall 286 to 242 of 837.
Calibration F1 is 0.654165 versus 0.805772 on fitting data. The 16k 1%/5% policies discard 63/133 selection
documents with eight/21 critical misses (1.45%/3.80%) and zero/four unclear references discarded. The 1% target
still fails on selection. No population bounds or new test-set winner are claimed.

Training/validation clocks are 32.6 versus 143.1 minutes (4.38x); new wall time is 2.41 hours. Warm means are
33.94 versus 34.02 ms on the same 128 documents. Training peak is 0.78/0.79 GiB; inference peak is 0.52 GiB.
Nested identities are exact, but raw characters grow 31,526,200 to 146,195,420, class/source proportions change
and optimizer steps grow 1,500 to 6,000. Two points cannot establish minimum label requirements or isolate data
quantity from additional optimization and composition. Epoch-selected selection and calibration are fitting
evidence; this job was chosen after seeing the primary test results and requires fresh held-out evaluation.

Independent verification and a development-only checkpoint replay passed. It checks source/runtime/artifact
identities, finite zero/best/final weights and matched initialization, ordered cohorts, nested train identities,
raw/calibrated metrics, calibration optimum/maximal feasible policies, selected epoch and resource clocks/sample.
Receipts and exact sources are in reports/core-v1/development-audit.zip; exported probabilities, train metadata,
decisions and histories support CPU replay. The report's separate section spans pages 17–19. Original freeze,
result, state and ownership hashes remain unchanged; the core still contains exactly 64 held-out comparisons.

## 13 October: compact BGE development comparison completed

The user authorized one further compact-encoder fit. BGE-small LoRA ran 14:36–14:58 UTC with the same
4k subset and 512-token/rank-8/unweighted-CE/0.0001 learning-rate/seed-42 recipe as 4k FinBERT. Three epochs
completed; raw selection F1 chooses epoch two. The controller released its awake guard normally.

Selection F1 is 0.658591 versus FinBERT LoRA 0.671517 and frozen BGE/logistic 0.645911. BGE's critical
misses are 84/552 versus 132/552 and 109/552, with default discards 219/837 versus 286/837 and 242/837.
The workload tradeoffs are explicit; unclear discards are seven/five/nine among 93. BGE's matched
epoch-zero selection F1 is 0.312025; training includes classifier-head and encoder-adapter updates,
and differing architecture/pretrained objectives/normalization/classifier recipes prevent causal attribution.

BGE train/validation takes 21.0 minutes (21.8 controller wall time), with 149,379 trainable parameters,
0.32 GiB training peak, 0.19 GiB inference peak and 20.23 ms warm mean. FinBERT uses 32.6 minutes,
298,755 parameters, 0.78/0.52 GiB and 33.94 ms. Timing samples match exactly; frozen BGE's mean is
20.40 ms. Frozen feature preparation/classifier fitting have separate accounting. BGE's 1%/5% rules
miss twelve/36 selection review-worthy documents (2.17%/6.52%) while discarding 41/91, with no unclear
discards. All three systems exceed both empirical targets on selection.

Independent artifact/source/runtime, original ordered-cohort, shared-subset, finite checkpoint and
epoch-selection, metric/temperature/maximal-policy and resource checks passed. Best and epoch-zero
BGE checkpoints were replayed on development data only. Published probabilities retain original and
common identity order; arithmetic/source/replay receipts are in compact-audit.zip. The new report section
is pages 19–20; all original core and completed FinBERT evidence hashes remain unchanged. These are
post-test fitting-cohort system comparisons, with no new held-out result or significance claim.

## 13 October 2026: editorial consolidation

Following the reader's request for a coherent report, the renderer now follows the research
questions rather than the order experiments were completed. Class/routing diagnostics sit
beside the held-out comparison; policy uncertainty accompanies calibration; source/length
and shared errors have one focused section. The two later fits share a development-only
section organized around label scaling and compact-encoder resource/workload tradeoffs.
Repeated tables, execution notes and selected error vignettes were removed from the narrative.
Full measurements, secondary contrasts, source records and audit receipts remain published.

The resulting PDF is eight pages, replacing the cumulative twenty-page version; page count
was an outcome of consolidation rather than a target. All eight pages and six unchanged
figures were freshly inspected. The full public CPU replay, Ruff and diff checks passed.
Scientific evidence bytes, the 64-comparison core freeze/results, benchmark ownership and
completed FinBERT/BGE artifacts are preserved. Only presentation/source hashes and navigation
were updated. Earlier page references in this chronological log describe earlier releases.

## 13 October 2026: LaTeX academic manuscript

The results are additionally written as a standard single-column academic manuscript
with an abstract, related work, explicit task/protocol and metric definitions, numbered
tables/figures, development-only extensions, discussion, conclusion and nine primary
references. The author order is Lucca Pereira Martins, then Matthew Ball; Lucca's supplied
email is the corresponding address. Affiliations were not supplied and no matching Matthew
email was found in available Git history, so neither is invented.

The twelve-page PDF compiled with Tectonic 0.17.0. All five tables (33 quantitative rows)
were checked against the frozen evidence, references resolve, fonts are embedded, and
the final compiler log has no overfull/underfull or undefined-reference diagnostics.
Every page and all six figures were inspected. The portable arXiv source archive contains
only the main TeX file and six byte-identical scientific figures; an extracted offline
build reproduces the page count and complete text. Source/build/output hashes and visual
receipts are in paper/. All 38 original public evidence files retain their exact bytes,
as do core/benchmark and completed development identities. No model, test inference,
paid API call or cloud job was dispatched. This prepares a manuscript for author review;
it does not constitute an arXiv submission.

## 13 October 2026: targeted prior-work audit

A broader primary-source search found substantial overlap that the initial related-work
section missed: Laya calibration/escalation on synthetic workflows, Laya-versus-baseline
financial complaint triage, and existing financial relevance/materiality filtering work.
The manuscript now cites the nine closest additional works and narrows its contribution
to an incremental public-source financial-domain comparison. The dated search, source
links, task differences, repository snapshots and limits are in prior-work-audit.md.
No exact match to the complete rubric/cohort/protocol combination was identified among
reviewed sources; that is not a proof of absence or grounds for a first-study claim.

The revised thirteen-page paper has eighteen references. All original scientific body,
five tables and six figures retain their prior content; the 38-file original evidence
package is byte-identical. Every final PDF page was inspected. The source archive
recompiles offline with identical text/page count and no layout/reference diagnostics.
Source/package hashes and visual/transcription receipts were updated. No model, test
evaluation, operating rule or paid computation was changed by the literature review.
