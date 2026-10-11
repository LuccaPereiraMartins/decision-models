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
