# Financial-document triage across public sources

## Main findings

At 4,000 training documents, Laya LoRA scores 0.7121 test macro F1 against 0.7011 for Word TF-IDF logistic. The paired difference interval [-0.013, 0.036] does not resolve an F1 advantage. Warm mean latency is 267.8 versus 4.52 ms: 59.2 times higher for Laya.

Adaptation improves agreement over an untrained decision head, and more training data can help. Neither finding removes the need for a strong lexical comparator. Longer contiguous context does not improve Laya head performance monotonically, and calibration miss targets do not transfer as test guarantees.

Later development fits address two remaining engineering questions: whether more labels help FinBERT, and whether a smaller general-purpose encoder offers a cheaper alternative. Their selection/calibration results are promising in different ways, but were obtained after the original test was inspected and cannot revise its conclusion.

All findings measure agreement with Luna rubric labels. Independent human review remains deferred; analyst-validated materiality, commercial-inbox usefulness and trading profitability are not established.

## Task, data and comparison protocol

The task is source-content-only inbox triage: routine, review-worthy or unclear. Review-worthy and unclear predictions go to review. The critical miss is a teacher review-worthy document predicted routine. Routine denotes ordinary administrative information; review-worthy denotes a potentially significant corporate, financial, economic or policy development; unclear denotes insufficient or conflicting content. No company history, portfolio context or subsequent market reaction is supplied. Macro F1 averages the three declared classes; it rewards classification agreement rather than directly measuring analyst workload.

The benchmark contains 25,000 training documents, 837 selection documents, 363 calibration documents and 1,800 test documents. Training ends in March 2026; selection/calibration cover April-June and test covers July-September. Event and verified near-duplicate groups cannot cross partitions. Previously experimented documents and linked events are excluded from fresh evaluation; duplicate retrieval remains approximate.

The core compares 64 frozen configurations on the same 1,800 test documents and 1,770 related-document groups. Main trained-model comparisons use the identical 4k subset. Model/epoch selection uses selection data; temperature and discard rules use calibration data. Every core choice froze before test inference. Paired 95% intervals use 2,000 shared whole-group bootstrap draws, conditional on one training seed; supplementary comparisons have no multiplicity adjustment.

| Source family | Train | Test | Test routine | Test review-worthy | Test unclear |
| --- | --- | --- | --- | --- | --- |
| 6k | 7134 | 561 | 105 | 297 | 159 |
| 8k | 7068 | 561 | 100 | 413 | 48 |
| ecb | 666 | 27 | 9 | 18 | 0 |
| fed | 1712 | 55 | 15 | 40 | 0 |
| news | 1373 | 35 | 10 | 25 | 0 |
| releases | 7047 | 561 | 62 | 499 | 0 |

EDGAR supplies 8-Ks, 6-Ks and release exhibits; other cohorts are official Fed, ECB and GOV.UK announcements. The test news cohort is entirely GOV.UK; Wikinews has no held-out coverage. SEC accounts for 1,683 of 1,800 test documents, so source diversity remains limited despite the larger corpus. Recorded dates are filing/publication dates at day precision; collected snapshots may differ from the text originally released. Pretraining exposure is also possible.

Methods and source rights are documented in [literature](https://github.com/LuccaPereiraMartins/systematic-trading/blob/091bf6e3b164e148bccb30bcc0f9180780f79c94/decision_models/literature.md) and [sources](https://github.com/LuccaPereiraMartins/systematic-trading/blob/091bf6e3b164e148bccb30bcc0f9180780f79c94/decision_models/sources.md). The frozen benchmark contains about $4.88 of retained teacher-label charges; pilots, failed requests and excluded labels also count against the separate project ledger. Hardware and human time are additional.

## Held-out model comparison

| Method | Train | Test macro F1 | 95% CI | Critical misses | Warm mean ms |
| --- | --- | --- | --- | --- | --- |
| Laya LoRA | 4000 | 0.7121 | [0.686, 0.735] | 177 | 267.77 |
| Word TF-IDF logistic | 4000 | 0.7011 | [0.674, 0.726] | 187 | 4.52 |
| Majority class | 4000 | 0.2786 | [0.274, 0.284] | 0 | 2.91 |
| Word TF-IDF svm | 4000 | 0.6834 | [0.656, 0.709] | 176 | 4.46 |
| Char TF-IDF logistic | 4000 | 0.6917 | [0.664, 0.716] | 230 | 8.79 |
| Char TF-IDF svm | 4000 | 0.6803 | [0.652, 0.705] | 207 | 8.63 |
| Combined TF-IDF logistic | 4000 | 0.7019 | [0.675, 0.727] | 183 | 10.70 |
| Combined TF-IDF svm | 4000 | 0.6981 | [0.671, 0.723] | 194 | 10.66 |
| FinBERT logistic | 4000 | 0.6661 | [0.641, 0.691] | 236 | 34.15 |
| BGE-small logistic | 4000 | 0.6394 | [0.613, 0.664] | 238 | 20.40 |
| Laya SDK | 0 | 0.3632 | [0.348, 0.379] | 321 | 277.76 |
| Laya head | 4000 | 0.5450 | [0.519, 0.570] | 231 | 253.86 |
| ModernBERT-large LoRA | 4000 | 0.6892 | [0.664, 0.713] | 267 | 147.43 |
| FinBERT LoRA | 4000 | 0.6419 | [0.616, 0.666] | 306 | 33.94 |

Critical-miss counts share a denominator of 1,292 teacher review-worthy documents. Unadapted references have no task training. The majority classifier predicts review-worthy throughout: its zero misses remove no review workload. The frozen FinBERT classifier outperforms the declared 4k FinBERT LoRA recipe on this test; that recipe comparison is not an exhaustive assessment of finance encoders or fine-tuning.

![Primary confusion matrices; rows are teacher references](confusion.png)

### Class agreement and review routing

A small three-class F1 difference need not translate into better review routing. The following decomposition was added after seeing the test results and is exploratory.

| Class | Laya class F1 | TF-IDF class F1 | Contribution to macro-F1 difference | 95% CI |
| --- | --- | --- | --- | --- |
| routine | 0.637 | 0.632 | 0.0017 | [-0.010, 0.013] |
| review_worthy | 0.848 | 0.843 | 0.0018 | [-0.004, 0.007] |
| unclear | 0.651 | 0.628 | 0.0075 | [-0.008, 0.022] |

About 68% of the point difference comes from unclear. Every class-contribution interval includes zero. Collapsing the existing predictions into discard versus review gives routing F1 0.773 versus 0.768, with paired difference interval [-0.018, 0.028]. Laya discards 191 versus 212 required-review references among 1,499, while sending 28 more documents for review. These routing and miss-rate differences remain unresolved; the two-class view does not replace the primary endpoint.

## Confidence and automatic discarding

Temperature scaling changes confidence without changing argmax labels. Lower calibration error does not itself establish that automatic discarding meets a miss-rate target.

| Primary method | Raw Brier | Calibrated Brier | Raw ECE | Calibrated ECE |
| --- | --- | --- | --- | --- |
| Laya LoRA | 0.303 | 0.300 | 0.049 | 0.034 |
| Word TF-IDF logistic | 0.313 | 0.304 | 0.067 | 0.028 |

Brier sums squared errors over three classes (0-2); ECE uses ten confidence bins. Reported confidence is model class confidence, not the probability of investment materiality.

| Method | Calibration target | Test workload removed | Critical misses / 1292 | Test miss rate |
| --- | --- | --- | --- | --- |
| Laya LoRA | 1% | 5.39% | 17 | 1.32% |
| Laya LoRA | 5% | 14.72% | 85 | 6.58% |
| Word TF-IDF logistic | 1% | 6.72% | 16 | 1.24% |
| Word TF-IDF logistic | 5% | 13.44% | 65 | 5.03% |

Thresholds maximize calibration workload removal subject to empirical 1%/5% review-worthy miss targets; predicted review-worthy/unclear always escalate. Laya exceeds both targets on test. This is the operational limit of the observed confidence gains: an analyst cannot interpret the target name as an achieved deployment bound. Zero observed errors on a small calibration set would also leave unseen errors unmeasured.

![Frozen policy tradeoffs and marginal group-bootstrap intervals](diagnostic-policies.png)

The ten plotted points reuse the original frozen policies. Circles/squares denote 1%/5% targets; bars are separate marginal 95% intervals, not joint regions. At 1%, Laya removes less workload than TF-IDF while the miss-rate difference is unresolved. At 5%, the nominal miss-rate-difference interval is unfavorable to Laya, while the workload difference remains unresolved. These conditional comparisons support interpretation, not a new rule selected on test.

## Adaptation, training data and context

Laya adaptation uses uniform whole-document pooling and overlapping windows; the SDK uses its native most-confident-window aggregation. Matched pre-update controls therefore matter. Encoder alternatives use fresh triage heads; FinBERT sentiment labels are not mapped onto the task. Different tokenization, pooling and head budgets make these system comparisons.

The three new core LoRA fits use the same 4k subset, unweighted cross-entropy, rank 8, alpha 16, dropout 0.05, head/adapter learning rates 0.0001 and seed 42, allowing at most three epochs. Raw selection F1 chooses the epoch. Laya and ModernBERT use context 1,024; FinBERT uses 512. Pretrained revisions, earlier head recipes and exact configurations are retained in the package.

| Method | Matched-zero test F1 | Adapted test F1 | Paired gain 95% CI |
| --- | --- | --- | --- |
| Laya LoRA | 0.364 | 0.712 | [0.317, 0.377] |
| Laya head | 0.364 | 0.545 | [0.151, 0.211] |
| ModernBERT-large LoRA | 0.096 | 0.689 | [0.569, 0.617] |
| FinBERT LoRA | 0.097 | 0.642 | [0.521, 0.568] |

Gains over a fresh untrained head establish supervised task learning, rather than proving adapter superiority over a trained frozen-feature classifier. Rank 8 updates about 27.166M parameters including Laya's native head, versus 0.923M for ModernBERT and 0.299M for FinBERT. Equal rank does not equalize the adaptation budget.

![Frozen sample-efficiency curves on the same test cohort](learning-curves.png)

| Method | Training increment | Test F1 gain | Paired 95% CI |
| --- | --- | --- | --- |
| Word TF-IDF logistic | 4,000 to 16,000 | 0.084 | [0.061, 0.109] |
| Word TF-IDF logistic | 16,000 to 25,000 | 0.002 | [-0.013, 0.016] |
| Laya LoRA | 1,000 to 4,000 | 0.071 | [0.049, 0.093] |

Laya LoRA improves from 1k to 4k within its fixed recipe. Lexical methods also benefit from more data; the word-logistic gain from 16k to 25k is unresolved. Subsets are nested, but class/source allocation changes as strata exhaust. Recipe screening and 1,200 selection/calibration labels are additional to the stated training sizes, so these curves do not identify a universal minimum label requirement. Exact allocations and all increments remain in the evidence package.

![Contiguous context ablation under fixed whole-document pooling](context.png)

Laya head context increases from 512 to 1,024 tokens yield a paired test-F1 difference interval [-0.014, 0.031]; 512 to 4,096 yields [-0.052, -0.006]. The pattern is not monotonic. Wider windows change contiguous context and the number of pooled votes; all normalized source tokens remain represented at every size. The supplementary, uncorrected head experiment neither isolates context from pooling effects nor establishes a LoRA context result.

## Throughput and resource cost

![Held-out agreement versus local warm throughput](throughput.png)

Timings use the same 128-document selection sample on an RTX 3060/CPU machine, including preprocessing and all source windows but excluding loading. Sklearn uses native document batching; neural document batches are serial with window batching. Hardware, electricity and acquisition costs are outside the recorded API charges.

| Neural method | Train/validation h | Training peak GiB | Inference peak GiB | Warm P95 ms |
| --- | --- | --- | --- | --- |
| Laya LoRA | 3.88 | 3.18 | 2.11 | 1127.9 |
| Laya head | 1.99 | 2.35 | 2.09 | 1065.1 |
| ModernBERT-large LoRA | 2.17 | 2.88 | 1.81 | 525.4 |
| FinBERT LoRA | 0.54 | 0.78 | 0.52 | 117.6 |

Training clocks include validation, excluding startup and final calibration/resource work. Frozen-feature preparation and classifier fitting have separate accounting; cache reuse means a classifier fit clock alone understates encoder preparation. Detailed batch throughput, feature clocks, checkpoint sizes and parameter counts are exported with the measurements.

## Where errors occur

The following source/length cuts and overlap analysis are exploratory, added after the primary results. They explain the current comparison without changing its selection.

![Primary F1 differences by source and document length](diagnostic-slices.png)

Bars are paired whole-group 95% intervals within each slice, retaining three fixed labels. Families without unclear references have an attainable macro-F1 ceiling of 2/3; compare methods within a slice rather than raw F1 across families. The 6-K difference has a nominal positive interval, with no multiplicity correction. All four length-band intervals include zero. Character cutpoints 2,695/4,589/9,835 come from selection quartiles; they are not token budgets. Source and length are confounded, so the figure is not a causal context test.

Both primary methods miss 105 teacher review-worthy documents; 72 are Laya-only misses and 82 are TF-IDF-only. Complementary errors suggest a possible future cascade, but do not measure its false-positive, workload or compute tradeoffs. Shared errors may also reflect questionable teacher references. Independent review is needed to distinguish those cases; high confidence alone does not settle them.

## Development evidence: more labels and a smaller encoder

These later fits answer two follow-up questions rather than extend the original test leaderboard. They were chosen after the test results were inspected. The 837-document selection cohort chooses epochs, and the 363-document calibration cohort fits confidence/policies; both are fitting data. The table below must not be compared directly with the held-out F1 table.

All neural fits start afresh from pinned weights, with uniform pooling, context 512, rank 8, alpha 16, adapter dropout 0.05, unweighted CE, learning rate 0.0001 and seed 42. Three epochs are allowed, including epoch zero in selection. Both FinBERT sizes choose epoch three; BGE chooses epoch two. Architecture and classifier recipes still differ.

| Method / train size | Selection F1 | Critical misses / 552 | Unclear discarded / 93 | Default discards / 837 | Warm mean ms |
| --- | --- | --- | --- | --- | --- |
| FinBERT LoRA (4000) | 0.6715 | 132 | 5 | 286 | 33.94 |
| FinBERT LoRA (16000) | 0.7636 | 67 | 17 | 242 | 34.02 |
| BGE-small LoRA (4k) | 0.6586 | 84 | 7 | 219 | 20.23 |
| Frozen BGE/logistic (4k) | 0.6459 | 109 | 9 | 242 | 20.40 |

### Does more data help the fixed FinBERT recipe?

Moving from 4k to 16k raises selection F1 by 0.0921 and reduces critical misses from 132 to 67. It sends 44 more documents for review and discards twelve additional unclear references. Training/validation grows from 32.6 to 143.1 minutes (4.38 times), while inference latency is approximately unchanged. More examples, 1,500 versus 6,000 optimizer steps and changing class/source proportions vary together. This promising fitting signal warrants fresh evaluation; two points do not establish a minimum data requirement.

### What does the compact encoder trade off?

At the matched 4k size, BGE LoRA has slightly lower selection F1 than FinBERT but lower warm latency (20.23 versus 33.94 ms) and memory (0.32 versus 0.78 GiB training peak; 0.19 versus 0.52 GiB inference peak). Its 84 versus 132 critical misses come with 67 extra documents for review and two extra unclear discards. Against frozen BGE, the small F1 gain and 25 fewer critical misses likewise come with 23 extra reviews. It is a resource/workload tradeoff, not evidence of operational dominance.

BGE training/validation takes 21.0 minutes and updates 149,379 parameters versus FinBERT's 298,755. Its matched epoch-zero F1 is 0.3120; the gain includes learning a fresh classifier as well as encoder adapters. Neither finance pretraining nor adapter updates are isolated by these different systems.

The 16k FinBERT 1% policy still misses 1.45% of selection review-worthy references. All three matched-4k systems exceed both calibration targets on selection. This echoes the core study's warning about target transfer. No additional test result or independent significance claim is made, and the original held-out conclusions remain unchanged.

## Interpretation, limits and reproduction

For this teacher-labeled public-source task, lexical classification remains a strong throughput baseline. Decision-model adaptation learns the task and benefits from more data, but the primary study does not resolve an F1 advantage that offsets its much higher measured latency. Neither longer context nor calibrated confidence establishes a reliable automatic-discard rule.

The study is limited by teacher references without independent human adjudication, one trained seed, approximate duplicate detection, institutional-source dominance and possible historical-text or pretraining exposure. Bootstrap intervals measure cohort sampling conditional on the fitted models, not training instability. Supplementary and post-hoc comparisons are uncorrected for multiplicity. The later encoder fits use development data and support choosing a fresh evaluation question, rather than choosing a deployment winner.

A stronger follow-up needs independently reviewed labels and fresh, representative held-out documents, then validation-selected methods and policies. Generative adaptation and RL/RLCD remain unmeasured. No claim about trading returns or analyst-validated materiality follows from agreement alone.

The [evidence package](https://github.com/LuccaPereiraMartins/systematic-trading/blob/091bf6e3b164e148bccb30bcc0f9180780f79c94/decision_models/reports/core-v1/README.md) supplies all 64 frozen comparisons, class/source metrics, paired intervals, exact development probabilities/decisions, parameter/resource records and independent audit receipts. `python decision_models/reports/core-v1/verify.py` replays the published arithmetic on CPU without weights, bodies or paid calls. Full training/inference replay additionally requires preserved corpus/checkpoints and pinned dependencies; [reproduce.md](https://github.com/LuccaPereiraMartins/systematic-trading/blob/091bf6e3b164e148bccb30bcc0f9180780f79c94/decision_models/reproduce.md) documents those access and source-rights limits.
