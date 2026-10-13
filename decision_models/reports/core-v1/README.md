# Bounded financial-document triage study

Start with [the report](report.md) or [its PDF](report.pdf). The first page gives the findings; later sections contain source coverage, class support, critical misses, calibration, learning/context curves and measured local resources.

Pages 15–17 of the 20-page PDF add exploratory diagnostics calculated after the primary results: class contributions, discard/review routing, shared critical misses, source/length slices and uncertainty around existing policies. Approximately 68.3% of the small observed F1 difference comes from the unclear class, but all class-contribution intervals include zero. The derived routing comparison and all four length bands also have unresolved F1 differences. These additions change no model, threshold or primary conclusion.

Pages 17–19 separately report a later, user-authorized FinBERT LoRA development fit at 16k documents. Against its fixed 4k recipe, selection F1 rises from 0.6715 to 0.7636 and teacher review-worthy misses fall from 132 to 67 among 552. Discarded unclear references increase from five to seventeen; training/validation takes 4.38 times longer while warm mean latency stays approximately unchanged. Selection and calibration are fitting cohorts, and this job was chosen after inspecting the primary test results. It is not a 65th held-out comparison or a revised test-set winner. See [the extension notes](../../development-extension.md) for the full boundary.

Pages 19–20 add the separate [BGE-small development comparison](../../compact-encoder-extension.md). At the same 4k training size, selection F1 is 0.6586 versus FinBERT LoRA 0.6715 and frozen BGE/logistic 0.6459. BGE LoRA's fewer critical misses come with more review workload; its warm mean is 20.23 ms versus FinBERT's 33.94 ms. All three systems exceed both calibration miss targets on selection. This post-test fitting evidence does not establish a new generalization result or isolate finance pretraining/LoRA effects.

All 64 comparisons use the same 1,800 held-out documents and 1,770 event/near-duplicate groups. The primary comparison was selected on validation at 4,000 training documents: Laya LoRA versus word TF-IDF/logistic regression. Laya's test macro F1 is 0.7121 versus 0.7011, with paired delta interval [-0.0133, 0.0358]. This does not resolve an F1 advantage. Its warm mean latency is 267.8 ms versus 4.52 ms on the common selection sample. Default critical misses are 177 versus 187 among 1,292 teacher review-worthy documents.

FinBERT is included as both frozen embeddings/logistic regression and 4k LoRA. Their observed F1 scores are 0.6661 and 0.6419, with 236 and 306 critical misses. These are declared recipes, not an exhaustive search over finance encoders or fine-tuning methods. Larger completed curves are supplementary; the main trained-model comparison remains at 4k.

References are Luna rubric labels. Independent human review is deferred. Intervals condition on one training seed; they do not measure training variability or correct for multiple supplementary comparisons. Institutional announcements dominate, and Wikinews has no held-out coverage. Analyst-validated materiality, commercial-news generalization and trading profitability remain unestablished.

## Recompute the published evidence

With the project's recorded Python dependencies installed, run from the repository root:

```powershell
python decision_models/reports/core-v1/verify.py
```

This uses CPU only. It verifies package byte hashes, rebuilds all 64 raw/calibrated metric breakdowns and frozen discard decisions, and reproduces the shared 2,000-draw group-bootstrap intervals and contrasts. It also replays the exploratory diagnostics and their paired intervals, plus the separate development metrics and calibration rules. It needs neither model weights nor document bodies, makes no API calls, and writes no study state. The publication directory preserves exact bytes across Windows/Linux checkouts.

| File | Contents |
| --- | --- |
| `frozen-comparisons.json` | Pre-test choices, roles, recipes and artifact identities |
| `predictions.json.gz` | Ordered cohort metadata, teacher references, all raw logits and per-document inference clocks; no document bodies |
| `comparison-details.json.gz` | Configurations, fixed decisions, full warm timing measurements and recorded training histories |
| `evaluations.json.gz` | All raw/calibrated overall and source/family/provenance metric breakdowns |
| `results.json.gz`, `bootstrap.json` | Original result summary, critical-miss identities, intervals and declared paired contrasts |
| `audit-evidence.zip` | Exact independent arithmetic verifier/receipts, freeze/plan/state, source archives and execution receipt |
| `diagnostic-inputs.json`, `diagnostics.json`, `diagnostics-source.py` | Recorded post-hoc scope, selection character lengths, calculations and exact analysis source |
| `diagnostic-slices.png`, `diagnostic-policies.png` | Source/length contrasts and the ten existing policy points with intervals; the report has six figures in total |
| `diagnostics-audit.zip` | Separate sklearn/row-weighted verifier and receipt for exploratory arithmetic and all 2,000 draws within each slice |
| `development-evidence.json.gz`, `development-summary.json`, `development-source.py` | Separate 4k/16k train metadata, selection/calibration probabilities and metrics, decisions, histories, measurements and descriptive calculation source; no new test predictions |
| `development-audit.zip` | Independent arithmetic/checkpoint replay receipts, development logits, exact run/configuration/source metadata and bounded runner; no corpus bodies or checkpoint tensors |
| `compact-evidence.json.gz`, `compact-summary.json`, `compact-source.py` | Matched-4k BGE/FinBERT development comparison, original/canonical identity orders, probabilities, decisions, histories and measured resources |
| `compact-audit.zip` | Independent metric/policy/source/cohort checks, best and matched epoch-zero checkpoint replay receipts/logits and exact bounded-run/source metadata; no bodies or weights |
| `report-source.py`, `report-manifest.json` | Exact renderer bytes, package versions and report-output hashes |
| `package-manifest.json`, `visual-verification.json` | Published-file hashes and every-page/figure review evidence |

The independent audits used sklearn class metrics and separately constructed resampling calculations, not the evaluator's bootstrap implementation. The smaller public replay script reuses the project functions to make the release easy to check. Neither establishes independent truth of the teacher references. Source/length intervals are exploratory, conditional on one training seed and uncorrected for multiplicity; source and length are confounded. Source scores retain three fixed labels, so families with no unclear references have a macro-F1 ceiling of 2/3. The retained `reference-quote.json` is an unexecuted scenario from the renderer, not a measured LLM comparison or additional spending.

## Training and data availability

Metric/statistics replay is available from this Git package. Exact training/inference replay additionally needs the preserved benchmark inputs, original/core run directories and pinned pretrained weights described in [reproduce.md](../../reproduce.md). The complete text corpus and model checkpoints are local artifacts, not included here. A live recollection may produce different text; a Git clone alone cannot reproduce the exact historical training set.

The exploratory appendix is opt-in for local report regeneration: preserve this package's `diagnostic-inputs.json` in the core run's `report/` directory before running the existing command with `--stage report`. Its freeze, primary/reference and ordered selection-length inputs are checked. Without that file, the renderer builds the original core sections alone; neither path fits models or changes study decisions.

Preserve `development-evidence.json.gz` in that same directory to include the development section. Its original freeze identity, fixed-recipe identities, nested training inputs and shared development cohorts are checked. The independent audit replays the completed 16k checkpoint on selection/calibration alone; public CPU replay reconstructs its metrics from the exported probabilities. Raw probabilities invert the saved scalar temperature, with checkpoint replay verifying the 16k values. Exact development training replay additionally needs the frozen inputs and recorded pretrained revision; keep it separate from the completed core study and use fresh test data for generalization.

Likewise preserve `compact-evidence.json.gz` and the recorded `compact.py` source for the compact section. CPU replay checks raw/calibrated probabilities, every metric, calibration policies and original calibration identity order for all three systems, plus matched epoch-zero arithmetic. Full BGE checkpoint/training replay requires the preserved run and local corpus; its independent audit used development cohorts only. All original held-out and completed FinBERT evidence bundles retain their earlier hashes.

Preserve the source rights and attribution in [sources.md](../../sources.md) with any corpus transfer. The local `benchmark-inputs.zip` contains the six exact input files and has SHA-256 `4800cdd95bfe1e1f02f28dfdf4bc6e4acf89ac5f6e9e335f735af18b740d5fbb`. Its absence from the public package is explicit; URLs and collection code do not guarantee byte-identical reconstruction.

The freeze SHA-256 is `0be83dda3a2936694d3768c1a15723fbd99b6cb895299d5458771ec32d6c664a`. It was finalized at 04:21 UTC on 13 October 2026 before held-out inference began at 04:24 UTC. The declaration was pushed after inference started, so this is an enforced pre-test freeze, not a claim of public preregistration. Inference finished at approximately 09:12 UTC. No model, epoch, calibration or threshold was changed using held-out outcomes.

The current holdout is closed to further selection. Larger LoRA, generative LLM adaptation, BGE adaptation, RL/RLCD and human review are deferred. Further model/policy development needs a fresh evaluation cohort.
