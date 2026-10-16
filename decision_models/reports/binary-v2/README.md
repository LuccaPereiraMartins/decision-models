# Binary financial-document triage evidence

The completed study contains seven core systems at shared 4k training data,
word TF-IDF/logistic and frozen FinBERT/logistic curves at 4k/16k/25k, and both
FinBERT and Laya LoRA at 4k/16k. The 16 frozen comparisons include three matched
initial-state controls. See [the paper](../../paper/paper.pdf), especially
the results and routing-policy tables.

The dataset has exactly 25,000 training, 1,000 selection, 500 calibration and
2,000 test documents. All references are teacher labels. Previously unclear
references received fresh binary decisions; definite references were retained.
The test reuses 1,800 earlier bodies, so it is not fresh independent confirmation.
SEC accounts for 1,870/2,000 test documents. The balanced 200-document training-only
audit measures teacher consistency (93%), not analyst validity.

Conservative incremental Luna charges total **$0.4430207 against the $2 cap**.
Preparation receipts retain their historical stage status; the completed
publication checks are recorded separately in `publication-checks.json`.

## Replay

From the repository root:

```sh
python decision_models/reports/binary-v2/verify.py
```

This requires Python and NumPy only. It independently checks cohort/order and
training identities, binary confusion/F1/false-negative/workload arithmetic,
raw and calibrated test probabilities, fitting histories and checkpoint/grid
selection, calibration temperature and maximal whole-tie policies, timing,
composition and paired group-bootstrap arithmetic. It verifies every published
file against the package manifest. It performs no fitting, inference or API calls.

To regenerate numerical tables/figures, run `python decision_models/report.py`.
See the [study guide](../../README.md#reproduce-the-published-evidence) for rendering,
compilation and exact fitting replay. The canonical paper has eight pages; older
editions and their original build records remain in Git history.

## Contents and boundaries

- `evidence.json.gz`: cohort metadata, public document IDs/URLs, ordered predictions,
  raw test logits, references, policies, fitting histories, resources and intervals.
- `freeze.json`: the exact revised model/data freeze; all choices preceded evaluation.
- `metadata.zip`: exact original run configurations, completion/calibration decisions,
  development predictions, measurements, histories and classifier searches.
- `source.zip`: producer, evaluator, exporter, renderer and replay source closure,
  including the actual earlier/later labelling-worker snapshots.
- `training-composition.json`: source/class/character totals for the nested subsets.
- `export-audit.json`, `cpu-replay.json`, `publication-checks.json`: local artifact
  audits, independent arithmetic replay and paper/archive verification.
- `package-manifest.json`: checksums for every file in this directory.

The exporter locally checked artifact/source/runtime identities, finite
initial/selected/final weights, selected epochs and matched initial states.
CPU replay validates recorded evidence and arithmetic; it does not rerun models.
Exact fitting additionally requires the retained local text snapshots and
pinned pretrained weights. Selected models are retained; redundant completed-run
checkpoints were subsequently removed. The finite-weight audit records the files
available when that audit was performed. The public package excludes
source bodies and weights. [The input receipt](input-archive.json) binds the
private six-file archive, whose members were compared byte-for-byte with frozen
inputs. Source rights and attribution are recorded in [sources.md](../../sources.md).

The 350 older training rows missing original document identifiers resolve by exact
body hash through the frozen SEC registry. Their URLs identify parent submissions;
the captured text uses edgartools extraction. The exact registry is archived and
checked against its preparation hash. Live recollection can produce different text.

[The audit correction](audit-membership-correction.json) records four training-only
substitutions made before labelling after ECB replacements, preserving balanced
100/100 audit cells. No completed reference was overridden. Original study data, results and ownership markers remain local; superseded
three-class publication bundles are available in Git history.
