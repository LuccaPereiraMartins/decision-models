# Binary triage evidence

The currently published files document label-blind data preparation. Binary
references and model results are still pending. Exact prepared counts are
25,000 training, 1,000 selection, 500 calibration and 2,000 test documents.

The new benchmark retains earlier evaluation bodies and related-document groups,
adds eligible source content, and freshly re-decides unclear labels rather than
mapping them mechanically. Its test is not fresh independent confirmation of
the previously inspected three-class study.

After the bounded study completes, `export_binary.py` will validate its frozen
identities and export body-free metadata, logits, decisions, fitting histories,
measurements and source/archive hashes. `binary_replay.py` independently checks
binary confusion/F1, false negatives, workload, calibrated decisions, maximal
feasible tied policies, temperature fitting, timing and paired group-bootstrap
arithmetic on CPU. Neither exporter nor replay fits models or calls an API.

Exact training replay additionally requires the local frozen document bodies,
checkpoints and pinned pretrained weights. The final public package will not
include source bodies or weights. Source rights and attribution remain in
[sources.md](../../sources.md).
