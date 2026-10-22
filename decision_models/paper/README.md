# Binary financial-document triage paper

[Read the paper](paper.pdf): six main pages, one reference page and one supporting
page. The [original evidence](../reports/binary-v2/README.md) supplies the frozen
comparisons; the [25k LoRA package](../reports/lora25k/README.md) supplies the
planned larger training-size runs and paired intervals on the same test cohort.
Both provide independent CPU replay. The original
[FinBERT 25k development evidence](../reports/binary-finbert25k/README.md) is preserved.

To compile independently, extract [arxiv-source.zip](arxiv-source.zip) into an empty
directory and run `tectonic --keep-logs paper.tex` (Tectonic 0.17.0 tested).
The archive contains LaTeX and two figures; it needs no private data or models.

From the repository root, regenerate numerical tables/figures with
`python decision_models/report.py`, then run
`python decision_models/build_paper.py --engine tectonic`.
Editorial text lives in `paper.tex` and the prose fragments in `generated/`;
the renderer replaces numerical tables and figures while preserving that prose.

`paper-manifest.json` binds source, frozen scientific inputs and compiled artifacts.
`review.json` records rendered-page and reproduction checks. Compilation alone does
not establish visual quality. Earlier editions remain in Git history.
