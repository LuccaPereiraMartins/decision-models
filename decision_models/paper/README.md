# Binary financial-document triage paper

[Read the paper](paper.pdf): six main pages, one reference page and one supporting
page. The [scientific evidence](../reports/binary-v2/README.md) supplies the frozen
comparisons and independent CPU replay. The later [FinBERT 25k fit](../reports/binary-finbert25k/README.md)
is separate development evidence.

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
