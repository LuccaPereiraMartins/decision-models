# Academic manuscript

Read [the compiled paper](paper.pdf) or edit [the LaTeX source](paper.tex).
The manuscript uses a standard single-column article layout, numbered equations,
tables and figures, an abstract, related work, methods, results, discussion and
linked references. It keeps post-hoc diagnostics marked and puts the two later
fits in one development-only section. There is no cumulative appendix.

Authors are Lucca Pereira Martins and Matthew Ball, in that order. The
corresponding address is lp573@cantab.ac.uk. Affiliations have not been supplied;
Matthew's email is omitted because no matching author email was found in the
available Git history. These omissions do not prevent compilation.

## Compile

With a standard TeX Live installation, work in this directory:

    pdflatex -interaction=nonstopmode -halt-on-error paper.tex
    pdflatex -interaction=nonstopmode -halt-on-error paper.tex

The manuscript has an inline bibliography, so no BibTeX/Biber step or custom
style file is needed. It also compiles with Tectonic:

    tectonic -X compile --keep-logs paper.tex

The first Tectonic build may download public TeX packages/fonts; subsequent
builds can use its cache. The exact local compiler/build information, input
and output hashes, and visual checks are recorded in [the manifest](paper-manifest.json)
and [visual receipt](visual-verification.json). PDF metadata can vary across compiler
versions; source and scientific figure bytes are separately hashed.

## arXiv source archive

[The source archive](arxiv-source.zip) contains only the main TeX file and its six
figures with relative paths. Extract the archive and compile from its root.
The included figures use PDFLaTeX-compatible PNG format and standard graphicx;
there is no shell escape, automated figure conversion, remote inclusion or
embedded JavaScript. References are compiled into the source and arXiv
identifiers are included where applicable. These choices follow
[arXiv's TeX submission guidance](https://info.arxiv.org/help/submit_tex.html).

This is a manuscript and source package prepared for author review, not an arXiv
submission or a claim of acceptance. Author affiliations/contact details,
submission metadata/category and license should be settled during submission.
Any future scientific revisions must retain the distinction between the closed
core holdout and later development evidence.

The original [evidence package](../reports/core-v1/README.md) remains byte-identical
to commit 431d1e4; neither models nor test results were rerun for this manuscript.
All tables refer to that evidence and the copied figures retain its exact hashes.
Full training-replay and corpus-access limits remain in [reproduce.md](../reproduce.md).
