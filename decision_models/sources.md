# Source provenance and reuse

This records the acquisition audit checked on 9 October 2026 and its completed
collection census. It is not a current legal assessment. Per-document provenance
retains the URL, publication date, retrieval time, attribution and response/body
hashes. Source bodies and raw responses remain local.

| Source | Acquisition | Recorded reuse conditions | Original accepted candidates |
| --- | --- | --- | ---: |
| SEC 8-K | EDGAR index; edgartools 5.59.1 | [Public filing reuse](https://www.sec.gov/files/about/webmaster-faq.htm); exclude images | 30,000 |
| SEC corporate releases | EX-99 exhibits discovered by release/wire queries | Same SEC policy; preserve parent accession/event | 30,000 |
| SEC 6-K | EDGAR primary documents; same SDK | Same SEC policy | 25,000 |
| Federal Reserve | Annual press-release/speech HTML archives | [Public domain unless indicated](https://www.federalreserve.gov/disclaimer.htm); exclude third-party media | 1,826 |
| ECB | Website JSON archives and English institutional HTML | [Attribution, accuracy and modifications](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html); exclude author-named documents | 722 |
| Wikinews | MediaWiki business/financial categories | [CC BY 4.0 since 16 December 2024; CC BY 2.5 earlier](https://en.wikinews.org/wiki/Wikinews:Copyright); retain attribution | 201 |
| HM Treasury | GOV.UK Search discovery and Content API | [Open Government Licence v3.0, except stated exclusions](https://www.gov.uk/help/reuse-govuk-content); omit assets/special-rights material | 1,257 |

These 89,006 candidates, plus 10,000 legacy SEC records, are acquisition counts,
not labelled examples or independent evaluation support. The subsequent ECB binary
extension also searches institutional monetary-policy and statistical release
archives. Final selected support is in the [paper's cohort table](paper/paper.pdf)
and frozen [evidence](reports/binary-v2/README.md). Discovery is a reproducible query
sample, not an exhaustive disclosure census or commercial financial-news feed.
No eligible VOA-original articles were retained; GDELT supplied discovery URLs only,
which do not confer rights to publisher content.

## Extraction and dates

SEC dates are filing dates, including for release exhibits; they need not be the
earliest publication elsewhere. Fed/ECB/article dates use publication metadata,
not discovery times. GOV.UK articles revised after their publication day were
excluded because the original body was unavailable. Historical pages can change;
retrieval snapshots cannot prove verbatim availability on their stated date.

Collectors retain original responses and extract text without assets. All 89,006
new bodies/dates passed an offline re-extraction audit. SEC filing indexes identify
primaries; release queries are subdivided when search results are capped. Failed
responses and interrupted append records remain local acquisition provenance.
A 6-K can be a thin cover sheet: external exhibits/context are not silently added.

## Identity and evaluation limits

Related filings and releases share accession-based events. The older edgartools
text format is linked to new HTML extraction by exact body reconstruction from
retained responses. The frozen legacy registry resolves all 10,000 old identities,
including 350 retained training rows lacking original document identifiers.
Near duplicates use five-word-shingle Jaccard checks after probabilistic LSH
retrieval; this does not prove every possible duplicate was found. Publication-date
boundaries and whole event groups determine cohorts, rather than company identity.

The binary preparation preserves previous annotations, records each retained or
redecided reference, and adds label-blind whole groups. SEC dominates evaluation;
Wikinews supplies no Q3 2026 test rows. Teacher agreement and the small non-expert
rubric pilot do not establish analyst validity or useful trading signals.
Exact fitting uses the retained frozen snapshots and [input receipt](reports/binary-v2/input-archive.json),
not live recollection. Earlier acquisition logs and audit details remain in Git
history and the frozen preparation/source archives.
