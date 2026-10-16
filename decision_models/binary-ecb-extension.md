# ECB coverage for the binary study

The binary study adds official statistical releases to the existing English institutional-release
whitelist. This collection is separate from the original corpus and its response cache. It uses the
fixed publication-date interval **1 January 2019–30 September 2026** and collects available eligible
content rather than a source quota. It does not add author-named articles, speeches, research papers,
PDF bodies, chart images, downloadable statistical tables, or external data.

## Discovery and reuse

The [ECB disclaimer and copyright conditions](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html)
allow reuse of institutional information with accurate reproduction, attribution and disclosure of
modifications. Author-named documents have a separate permission requirement and remain excluded.
The new corpus retains the ECB source, conditions URL, extraction modifications and response/body
hashes. The article text is a retrieved snapshot, not proof of the exact historical release bytes;
linked live data and attachments are not used.

The publication archive's cached snapshot has **20,149 records**, version `1791461849/byBkup0p`.
Its 83 cached discovery responses were copied into the new cache through a read-only connection.
No original corpus, state, index or raw response was changed.

Discovery cross-checks the following twelve official [statistical-release](https://www.ecb.europa.eu/press/stats/html/index.en.html)
series, including the snippets specified by each page's own `data-snippets` attribute:

- Monetary developments (`md`), balance of payments (`bop`), institutional-sector developments
  (`ffi`) and bank interest rates (`mfi`).
- Payments (`paysec`), climate indicators (`cci`), pension funds, insurance corporations (`icb`),
  financial vehicle corporations (`fvc`), investment funds (`if`), euro money markets and securities
  issues (`sis`).

The unique eligible discovery pool contains **1,256 URLs**: the original 722 corpus URLs and **534
new statistical releases**. The series archives corroborate the 534 candidates; they do
not add further eligible URLs beyond the publication index. Candidate publication dates place 509
before April 2026, 12 in April–June and 13 in July–September. All 534 passed extraction and language
checks. ECB additions alone cannot fill the larger development and test cohorts; no newer date or
ineligible document was added to meet a count.

## Reproducible collection

Run from the repository root; the contact user agent is read from the existing local environment:

    python -u decision_models/collect_sources.py --family ecb --ecb-statistics --exclude-output decision_models/data/research/corpus/ecb.jsonl --number 5000 --start 2019-01-01 --end 2026-09-30 --output decision_models/data/research/binary-v2-corpus/ecb.jsonl

The 5,000 limit is a safety ceiling, not a target. Requests share one cache and courteous request
stream. Repeating the command resumes the saved index and accepted/rejected outcomes without
requesting cached pages again. `ecb.discovery.json` records archive candidates and discovery
exclusions; `ecb.state.json` records extraction outcomes and errors. The output directory has its
own raw-response database, command/process receipt and collection logs.

English HTML remains an explicit eligibility requirement. Extraction checks the official final
host, release title, named-author metadata/bylines and agreement between publication dates and
discovery. The visible release date takes precedence over inconsistent page metadata; both values
are retained. Language identification must return English with probability at least 0.9. Original
corpus IDs and exact body duplicates are excluded, as are duplicates within the additions.

## Verification

The completed collection has **534 accepted additions**, 722 excluded original URLs, no rejected
eligible release and no pending candidate. Publication dates range from 3 January 2019 to
25 September 2026. Body lengths range from 2,975 to 17,111 characters (median 6,573).

Every accepted body was re-extracted offline from the new cache, every raw/body hash checked and
every English-language decision recomputed. All 534 visible release dates agree with discovery.
Three 2019 releases have inconsistent page metadata referring to earlier periods; their visible
dates and both archive entries agree. Initial exclusion attempts were preserved and the three
releases were replayed from cache. All 634 cached response hashes match. There are no duplicate
IDs/bodies within the additions or against the original ECB corpus, whose three file hashes remain
unchanged. Named-author, redirect, date-mismatch and byline exclusion fixtures also pass.

The private audit receipt is `data/research/binary-v2-corpus/ecb-verification.json`; the accepted
corpus hash is `aeb211bd3f6ff084c8aa698cb1916c2f0c97a673be2e68ec83b002c4639cd538`.
Ruff, Python compilation and diff whitespace checks pass. No paid calls, annotations or models were
run by this collector.
