# Prior-work and novelty audit

Search date: 13 October 2026. Relevant prior work exists. The original manuscript's
related-work section was too generic; the revised paper explicitly treats this study
as an incremental financial-domain benchmark, rather than a new triage problem or
new calibration/adaptation method.

The closest model/task combination found is Eduardo Stuani's public financial-complaint
benchmark. The closest Laya reliability study is Nandakishore's September preprint.
Neither evaluates the particular investment-review rubric and document cohort used
here. That distinction supports a narrow domain comparison; it does not establish
that no other exact match exists.

## Closest primary sources

| Work | Relevant overlap | Difference from this study |
| --- | --- | --- |
| [Nandakishore, 2026](https://arxiv.org/html/2609.33843v1), arXiv:2609.33843 | Laya calibration, selective escalation, synthetic-teacher agreement and failed target transfer. | Specialized Typed-Decisions checkpoint and synthetic workflow questions; we adapt the root checkpoint to real financial texts. |
| [Stuani, 2026](https://github.com/stuaninauts/bench-laya-vs-llms) | Financial triage; Laya, ModernBERT, TF-IDF, encoder and LLM comparisons; timing, calibration and paired uncertainty. | Complaint routing/fraud-report flags with consumer-selected labels, rather than investment-review labels. Public engineering benchmark. |
| [Borchers, 2026](https://www.drborchers.com/en/blog/do-you-need-a-decision-model/) | Fine-tuned Laya/ModernBERT, an embedding/logistic comparator, learning curves and uncertainty. | Research-paper categories and constructed probes. Engineering report dated 4 October, updated 7 October. |
| [Ahbali et al., 2022](https://aclanthology.org/2022.naacl-industry.40.pdf) | Industrial financial-news filtering with TF-IDF/SVM before credit-event analysis. | Business/finance versus out-of-domain filtering and company credit analysis, not general review-worthiness. |
| [Daudert, FinLin, 2021/2022](https://link.springer.com/article/10.1007/s10579-021-09555-3) | Multi-source financial text with expert relevance and sentiment annotations. | Entity-specific influence on aggregate sentiment; different labels, annotation and sources. Online 2021, journal issue 2022. |
| [Del Corro and Hoffart, 2020](https://arxiv.org/abs/2001.09466) | Automatic financial-news relevance assessment. | Headline ranking derived from stock-prediction attention and market data. |
| [Feuerriegel and Pröllochs, 2018](https://arxiv.org/abs/1805.03308) | Regulatory 8-K categorization and financial relevance. | Topic modeling followed by market-reaction analysis, rather than teacher review triage. |
| [Park, workshop contribution 2024](https://www.bis.org/2026-07/ifcb65_09_rh.pdf), IFC Bulletin 65 | LLM assessment of material versus non-material disclosure risks. | Physical-risk examples supplied with company financial, operational and geographic context. Our source-only labels cannot replace that assessment. |
| [Johnsen and Shasharina, 2026](https://tommijohnsen.substack.com/p/we-trusted-finbert-to-filter-the) | Headline noise filtering, FinBERT/LLM escalation and missed signals in the neutral pile. | Target-company sentiment/impact, teacher comparisons and return analysis; practitioner report, not an independent validation of our rubric. |

For dynamic repository descriptions, recorded HEADs are
95cdee3e8539bf9e8eb09c575e680731cfe51b68 for Stuani's benchmark and
1299fc8235e3bd052c805482832f64eb02df85be for Roshan Santhosh's
[decision-model benchmark repository](https://github.com/rsk2327/decision-model-benchmarks).
The latter covers Banking77, ContractNLI, DynaSent and SNLI, not investment review.
External numerical results have not been independently rerun in this project.

## Other screened adjacent work

| Source | Reason it does not establish an exact task match |
| --- | --- |
| [Peng et al., 2021](https://aclanthology.org/2021.econlp-1.5/) | BERT/FinBERT domain-pretraining comparison across financial tasks; supports caution about adaptation claims. |
| [Banerjee et al., 2024](https://aclanthology.org/2024.finnlp-1.25/) | Financial-news ESG impact duration, rather than routine/review-worthy/unclear routing. |
| [FLaME, 2025](https://aclanthology.org/2025.gem-1.72/) | Financial language-model evaluation taxonomy/suite; broader evaluation context. |
| [Jin et al., 2026](https://arxiv.org/abs/2609.20110) | Calibrated financial-document field extraction/auto-approval, rather than content triage. |
| [Beyond Sentiment, 2026](https://arxiv.org/html/2607.28496v1) | Structured financial-news extraction and stock prediction, rather than this review convention. |

Public SEC-materiality triage products and financial-classification tutorials also
surfaced. Their existence reinforces that the application idea is established;
marketing descriptions were not treated as scientific performance evidence.

## Search scope and limits

The search used general web discovery followed by primary arXiv, ACL Anthology,
publisher, BIS, author-report and repository pages. Full relevant sections were
read where accessible: the Laya preprint's setup/comparators/escalation sections,
the complaint benchmark's task/protocol/limitations, FinLin's relevance definition
and annotation description, the credit-news paper's relevance filter, and Park's
materiality examples/prompt. The remaining adjacent records were screened using
their primary descriptions. This is a targeted review, not a systematic review
with a complete search-index census or formal exclusion protocol.

Queries included:

- financial document triage materiality classification SEC filings machine learning analyst review
- financial news materiality classification information triage machine learning routine disclosure
- Laya financial classification decision model benchmark adaptation
- financial document classification review worthy unclear materiality FinBERT
- financial news relevance classification materiality investor review dataset paper
- 8-K materiality classification machine learning paper
- financial document relevance classification analyst review worthy routine paper
- financial triage Laya research
- review-worthy financial document classification paper
- routine material financial news classification machine learning
- financial news importance classification benchmark (arXiv/ACL)
- materiality 8-K language model (arXiv/ACL/SSRN/BIS)
- financial document triage classification (arXiv/ACL)
- Laya financial review materiality

Additional title/author searches resolved source metadata. Publication dates were
checked separately from retrieval dates; the online/journal dates for FinLin and
the workshop provenance for Park are distinguished above. New or unindexed work,
paywalled details and other languages may be missed. No exact match to the full
cohort/rubric/protocol combination was identified among the reviewed sources.
That is a bounded search result, not a global absence or first-study claim.

## Defensible contribution

The defensible contribution is the particular public-source teacher-agreement
benchmark, temporal/event-group controls, equal-data comparison and whole-document
context/resource evidence. The study extends existing findings to that setting.
It cannot claim to invent financial relevance filtering, to be the first financial
Laya benchmark, to originate calibration/deferral analysis, or to establish that
lexical baselines are newly useful. Independent reference validity and a fresh
representative test cohort remain the requirements for stronger operational claims.

The LaTeX paper now cites the nine closest works, narrows its introduction and
related-work contribution statements, and retains all original measurements.
No test was reopened and no new model or threshold was selected.
