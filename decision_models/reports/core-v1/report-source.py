"""Build a source-backed Markdown/PDF research report from verified frozen test artifacts."""

from collections import Counter, defaultdict
from importlib.metadata import version
import json
import re
import subprocess
from xml.sax.saxutils import escape

import numpy as np

from evaluate_study import bootstrap, validate_freeze
from prepare import family
from protocol import breakdown, fingerprint
from schemas import HERE, LABELS, annotation, body_hash, load_split, read_records, save


def number(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def ci(value):
    return "-" if value["lower"] is None else f"[{value['lower']:.3f}, {value['upper']:.3f}]"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def label(config):
    if config.get("kind") == "majority":
        return "Majority class"
    kind = {"laya": "Laya", "tfidf": "Word TF-IDF", "char": "Char TF-IDF", "combined": "Combined TF-IDF",
            "majority": "Majority", "finbert": "FinBERT", "bge": "BGE-small", "modernbert": "ModernBERT-large",
            "qwen17": "Qwen3-1.7B", "qwen4": "Qwen3-4B"}[config.get("kind", "laya")]
    suffix = config.get("classifier", config.get("adaptation", "SDK"))
    suffix = "LoRA" if suffix == "lora" else suffix
    if config["family"] == "research_zero":
        suffix += " epoch zero"
    return kind + " " + suffix


def human_audit(path, records, predictions, reference):
    if path is None:
        return {"status": "Pending blinded non-expert review", "resolved": 0, "unresolved": 0}
    reviewed = read_records(path)
    if {body_hash(r) for r in reviewed} != {body_hash(r) for r in records} or len(reviewed) != len(records):
        raise ValueError("Human audit must overlay exactly the frozen test bodies")
    labels = {body_hash(r): r["human"]["label"] for r in reviewed if r["human"]["label"] is not None}
    unresolved = sum(bool(r["human"].get("unresolved")) for r in reviewed)
    subset = {name: [{**r, "reference": labels[r["body_sha256"]], "reference_source": "human"}
                     for r in rows if r["body_sha256"] in labels] for name, rows in predictions.items()}
    truth = {body_hash(r): r["llm"]["label"] for r in records}
    return {"status": "Blinded non-expert audit; stratified review sample, not analyst ground truth",
            "source_sha256": fingerprint(path), "resolved": len(labels), "unresolved": unresolved,
            "teacher_agreement": sum(truth[k] == value for k, value in labels.items()) / len(labels) if labels else None,
            "models": {name: breakdown(rows) for name, rows in subset.items()},
            "bootstrap": bootstrap(subset, reference) if labels else None}




def training_composition(splits):
    manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    digest = fingerprint(splits / "subsets.json")
    if digest != manifest["subsets_sha256"]:
        raise ValueError("Frozen training subsets changed")
    rows = load_split(splits, "train")
    by_hash = {body_hash(row): row for row in rows}
    subsets = json.loads((splits / "subsets.json").read_text(encoding="utf-8"))["subsets"]
    counts, previous = {}, []
    for size in sorted(subsets, key=int):
        identities = subsets[size]
        if len(identities) != int(size) or len(set(identities)) != int(size) or identities[:len(previous)] != previous:
            raise ValueError("Training composition requires distinct, nested frozen subsets")
        selected = [by_hash[key] for key in identities]
        counts[size] = {"documents": len(selected), "groups": len({row.get("group_id", body_hash(row)) for row in selected}),
                        "labels": dict(Counter(annotation(row)["label"] for row in selected)),
                        "families": dict(Counter(family(row) for row in selected)),
                        "providers": dict(Counter(row.get("source", "sec") for row in selected))}
        previous = identities
    return {"train_sha256": manifest["splits"]["train"]["sha256"], "subsets_sha256": digest, "subsets": counts}




def uncertainty_bands(rows):
    """Fixed descriptive bands; these do not select a deployment threshold."""
    uncertainty = np.array([row["uncertainty"] for row in rows], dtype=float)
    if not np.isfinite(uncertainty).all() or ((uncertainty < 0) | (uncertainty > 1)).any():
        raise ValueError("Expected finite uncertainty in [0, 1]")
    bands = []
    bounds = (0.0, 0.1, 0.2, 0.4, 1.0)
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        mask = (uncertainty >= lower) & ((uncertainty <= upper) if upper == 1 else (uncertainty < upper))
        selected = [row for row, keep in zip(rows, mask, strict=True) if keep]
        support = sum(row["reference"] == "review_worthy" for row in selected)
        misses = sum(row["reference"] == "review_worthy" and row["prediction"] == "routine" for row in selected)
        bands.append({"lower": lower, "upper": upper, "count": len(selected),
                      "accuracy": sum(row["reference"] == row["prediction"] for row in selected) / len(selected) if selected else None,
                      "review_worthy_support": support, "dangerous_misses": misses,
                      "dangerous_miss_rate": misses / support if support else None})
    return bands




def reference_quote(records):
    """Local cost scenarios only: no API client, credentials, dispatch or paid benchmark."""
    import tiktoken
    from label import RUBRIC
    tokenizer = tiktoken.get_encoding("o200k_base")
    estimated = [len(tokenizer.encode(RUBRIC + r["body"], disallowed_special=())) + 128 for r in records]
    bound = [len((RUBRIC + r["body"]).encode("utf-8")) + 2048 for r in records]
    scenarios = {}
    for output in (128, 1024):
        scenarios[str(output)] = {"approximate_usd": sum(n / 1e6 * 1.25 * (2 if n > 272000 else 1) +
                                                         output / 1e6 * 5 * (1.5 if n > 272000 else 1) for n in estimated),
                                 "conservative_usd": sum(n / 1e6 * 1.25 * (2 if n > 272000 else 1) +
                                                         output / 1e6 * 5 * (1.5 if n > 272000 else 1) for n in bound)}
    return {"model": "gpt-6.1-sol", "tier": "flex", "executed": False, "documents": len(records),
            "rates_usd_per_million": {"input": 1.0, "cached_input": .05, "cache_write": 1.25, "output": 5.0},
            "pricing_checked": "2026-10-10", "pricing_source": "https://developers.openai.com/api/docs/pricing/",
            "tokenizer": "o200k_base approximation; UTF-8 byte upper bound plus 2048 prompt/schema overhead",
            "scenarios_output_tokens_per_document": scenarios,
            "limits": "Assumes all input written to cache, no reuse discounts. Above 272k input, 2x input/1.5x output. Reasoning counts as output; not a service guarantee."}




def figures(output, entries, configs, results, intervals, main, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    grouped = defaultdict(list)
    for name, entry in entries.items():
        if entry["role"] == "curve":
            grouped[label(configs[name])].append(name)
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (title, names) in enumerate(grouped.items()):
        names.sort(key=lambda n: entries[n]["samples"])
        x = [entries[n]["samples"] for n in names]
        y = [results[n]["macro_f1"] for n in names]
        ax.plot(x, y, "o-", label=title, alpha=.8, color=plt.get_cmap("tab20")(index % 20))
    ax.set(xscale="log", xlabel="Training documents (validation-selected recipes)", ylabel="Test macro F1",
           ylim=(-.02, 1.05), title="Sample efficiency on the same temporal test cohort")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning-curves.png", dpi=180)
    plt.close(fig)
    names = [n for n, e in entries.items() if e["kind"] == "laya" and e["role"] in ("context", "curve") and
             e["samples"] == min(max(e["samples"] for e in entries.values()), 4000)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for adaptation in ("head", "lora"):
        selected = sorted([n for n in names if configs[n]["adaptation"] == adaptation], key=lambda n: configs[n]["max_len"])
        x, y = [configs[n]["max_len"] for n in selected], [results[n]["macro_f1"] for n in selected]
        if selected:
            line, = ax.plot(x, y, "o-", label="LoRA" if adaptation == "lora" else adaptation)
            ax.vlines(x, [intervals[n]["macro_f1"]["lower"] for n in selected],
                      [intervals[n]["macro_f1"]["upper"] for n in selected], color=line.get_color())
    ax.set(xlabel="Laya total context tokens", ylabel="Test macro F1", ylim=(-.02, 1.05),
           title="Context ablation; group-bootstrap 95% intervals")
    if names:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "context.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, name in enumerate(main):
        perf = json.loads((output.parent / entries[name]["directory"] / "performance.json").read_text())
        x = perf["batch_throughput"]["1"]["documents_per_second"]
        ax.scatter(x, results[name]["macro_f1"], color=plt.get_cmap("tab20")(index % 20),
                   label=f"{label(configs[name])} (n={entries[name]['samples']})")
    ax.set(xscale="log", xlabel="Warm single-document throughput (documents/s)", ylabel="Test macro F1", ylim=(-.02, 1.05),
           title="Agreement and local throughput; shared RTX 3060/CPU environment")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "throughput.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, primary, strict=True):
        matrix = np.array(results[name]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=["routine", "review", "unclear"],
               yticklabels=["routine", "review", "unclear"], xlabel="Prediction", ylabel="Teacher reference", title=label(configs[name]))
    fig.tight_layout()
    fig.savefig(output / "confusion.png", dpi=180)
    plt.close(fig)


def pdf(markdown, destination):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize, styles["BodyText"].leading = 10, 14
    styles["Heading2"].keepWithNext = True
    cell = ParagraphStyle("Cell", fontSize=7.5, leading=10, alignment=TA_LEFT)
    header = ParagraphStyle("Header", parent=cell, textColor=colors.white)
    width = A4[0] - 84
    flow = []
    def text(value):
        value = escape(value).replace("`", "")
        value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#1855a0">\1</link>', value)
        return value
    lines, index = markdown.splitlines(), 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("| "):
            rows = []
            while index < len(lines) and lines[index].startswith("| "):
                columns = [value.strip() for value in lines[index].strip("| ").split("|")]
                if not all(value == "---" for value in columns):
                    rows.append([Paragraph(text(value), header if not rows else cell) for value in columns])
                index += 1
            widths = [width / len(rows[0])] * len(rows[0])
            if len(widths) > 4:
                widths = [width * .28] + [width * .72 / (len(widths) - 1)] * (len(widths) - 1)
            content = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            content.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203b58")),
                                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")])]))
            flow += [KeepTogether([content, Spacer(1, 12)])] if len(rows) <= 6 else [content, Spacer(1, 12)]
            continue
        match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if match:
            image = Image(str(destination.parent / match[2]))
            image.drawHeight = image.imageHeight * width / image.imageWidth
            image.drawWidth = width
            flow += [image, Paragraph(text(match[1]), styles["Caption"] if "Caption" in styles else cell), Spacer(1, 12)]
        elif line:
            level = len(line) - len(line.lstrip("#"))
            style = styles["Title" if level == 1 else "Heading2" if level else "BodyText"]
            flow.append(Paragraph(text(line.lstrip("# ") if level else line), style))
        index += 1
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 25, "Financial-document triage | fixed-seed research study")
        canvas.drawRightString(A4[0] - 42, 25, str(document.page))
    SimpleDocTemplate(str(destination), title="Financial-document triage across public sources",
                      author="Financial-document triage research", pagesize=A4,
                      rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=42).build(
        flow, onFirstPage=footer, onLaterPages=footer)









def narrative(freeze, configs, scored, predictions, intervals, statistics, names, performance,
              histories, raw_scores, composition, diagnostic, development, compact, audit):
    """One argument-led report; detailed evidence stays in the published replay package."""
    primary, reference = freeze['primary_decision'], freeze['primary_linear']
    a, b = scored[primary], scored[reference]
    pair = intervals[primary]['paired_delta_vs_reference']
    primary_n = freeze.get('primary_samples', freeze['entries'][primary]['samples'])
    manifest = freeze['manifest']['splits']
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    base = f'https://github.com/LuccaPereiraMartins/systematic-trading/blob/{revision}/decision_models'
    latencies = {n: performance[n]['latency_seconds']['mean'] * 1000 for n in names}
    parts = ['# Financial-document triage across public sources',
             '## Main findings',
             f"At {primary_n:,} training documents, {label(configs[primary])} scores {a['macro_f1']:.4f} test macro F1 "
             f"against {b['macro_f1']:.4f} for {label(configs[reference])}. The paired difference interval "
             f"{ci(pair['macro_f1'])} does not resolve an F1 advantage. Warm mean latency is "
             f"{latencies[primary]:.1f} versus {latencies[reference]:.2f} ms: {latencies[primary] / latencies[reference]:.1f} times higher for Laya.",
             'Adaptation improves agreement over an untrained decision head, and more training data can help. '
             'Neither finding removes the need for a strong lexical comparator. Longer contiguous context does '
             'not improve Laya head performance monotonically, and calibration miss targets do not transfer as test guarantees.',
             'Later development fits address two remaining engineering questions: whether more labels help FinBERT, '
             'and whether a smaller general-purpose encoder offers a cheaper alternative. Their selection/calibration '
             'results are promising in different ways, but were obtained after the original test was inspected and cannot revise its conclusion.',
             'All findings measure agreement with Luna rubric labels. Independent human review remains deferred; '
             'analyst-validated materiality, commercial-inbox usefulness and trading profitability are not established.',
             '## Task, data and comparison protocol',
             'The task is source-content-only inbox triage: routine, review-worthy or unclear. Review-worthy and '
             'unclear predictions go to review. The critical miss is a teacher review-worthy document predicted routine. '
             'Routine denotes ordinary administrative information; review-worthy denotes a potentially significant '
             'corporate, financial, economic or policy development; unclear denotes insufficient or conflicting content. '
             'No company history, portfolio context or subsequent market reaction is supplied. Macro F1 averages the '
             'three declared classes; it rewards classification agreement rather than directly measuring analyst workload.',
             f"The benchmark contains {manifest['train']['count']:,} training documents, {manifest['selection']['count']:,} "
             f"selection documents, {manifest['calibration']['count']:,} calibration documents and {manifest['test']['count']:,} "
             'test documents. Training ends in March 2026; selection/calibration cover April-June and test covers '
             'July-September. Event and verified near-duplicate groups cannot cross partitions. Previously experimented '
             'documents and linked events are excluded from fresh evaluation; duplicate retrieval remains approximate.',
             'The core compares 64 frozen configurations on the same 1,800 test documents and 1,770 related-document '
             'groups. Main trained-model comparisons use the identical 4k subset. Model/epoch selection uses selection '
             'data; temperature and discard rules use calibration data. Every core choice froze before test inference. '
             'Paired 95% intervals use 2,000 shared whole-group bootstrap draws, conditional on one training seed; '
             'supplementary comparisons have no multiplicity adjustment.',
             table(['Source family', 'Train', 'Test', 'Test routine', 'Test review-worthy', 'Test unclear'],
                   [[f, manifest['train']['families'].get(f, 0), manifest['test']['families'].get(f, 0),
                     *[sum(r['family'] == f and r['reference'] == k for r in predictions[primary]) for k in LABELS]]
                    for f in sorted(manifest['test']['families'])]),
             'EDGAR supplies 8-Ks, 6-Ks and release exhibits; other cohorts are official Fed, ECB and GOV.UK '
             'announcements. The test news cohort is entirely GOV.UK; Wikinews has no held-out coverage. SEC accounts '
             'for 1,683 of 1,800 test documents, so source diversity remains limited despite the larger corpus. '
             'Recorded dates are filing/publication dates at day precision; collected snapshots may differ from '
             'the text originally released. Pretraining exposure is also possible.',
             f"Methods and source rights are documented in [literature]({base}/literature.md) and [sources]({base}/sources.md). "
             'The frozen benchmark contains about $4.88 of retained teacher-label charges; pilots, failed requests and '
             'excluded labels also count against the separate project ledger. Hardware and human time are additional.',
             '## Held-out model comparison',
             table(['Method', 'Train', 'Test macro F1', '95% CI', 'Critical misses', 'Warm mean ms'],
                   [[label(configs[n]), freeze['entries'][n]['samples'], number(scored[n]['macro_f1'], 4),
                     ci(intervals[n]['macro_f1']), scored[n]['default_discard']['dangerous_misses'], number(latencies[n], 2)] for n in names]),
             'Critical-miss counts share a denominator of 1,292 teacher review-worthy documents. Unadapted references '
             'have no task training. The majority classifier predicts review-worthy throughout: its zero misses remove '
             'no review workload. The frozen FinBERT classifier outperforms the declared 4k FinBERT LoRA recipe on '
             'this test; that recipe comparison is not an exhaustive assessment of finance encoders or fine-tuning.',
             '![Primary confusion matrices; rows are teacher references](confusion.png)']
    if diagnostic:
        routing = diagnostic['routing']
        parts += ['### Class agreement and review routing',
                  'A small three-class F1 difference need not translate into better review routing. The following '
                  'decomposition was added after seeing the test results and is exploratory.',
                  table(['Class', 'Laya class F1', 'TF-IDF class F1', 'Contribution to macro-F1 difference', '95% CI'],
                        [[r['label'], number(r['primary_f1']), number(r['reference_f1']),
                          number(r['macro_f1_contribution'], 4), ci(r['interval'])] for r in diagnostic['class_decomposition']]),
                  'About 68% of the point difference comes from unclear. Every class-contribution interval includes '
                  'zero. Collapsing the existing predictions into discard versus review gives routing F1 '
                  f"{routing[primary]['macro_f1']:.3f} versus {routing[reference]['macro_f1']:.3f}, with paired difference "
                  f"interval {ci(diagnostic['paired_routing_intervals']['macro_f1'])}. Laya discards 191 versus 212 "
                  'required-review references among 1,499, while sending 28 more documents for review. These routing '
                  'and miss-rate differences remain unresolved; the two-class view does not replace the primary endpoint.']
    parts += ['## Confidence and automatic discarding',
              'Temperature scaling changes confidence without changing argmax labels. Lower calibration error '
              'does not itself establish that automatic discarding meets a miss-rate target.',
              table(['Primary method', 'Raw Brier', 'Calibrated Brier', 'Raw ECE', 'Calibrated ECE'],
                    [[label(configs[n]), number(raw_scores[n]['brier']), number(scored[n]['brier']),
                      number(raw_scores[n]['ece_10_bins']), number(scored[n]['ece_10_bins'])] for n in (primary, reference)]),
              'Brier sums squared errors over three classes (0-2); ECE uses ten confidence bins. Reported confidence '
              'is model class confidence, not the probability of investment materiality.',
              table(['Method', 'Calibration target', 'Test workload removed', 'Critical misses / 1292', 'Test miss rate'],
                    [[label(configs[n]), f'{100 * float(key):.0f}%', f"{100 * value['workload_reduction']:.2f}%",
                      value['dangerous_misses'], f"{100 * value['dangerous_miss_rate']:.2f}%"]
                     for n in (primary, reference) for key, value in scored[n]['policies'].items()]),
              'Thresholds maximize calibration workload removal subject to empirical 1%/5% review-worthy miss '
              'targets; predicted review-worthy/unclear always escalate. Laya exceeds both targets on test. '
              'This is the operational limit of the observed confidence gains: an analyst cannot interpret the '
              'target name as an achieved deployment bound. Zero observed errors on a small calibration set would '
              'also leave unseen errors unmeasured.']
    if diagnostic:
        parts += ['![Frozen policy tradeoffs and marginal group-bootstrap intervals](diagnostic-policies.png)',
                  'The ten plotted points reuse the original frozen policies. Circles/squares denote 1%/5% targets; '
                  'bars are separate marginal 95% intervals, not joint regions. At 1%, Laya removes less workload '
                  'than TF-IDF while the miss-rate difference is unresolved. At 5%, the nominal miss-rate-difference '
                  'interval is unfavorable to Laya, while the workload difference remains unresolved. '
                  'These conditional comparisons support interpretation, not a new rule selected on test.']
    contrasts = statistics['contrasts']
    parts += ['## Adaptation, training data and context',
              'Laya adaptation uses uniform whole-document pooling and overlapping windows; the SDK uses its '
              'native most-confident-window aggregation. Matched pre-update controls therefore matter. Encoder '
              'alternatives use fresh triage heads; FinBERT sentiment labels are not mapped onto the task. '
              'Different tokenization, pooling and head budgets make these system comparisons.',
              'The three new core LoRA fits use the same 4k subset, unweighted cross-entropy, rank 8, alpha 16, '
              'dropout 0.05, head/adapter learning rates 0.0001 and seed 42, allowing at most three epochs. '
              'Raw selection F1 chooses the epoch. Laya and ModernBERT use context 1,024; FinBERT uses 512. '
              'Pretrained revisions, earlier head recipes and exact configurations are retained in the package.',
              table(['Method', 'Matched-zero test F1', 'Adapted test F1', 'Paired gain 95% CI'],
                    [[label(configs[n]), number(scored[freeze['entries'][n]['matched_zero']]['macro_f1']),
                      number(scored[n]['macro_f1']), ci(contrasts[n + '-adaptation']['metrics']['macro_f1'])]
                     for n in names if freeze['entries'][n].get('matched_zero')]),
              'Gains over a fresh untrained head establish supervised task learning, rather than proving adapter '
              'superiority over a trained frozen-feature classifier. Rank 8 updates about 27.166M parameters '
              'including Laya\'s native head, versus 0.923M for ModernBERT and 0.299M for FinBERT. Equal rank '
              'does not equalize the adaptation budget.',
              '![Frozen sample-efficiency curves on the same test cohort](learning-curves.png)']
    selected = [v for k, v in contrasts.items() if k.endswith('-sample-increment') and
                ((configs[v['left']].get('kind') == 'laya' and configs[v['left']].get('adaptation') == 'lora') or
                 (configs[v['left']].get('kind') == 'tfidf' and configs[v['left']].get('classifier') == 'logistic' and
                  freeze['entries'][v['left']]['samples'] >= 16000))]
    parts += [table(['Method', 'Training increment', 'Test F1 gain', 'Paired 95% CI'],
                    [[label(configs[v['left']]), f"{freeze['entries'][v['right']]['samples']:,} to {freeze['entries'][v['left']]['samples']:,}",
                      number(scored[v['left']]['macro_f1'] - scored[v['right']]['macro_f1']), ci(v['metrics']['macro_f1'])] for v in selected]),
              'Laya LoRA improves from 1k to 4k within its fixed recipe. Lexical methods also benefit from more '
              'data; the word-logistic gain from 16k to 25k is unresolved. Subsets are nested, but class/source '
              'allocation changes as strata exhaust. Recipe screening and 1,200 selection/calibration labels '
              'are additional to the stated training sizes, so these curves do not identify a universal minimum '
              'label requirement. Exact allocations and all increments remain in the evidence package.',
              '![Contiguous context ablation under fixed whole-document pooling](context.png)',
              'Laya head context increases from 512 to 1,024 tokens yield a paired test-F1 difference interval '
              '[-0.014, 0.031]; 512 to 4,096 yields [-0.052, -0.006]. The pattern is not monotonic. Wider windows '
              'change contiguous context and the number of pooled votes; all normalized source tokens remain '
              'represented at every size. The supplementary, uncorrected head experiment neither isolates '
              'context from pooling effects nor establishes a LoRA context result.',
              '## Throughput and resource cost',
              '![Held-out agreement versus local warm throughput](throughput.png)',
              'Timings use the same 128-document selection sample on an RTX 3060/CPU machine, including '
              'preprocessing and all source windows but excluding loading. Sklearn uses native document batching; '
              'neural document batches are serial with window batching. Hardware, electricity and acquisition '
              'costs are outside the recorded API charges.',
              table(['Neural method', 'Train/validation h', 'Training peak GiB', 'Inference peak GiB', 'Warm P95 ms'],
                    [[label(configs[n]), number(histories[n]['seconds'] / 3600, 2),
                      number(performance[n]['training_peak_vram_bytes'] / 2**30, 2),
                      number(performance[n]['peak_vram_bytes'] / 2**30, 2),
                      number(performance[n]['latency_seconds']['p95'] * 1000, 1)] for n in names if n in histories]),
              'Training clocks include validation, excluding startup and final calibration/resource work. '
              'Frozen-feature preparation and classifier fitting have separate accounting; cache reuse means '
              'a classifier fit clock alone understates encoder preparation. Detailed batch throughput, '
              'feature clocks, checkpoint sizes and parameter counts are exported with the measurements.']
    if diagnostic:
        overlap = diagnostic['critical_miss_overlap']
        parts += ['## Where errors occur',
                  'The following source/length cuts and overlap analysis are exploratory, added after the primary '
                  'results. They explain the current comparison without changing its selection.',
                  '![Primary F1 differences by source and document length](diagnostic-slices.png)',
                  'Bars are paired whole-group 95% intervals within each slice, retaining three fixed labels. '
                  'Families without unclear references have an attainable macro-F1 ceiling of 2/3; compare '
                  'methods within a slice rather than raw F1 across families. The 6-K difference has a nominal '
                  'positive interval, with no multiplicity correction. All four length-band intervals include '
                  'zero. Character cutpoints 2,695/4,589/9,835 come from selection quartiles; they are not '
                  'token budgets. Source and length are confounded, so the figure is not a causal context test.',
                  f"Both primary methods miss {overlap['both']} teacher review-worthy documents; {overlap['primary_only']} "
                  f"are Laya-only misses and {overlap['reference_only']} are TF-IDF-only. Complementary errors suggest "
                  'a possible future cascade, but do not measure its false-positive, workload or compute tradeoffs. '
                  'Shared errors may also reflect questionable teacher references. Independent review is needed '
                  'to distinguish those cases; high confidence alone does not settle them.']
    if development or compact:
        parts += ['## Development evidence: more labels and a smaller encoder',
                  'These later fits answer two follow-up questions rather than extend the original test leaderboard. '
                  'They were chosen after the test results were inspected. The 837-document selection cohort chooses '
                  'epochs, and the 363-document calibration cohort fits confidence/policies; both are fitting data. '
                  'The table below must not be compared directly with the held-out F1 table.',
                  'All neural fits start afresh from pinned weights, with uniform pooling, context 512, rank 8, '
                  'alpha 16, adapter dropout 0.05, unweighted CE, learning rate 0.0001 and seed 42. Three epochs '
                  'are allowed, including epoch zero in selection. Both FinBERT sizes choose epoch three; BGE '
                  'chooses epoch two. Architecture and classifier recipes still differ.']
        rows = []
        if development:
            rows += [(f'FinBERT LoRA ({n})', r) for n, r in development['models'].items()]
        if compact:
            rows += [('BGE-small LoRA (4k)', compact['models']['bge_lora']), ('Frozen BGE/logistic (4k)', compact['models']['bge_frozen'])]
        parts += [table(['Method / train size', 'Selection F1', 'Critical misses / 552', 'Unclear discarded / 93', 'Default discards / 837', 'Warm mean ms'],
                        [[name, number(r['selection']['macro_f1'], 4), r['selection']['default_discard']['dangerous_misses'],
                          r['selection']['default_discard']['unclear_discarded'], r['selection']['default_discard']['discarded'],
                          number(r['warm_mean_ms'], 2)] for name, r in rows]),
                  '### Does more data help the fixed FinBERT recipe?',
                  'Moving from 4k to 16k raises selection F1 by 0.0921 and reduces critical misses from 132 to 67. '
                  'It sends 44 more documents for review and discards twelve additional unclear references. '
                  'Training/validation grows from 32.6 to 143.1 minutes (4.38 times), while inference latency '
                  'is approximately unchanged. More examples, 1,500 versus 6,000 optimizer steps and changing '
                  'class/source proportions vary together. This promising fitting signal warrants fresh evaluation; '
                  'two points do not establish a minimum data requirement.',
                  '### What does the compact encoder trade off?',
                  'At the matched 4k size, BGE LoRA has slightly lower selection F1 than FinBERT but lower warm '
                  'latency (20.23 versus 33.94 ms) and memory (0.32 versus 0.78 GiB training peak; 0.19 versus '
                  '0.52 GiB inference peak). Its 84 versus 132 critical misses come with 67 extra documents '
                  'for review and two extra unclear discards. Against frozen BGE, the small F1 gain and 25 '
                  'fewer critical misses likewise come with 23 extra reviews. It is a resource/workload tradeoff, '
                  'not evidence of operational dominance.',
                  'BGE training/validation takes 21.0 minutes and updates 149,379 parameters versus FinBERT\'s '
                  '298,755. Its matched epoch-zero F1 is 0.3120; the gain includes learning a fresh classifier '
                  'as well as encoder adapters. Neither finance pretraining nor adapter updates are isolated by '
                  'these different systems.',
                  'The 16k FinBERT 1% policy still misses 1.45% of selection review-worthy references. All three '
                  'matched-4k systems exceed both calibration targets on selection. This echoes the core study\'s '
                  'warning about target transfer. No additional test result or independent significance claim '
                  'is made, and the original held-out conclusions remain unchanged.']
    parts += ['## Interpretation, limits and reproduction',
              'For this teacher-labeled public-source task, lexical classification remains a strong throughput '
              'baseline. Decision-model adaptation learns the task and benefits from more data, but the primary '
              'study does not resolve an F1 advantage that offsets its much higher measured latency. Neither '
              'longer context nor calibrated confidence establishes a reliable automatic-discard rule.',
              'The study is limited by teacher references without independent human adjudication, one trained '
              'seed, approximate duplicate detection, institutional-source dominance and possible historical-text '
              'or pretraining exposure. Bootstrap intervals measure cohort sampling conditional on the fitted '
              'models, not training instability. Supplementary and post-hoc comparisons are uncorrected for '
              'multiplicity. The later encoder fits use development data and support choosing a fresh evaluation '
              'question, rather than choosing a deployment winner.',
              'A stronger follow-up needs independently reviewed labels and fresh, representative held-out '
              'documents, then validation-selected methods and policies. Generative adaptation and RL/RLCD '
              'remain unmeasured. No claim about trading returns or analyst-validated materiality follows from '
              'agreement alone.',
              f"The [evidence package]({base}/reports/core-v1/README.md) supplies all 64 frozen comparisons, class/source "
              'metrics, paired intervals, exact development probabilities/decisions, parameter/resource records '
              'and independent audit receipts. `python decision_models/reports/core-v1/verify.py` replays '
              'the published arithmetic on CPU without weights, bodies or paid calls. Full training/inference '
              f"replay additionally requires preserved corpus/checkpoints and pinned dependencies; [reproduce.md]({base}/reproduce.md) "
              'documents those access and source-rights limits.']
    return parts

def build(splits, study, human=None):
    plan = json.loads((study / "plan.json").read_text(encoding="utf-8"))
    freeze = validate_freeze(splits, study)
    results = json.loads((study / "results.json").read_text(encoding="utf-8"))
    for key, name in (("bootstrap", "bootstrap.json"), ("matched_coverage", "matched-coverage.json")):
        if fingerprint(study / name) != results[f"{key}_sha256"]:
            raise ValueError("Report statistics changed after test summarization")
    entries = freeze["entries"]
    predictions, configs, scored = {}, {}, {}
    for name, entry in entries.items():
        path = study / "test" / f"{name}.json"
        if fingerprint(path) != results["test_artifacts"][name]:
            raise ValueError("Test evidence changed before report generation")
        test = json.loads(path.read_text(encoding="utf-8"))
        if test["freeze_sha256"] != fingerprint(study / "freeze.json"):
            raise ValueError("Report test results use another freeze")
        predictions[name], scored[name] = test["predictions"], test["metrics"]["overall"]
        configs[name] = json.loads((study / entry["directory"] / "config.json").read_text(encoding="utf-8"))
    records = load_split(splits, "test")
    audit = human_audit(human, records, predictions, freeze["primary_linear"])
    if plan.get("preset") == "core" and human is None:
        audit["status"] = "Deferred for this version by agreement"
    quote = reference_quote(records)
    composition = training_composition(splits)
    bands = {name: uncertainty_bands(rows) for name, rows in predictions.items()}
    output = study / "report"
    output.mkdir(exist_ok=True)
    save(audit, output / "human-audit.json")
    save(quote, output / "reference-quote.json")
    save(composition, output / "training-composition.json")
    save({"definition": "1 minus maximum calibrated class probability",
          "method": "Fixed descriptive bands; argmax review-to-routine misses divided by review-worthy support within each band",
          "models": bands}, output / "uncertainty-bands.json")
    statistics = json.loads((study / "bootstrap.json").read_text())
    intervals = statistics["models"]
    main = {}
    for name, entry in entries.items():
        if entry["role"] in ("curve", "reference"):
            if plan.get("preset") == "core" and entry["role"] == "curve" and entry["samples"] != freeze["primary_samples"]:
                continue
            key = label(configs[name])
            if key not in main or entry["samples"] > entries[main[key]]["samples"]:
                main[key] = name
    primary, reference = freeze["primary_decision"], freeze["primary_linear"]
    names = list(dict.fromkeys((primary, reference, *main.values())))
    raw_scores = {name: json.loads((study / "test" / f"{name}.json").read_text())["raw_metrics"]["overall"] for name in names}
    figures(output, entries, configs, scored, intervals, names, (primary, reference))
    performance = {n: json.loads((study / entries[n]['directory'] / 'performance.json').read_text()) for n in names}
    histories = {n: json.loads((study / entries[n]['directory'] / 'history.json').read_text())
                 for n in names if (study / entries[n]['directory'] / 'history.json').exists()}
    diagnostic = development = compact = None
    diagnostic_outputs = ()
    inputs_path = output / "diagnostic-inputs.json"
    if inputs_path.exists():
        from diagnostics import analyze, figures as diagnostic_figures
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        if inputs['freeze_sha256'] != fingerprint(study / "freeze.json"):
            raise ValueError("Exploratory inputs belong to another frozen study")
        if (inputs['primary'], inputs['reference']) != (primary, reference):
            raise ValueError("Exploratory inputs cannot change the frozen primary comparison")
        selection_path = study / entries[reference]['directory'] / "selection.json"
        selection_rows = json.loads(selection_path.read_text(encoding="utf-8"))['predictions']
        if fingerprint(selection_path) != inputs['selection_sha256'] or [r['characters'] for r in selection_rows] != inputs['selection_characters']:
            raise ValueError("Exploratory length bands must use the unchanged selection cohort")
        diagnostic = analyze(predictions, inputs, statistics)
        save(diagnostic, output / "diagnostics.json")
        (output / "diagnostics-source.py").write_bytes((HERE / "diagnostics.py").read_bytes())
        labels = {n: label(configs[n]) for n in inputs['policy_models']}
        diagnostic_figures(output, diagnostic, labels)
        diagnostic_outputs = ('diagnostic-inputs.json', 'diagnostics.json', 'diagnostics-source.py',
                              'diagnostic-slices.png', 'diagnostic-policies.png')
    development_outputs = ()
    evidence_path = output / 'development-evidence.json.gz'
    if evidence_path.exists():
        import gzip
        from development import summarize
        evidence = json.loads(gzip.decompress(evidence_path.read_bytes()))
        if evidence['freeze_sha256'] != fingerprint(study / 'freeze.json'):
            raise ValueError('Development evidence belongs to another original study')
        development = summarize(evidence)
        save(development, output / 'development-summary.json')
        (output / 'development-source.py').write_bytes((HERE / 'development.py').read_bytes())
        development_outputs = ('development-evidence.json.gz', 'development-summary.json', 'development-source.py')
    compact_outputs = ()
    evidence_path = output / 'compact-evidence.json.gz'
    if evidence_path.exists():
        import gzip
        from compact import summarize
        evidence = json.loads(gzip.decompress(evidence_path.read_bytes()))
        if evidence['freeze_sha256'] != fingerprint(study / 'freeze.json'):
            raise ValueError('Compact development evidence belongs to another original study')
        compact = summarize(evidence)
        save(compact, output / 'compact-summary.json')
        (output / 'compact-source.py').write_bytes((HERE / 'compact.py').read_bytes())
        compact_outputs = ('compact-evidence.json.gz', 'compact-summary.json', 'compact-source.py')
    parts = narrative(freeze, configs, scored, predictions, intervals, statistics, names, performance,
                      histories, raw_scores, composition, diagnostic, development, compact, audit)
    content = "\n\n".join(parts) + "\n"
    (output / "report.md").write_text(content, encoding="utf-8")
    (output / "report-source.py").write_bytes((HERE / "report.py").read_bytes())
    pdf(content, output / "report.pdf")
    save({"freeze_sha256": fingerprint(study / "freeze.json"), "results_sha256": fingerprint(study / "results.json"),
          "report_source_sha256": fingerprint(HERE / "report.py"),
          "versions": {n: version(n) for n in ('matplotlib', 'reportlab', 'pypdf')},
          "outputs": {name: fingerprint(output / name) for name in
                      ('report.md', 'report.pdf', 'report-source.py', 'human-audit.json', 'reference-quote.json', 'training-composition.json', 'uncertainty-bands.json',
                       'learning-curves.png', 'context.png', 'throughput.png', 'confusion.png') + diagnostic_outputs + development_outputs + compact_outputs}}, output / "manifest.json")
    print(f"Report written to {output}; render and inspect PDF before publication", flush=True)
