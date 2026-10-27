"""Compile the canonical paper and bind its source archive to the frozen evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(engine):
    import report

    paper = HERE / "paper"
    evidence = HERE / "reports/binary-v2"
    sources = sorted(p for p in paper.rglob("*") if p.suffix in (".tex", ".png"))
    additional = HERE / "reports/lora25k"
    inputs = json.loads((evidence / "manuscript-inputs.json").read_text(encoding="utf-8"))
    for name, expected in inputs["generated_files"].items():
        name = name.replace("\\", "/")
        if name == "figures/confusion.png" or name in {
            "references.tex",
            "generated/data-table.tex",
            "generated/supporting.tex",
        }:
            if sha(paper / name) != expected:
                raise ValueError(f"Scientific table, figure or reference changed: {name}")
    tables = re.findall(
        r"\\begin\{table\}.*?\\end\{table\}", (paper / "generated/main-results.tex").read_text(encoding="utf-8"), re.S
    )
    tables_sha256 = hashlib.sha256("\n".join(tables).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="triage-paper-") as temporary:
        clean = Path(temporary)
        expected = clean / "verified"
        shutil.copytree(paper, expected)
        report.build(evidence, expected, additional)
        for name in ("generated/data-table.tex", "generated/supporting.tex", "generated/additional-results.tex",
                     "figures/confusion.png", "figures/learning-curves.png"):
            if sha(paper / name) != sha(expected / name):
                raise ValueError(f"Evidence-derived output changed: {name}; regenerate with report.py")
        rebuilt_tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}",
                                    (expected / "generated/main-results.tex").read_text(encoding="utf-8"), re.S)
        if tables != rebuilt_tables:
            raise ValueError("Result tables disagree with verified original/additional evidence")
        archive = clean / "arxiv-source.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sources:
                member = zipfile.ZipInfo(path.relative_to(paper).as_posix(), (1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(member, path.read_bytes())
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(clean)
        subprocess.run([engine, "--keep-logs", "paper.tex"], cwd=clean, check=True)
        log = (clean / "paper.log").read_text(encoding="utf-8", errors="replace")
        if re.search(r"Overfull \\[hv]box|Undefined control sequence|Missing character|undefined references", log):
            raise ValueError("LaTeX diagnostics require review; output not published")
        shutil.copyfile(clean / "paper.pdf", paper / "paper.pdf")
        shutil.copyfile(archive, paper / archive.name)
        shutil.copyfile(clean / "paper.log", paper / "build-log.txt")
    manifest = {
        "edition": "Binary financial-document triage",
        "status": "compiled; visual review recorded separately in review.json",
        "freeze_sha256": sha(evidence / "freeze.json"),
        "evidence_sha256": {name: sha(evidence / name) for name in ("evidence.json.gz", "source.zip", "metadata.zip")},
        "additional_evidence_sha256": sha(additional / "evidence.json.gz"),
        "additional_verifier_sha256": sha(additional / "verify.py"),
        "renderer_sha256": sha(HERE / "report.py"),
        "tables_sha256": tables_sha256,
        "builder_sha256": sha(Path(__file__)),
        "engine": subprocess.check_output([engine, "--version"], text=True).strip(),
        "source_files": {p.relative_to(paper).as_posix(): sha(p) for p in sources},
        "artifacts": {name: sha(paper / name) for name in ("paper.pdf", "arxiv-source.zip", "build-log.txt")},
        "checks": "Protected original inputs; all numerical tables/figures regenerated from verified original and additional evidence; fresh source-archive compilation",
    }
    (paper / "paper-manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="tectonic", help="Tectonic executable (0.17.0 tested)")
    build(parser.parse_args().engine)
