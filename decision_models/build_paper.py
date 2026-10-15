"""Compile and package an editorial revision without changing the frozen study."""

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
    original = HERE / "paper-binary-v2"
    paper = HERE / "paper-binary-revised"
    sources = sorted(p for p in paper.rglob("*") if p.suffix in (".tex", ".png"))
    parent = json.loads((original / "paper-manifest.json").read_text(encoding="utf-8"))
    for name, expected in {**parent["source_files"], **parent["artifacts"]}.items():
        if sha(original / name) != expected:
            raise ValueError(f"Original publication changed: {name}")
    for path in sources:
        relative = path.relative_to(paper)
        if path.suffix == ".png" or relative.as_posix() in {
            "references.tex", "generated/data-table.tex", "generated/supporting.tex"
        }:
            if path.read_bytes() != (original / relative).read_bytes():
                raise ValueError(f"Editorial revision changed scientific content: {relative}")
    results = [
        (directory / "generated/main-results.tex").read_text(encoding="utf-8")
        for directory in (original, paper)
    ]
    for pattern in (r"\d+(?:\.\d+)?", r"\\begin\{table\}.*?\\end\{table\}"):
        if re.findall(pattern, results[0], re.S) != re.findall(pattern, results[1], re.S):
            raise ValueError("Editorial revision changed result numbers or tables")

    with tempfile.TemporaryDirectory(prefix="triage-paper-") as temporary:
        clean = Path(temporary)
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
        "edition": "Editorial revision; unchanged binary-v2 results",
        "status": "compiled; visual review recorded separately in review.json",
        "parent_manifest_sha256": sha(original / "paper-manifest.json"),
        "freeze_sha256": parent["evidence_sha256"]["freeze.json"],
        "evidence_sha256": parent["evidence_sha256"],
        "builder_sha256": sha(Path(__file__)),
        "engine": subprocess.check_output([engine, "--version"], text=True).strip(),
        "source_files": {p.relative_to(paper).as_posix(): sha(p) for p in sources},
        "artifacts": {name: sha(paper / name) for name in ("paper.pdf", "arxiv-source.zip", "build-log.txt")},
        "checks": "Original release hashes, unchanged result numbers/tables, figures and references; fresh archive build",
    }
    (paper / "paper-manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="tectonic", help="Tectonic executable (0.17.0 tested)")
    build(parser.parse_args().engine)
