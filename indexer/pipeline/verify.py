"""Stage 1b: Verify. Check each extracted document BEFORE it is chunked (RAGOps A1).

The data lifecycle needs its own quality gate: a bad document does more damage in an index
than a bad line of code, because every answer that touches it is wrong and nothing fails
loudly. Four checks per document, each `pass`, `warn` or `fail`:

  quality       is the text usable? Empty pages, garbled characters, mostly-OCR scans.
  completeness  does it carry what the pipeline needs? A known doc_version, at least one
                page, enough text; and (corpus level) every file the manifest lists exists.
  recency       how old is its version? A superseded document is kept (the API filters it
                at query time; see doc_version / superseded) but flagged; a current one
                that is old is a warning, a very old one fails.
  uniqueness    exact duplicate (hash of the normalised text) or near-duplicate (cosine of
                the embeddings of a text sample) of a document accepted earlier in this run.

Any `fail` quarantines the document: it is not chunked or embedded, the original file is
copied to <verify_dir>/quarantine/ with a reason file, and the run goes on with the rest.
`warn` keeps the document and appears in the report. Every run writes
<verify_dir>/verification-<UTC stamp>.json and .md. Thresholds are environment variables
(VERIFY_*), so on Kubernetes they live in the ConfigMap like every other knob.

Standalone, without a cluster:
    python -m pipeline.verify --corpus ../corpus/raw            # report only
    python -m pipeline.verify --corpus ../corpus/raw --no-embed # skip the near-duplicate check
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import Document

EmbedFn = Callable[[list[str]], list[list[float]]]
MIN_TEXT_CHARS = 25          # same floor extract.py uses for "this page has a text layer"
YEAR = re.compile(r"(?<!\d)(\d{4})(?:\s*[-/]\s*(\d{2,4}))?(?!\d)")


def _env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Thresholds:
    min_total_chars: int = 100           # completeness: less text than this is not a document
    max_empty_page_share: float = 0.5    # quality: pages with no usable text; fail above
    warn_empty_page_share: float = 0.2
    max_garbled_share: float = 0.10      # quality: replacement/control characters; fail above
    warn_garbled_share: float = 0.03
    max_ocr_share: float = 0.8           # quality: fail only if this many pages are OCR'd AND the OCR is thin
    min_ocr_chars_per_page: int = 300    # ... i.e. fewer characters per page than this (OCR found little text)
    warn_ocr_share: float = 0.3
    warn_age_years: int = 3              # recency: current document older than this warns
    fail_age_years: int = 8              # ... and fails above this
    near_dup_fail: float = 0.985         # uniqueness: cosine of the text-sample embeddings
    near_dup_warn: float = 0.95

    @classmethod
    def from_env(cls) -> Thresholds:
        d = cls()
        return cls(
            min_total_chars=int(_env("VERIFY_MIN_TOTAL_CHARS", d.min_total_chars)),
            max_empty_page_share=_env("VERIFY_MAX_EMPTY_PAGE_SHARE", d.max_empty_page_share),
            warn_empty_page_share=_env("VERIFY_WARN_EMPTY_PAGE_SHARE", d.warn_empty_page_share),
            max_garbled_share=_env("VERIFY_MAX_GARBLED_SHARE", d.max_garbled_share),
            warn_garbled_share=_env("VERIFY_WARN_GARBLED_SHARE", d.warn_garbled_share),
            max_ocr_share=_env("VERIFY_MAX_OCR_SHARE", d.max_ocr_share),
            min_ocr_chars_per_page=int(_env("VERIFY_MIN_OCR_CHARS_PER_PAGE", d.min_ocr_chars_per_page)),
            warn_ocr_share=_env("VERIFY_WARN_OCR_SHARE", d.warn_ocr_share),
            warn_age_years=int(_env("VERIFY_WARN_AGE_YEARS", d.warn_age_years)),
            fail_age_years=int(_env("VERIFY_FAIL_AGE_YEARS", d.fail_age_years)),
            near_dup_fail=_env("VERIFY_NEAR_DUP_FAIL", d.near_dup_fail),
            near_dup_warn=_env("VERIFY_NEAR_DUP_WARN", d.near_dup_warn),
        )


@dataclass
class Check:
    name: str
    status: str            # pass | warn | fail | skip
    detail: str


@dataclass
class DocReport:
    source: str
    doc_version: str
    superseded: bool
    pages: int
    chars: int
    checks: list[Check] = field(default_factory=list)
    decision: str = "accept"            # accept | quarantine

    @property
    def reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if c.status == "fail"]


# ------------------------------------------------------------------------------- helpers

def full_text(doc: Document) -> str:
    return "\n".join(p.text for p in doc.pages)


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def text_hash(doc: Document) -> str:
    return hashlib.sha256(normalise(full_text(doc)).encode("utf-8")).hexdigest()


def fingerprint(doc: Document, size: int = 1500) -> str:
    """A sample of the text that stands for the document in the near-duplicate comparison:
    the start of the first page and of the middle page (an embedding of a whole 50-page
    document is not possible, and the start alone would match any two documents that share
    a cover page)."""
    if not doc.pages:
        return ""
    pages = [doc.pages[0], doc.pages[len(doc.pages) // 2]]
    return " ".join(normalise(p.text)[:size] for p in pages)


def version_year(version: str) -> int | None:
    """2026-06 -> 2026, 2019-20 -> 2020 (a fiscal year counts from its end), 2026 -> 2026."""
    m = YEAR.search(version or "")
    if not m:
        return None
    first, second = int(m.group(1)), m.group(2)
    if second and len(second) == 2 and int(second) == (first + 1) % 100:
        return first + 1
    return first


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


# ------------------------------------------------------------------------------- checks

def check_quality(doc: Document, t: Thresholds) -> Check:
    if not doc.pages:
        return Check("quality", "fail", "no pages")
    n = len(doc.pages)
    empty = sum(1 for p in doc.pages if len(p.text.strip()) < MIN_TEXT_CHARS) / n
    ocr = sum(1 for p in doc.pages if p.ocr) / n
    text = full_text(doc)
    garbled = sum(1 for ch in text if ch == "�" or (ord(ch) < 32 and ch not in "\n\t\r")) / max(len(text), 1)
    density = len(text) / n
    stats = (f"empty pages {empty:.0%}, OCR pages {ocr:.0%}, garbled chars {garbled:.1%}, "
             f"{density:.0f} chars/page")
    # A scan is not bad data; unreadable OCR is. Mostly-OCR fails only when the OCR also
    # found little text. (The first run quarantined a clean 26-page scan: see the report.)
    thin_scan = ocr > t.max_ocr_share and density < t.min_ocr_chars_per_page
    if empty > t.max_empty_page_share or garbled > t.max_garbled_share or thin_scan:
        return Check("quality", "fail", stats)
    if empty > t.warn_empty_page_share or garbled > t.warn_garbled_share or ocr > t.warn_ocr_share:
        return Check("quality", "warn", stats)
    return Check("quality", "pass", stats)


def check_completeness(doc: Document, t: Thresholds) -> Check:
    problems = []
    if not doc.pages:
        problems.append("no pages")
    if doc.doc_version.strip().lower() in ("", "unknown"):
        problems.append("doc_version unknown (recency and superseded filtering need it)")
    chars = sum(len(p.text.strip()) for p in doc.pages)
    if chars < t.min_total_chars:
        problems.append(f"only {chars} characters of text (need {t.min_total_chars})")
    if problems:
        return Check("completeness", "fail", "; ".join(problems))
    return Check("completeness", "pass", f"version {doc.doc_version}, {len(doc.pages)} page(s), {chars} characters")


def check_recency(doc: Document, t: Thresholds, today: dt.date) -> Check:
    year = version_year(doc.doc_version)
    if year is None:
        return Check("recency", "skip", f"no year in doc_version {doc.doc_version!r}")
    age = today.year - year
    if doc.superseded:
        return Check("recency", "warn", f"superseded version from {year}: kept, filtered at query time")
    if age > t.fail_age_years:
        return Check("recency", "fail", f"current document is {age} years old (limit {t.fail_age_years})")
    if age > t.warn_age_years:
        return Check("recency", "warn", f"{age} years old (warn above {t.warn_age_years})")
    return Check("recency", "pass", f"version year {year}, {age} year(s) old")


def check_uniqueness(doc: Document, seen_hash: dict[str, str], vec: list[float] | None,
                     seen_vec: list[tuple[str, list[float]]], t: Thresholds) -> Check:
    h = text_hash(doc)
    if h in seen_hash:
        return Check("uniqueness", "fail", f"exact duplicate of {seen_hash[h]} (same normalised text hash)")
    if vec is None:
        return Check("uniqueness", "pass", "no exact duplicate; near-duplicate check skipped (no embeddings)")
    best_name, best = "", 0.0
    for name, other in seen_vec:
        c = cosine(vec, other)
        if c > best:
            best_name, best = name, c
    if best >= t.near_dup_fail:
        return Check("uniqueness", "fail", f"near-duplicate of {best_name} (cosine {best:.3f} >= {t.near_dup_fail})")
    if best >= t.near_dup_warn:
        return Check("uniqueness", "warn", f"very similar to {best_name} (cosine {best:.3f})")
    return Check("uniqueness", "pass", f"closest earlier document {best_name or '-'} at cosine {best:.3f}")


# ------------------------------------------------------------------------------- the run

def verify_documents(docs: list[Document], t: Thresholds | None = None, embed_fn: EmbedFn | None = None,
                     today: dt.date | None = None) -> list[DocReport]:
    """Run the four checks on every document, in order; an earlier accepted document wins a
    duplicate pair. Returns one report per document; `decision` says what to do with it."""
    t = t or Thresholds.from_env()
    today = today or dt.date.today()
    vectors: list[list[float]] | None = None
    note = ""
    if embed_fn is not None and docs:
        try:
            vectors = embed_fn([fingerprint(d) for d in docs])
        except Exception as exc:  # noqa: BLE001 - verification must not sink the run
            note = f"near-duplicate check skipped: embedding failed ({exc!r})"
    seen_hash: dict[str, str] = {}
    seen_vec: list[tuple[str, list[float]]] = []
    reports = []
    for i, d in enumerate(docs):
        rep = DocReport(d.source, d.doc_version, d.superseded, len(d.pages), sum(len(p.text) for p in d.pages))
        rep.checks = [check_quality(d, t), check_completeness(d, t), check_recency(d, t, today),
                      check_uniqueness(d, seen_hash, vectors[i] if vectors else None, seen_vec, t)]
        if note:
            rep.checks[-1].detail += f" [{note}]"
        if any(c.status == "fail" for c in rep.checks):
            rep.decision = "quarantine"
        else:
            seen_hash[text_hash(d)] = d.source
            if vectors:
                seen_vec.append((d.source, vectors[i]))
        reports.append(rep)
    return reports


def manifest_problems(corpus_dir: str) -> list[str]:
    """Corpus level completeness: every file _index.json lists must exist."""
    index = Path(corpus_dir) / "_index.json"
    if not index.exists():
        return []
    entries = json.loads(index.read_text(encoding="utf-8"))
    return [f"manifest lists {e['file']} but the file is missing"
            for e in entries if not (Path(corpus_dir) / e["file"]).exists()]


def write_report(reports: list[DocReport], out_dir: str, t: Thresholds, extra: list[str] | None = None,
                 now: dt.datetime | None = None) -> tuple[Path, Path]:
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted = [r for r in reports if r.decision == "accept"]
    quarantined = [r for r in reports if r.decision == "quarantine"]
    warned = [r for r in accepted if any(c.status == "warn" for c in r.checks)]
    data = {"run": stamp, "thresholds": asdict(t), "documents": len(reports), "accepted": len(accepted),
            "quarantined": len(quarantined), "accepted_with_warnings": len(warned),
            "corpus_problems": extra or [], "reports": [{**asdict(r), "reasons": r.reasons} for r in reports]}
    jp, mp = out / f"verification-{stamp}.json", out / f"verification-{stamp}.md"
    jp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [f"# Verification report {stamp}", "",
             f"{len(reports)} documents: **{len(accepted)} accepted** ({len(warned)} with warnings), "
             f"**{len(quarantined)} quarantined**.", ""]
    if extra:
        lines += ["Corpus problems:", *[f"- {e}" for e in extra], ""]
    lines += ["| Document | Version | Decision | quality | completeness | recency | uniqueness |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in reports:
        by = {c.name: c.status for c in r.checks}
        lines.append(f"| {r.source} | {r.doc_version} | {r.decision} | {by.get('quality')} | "
                     f"{by.get('completeness')} | {by.get('recency')} | {by.get('uniqueness')} |")
    lines += ["", "## Details", ""]
    for r in reports:
        lines.append(f"### {r.source} ({r.decision})")
        lines += [f"- {c.name} [{c.status}]: {c.detail}" for c in r.checks]
        lines.append("")
    mp.write_text("\n".join(lines), encoding="utf-8")
    return jp, mp


def quarantine_files(reports: list[DocReport], paths: dict[str, Path], out_dir: str) -> list[Path]:
    """Copy each quarantined original into <out_dir>/quarantine/ with a <name>.reason.json."""
    qdir = Path(out_dir) / "quarantine"
    moved = []
    for r in reports:
        if r.decision != "quarantine":
            continue
        qdir.mkdir(parents=True, exist_ok=True)
        src = paths.get(r.source)
        if src and src.exists():
            shutil.copy2(src, qdir / r.source)
        (qdir / f"{r.source}.reason.json").write_text(json.dumps({"source": r.source, "reasons": r.reasons}, indent=2))
        moved.append(qdir / r.source)
    return moved


def main() -> int:
    from .clean import clean
    from .config import get_settings
    from .embed import embed
    from .extract import extract
    from .run import discover

    ap = argparse.ArgumentParser(description="Verify the documents in a corpus folder (report only).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS_DIR", "/corpus"))
    ap.add_argument("--out", default=os.environ.get("VERIFY_DIR", "verification"))
    ap.add_argument("--no-embed", action="store_true", help="skip the near-duplicate check")
    ap.add_argument("--quarantine", action="store_true", help="also copy failed files to <out>/quarantine/")
    args = ap.parse_args()
    settings = get_settings()
    docs, paths = [], {}
    for path, version, superseded in discover(args.corpus):
        docs.append(clean(extract(str(path), version, superseded)))
        paths[path.name] = path
    t = Thresholds.from_env()
    reports = verify_documents(docs, t, None if args.no_embed else (lambda texts: embed(texts, settings)))
    jp, mp = write_report(reports, args.out, t, manifest_problems(args.corpus))
    if args.quarantine:
        quarantine_files(reports, paths, args.out)
    print(mp.read_text(encoding="utf-8"))
    print(f"written: {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
