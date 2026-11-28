"""Stage 1b: Verify. Check every new or changed document BEFORE it is chunked (RAGOps A1).

A bad document does more damage in an index than a bad line of code: every answer that touches
it is wrong and nothing fails loudly. Four checks per document, each recorded with a reason:

  quality       is the text usable? Failed extraction, empty pages, garbled characters, scans
                whose OCR found almost nothing.
  completeness  required metadata present (source, title, date or version, page count) and the
                text not truncated (a .md/.txt that stops mid-sentence).
  recency       is a newer version of the same doc_id present? The older one is NOT quarantined:
                the report says `fail`, and ingestion archives it (metadata only; it can be
                restored). A superseded or old current document is a warning.
  uniqueness    exact duplicate (hash of the file bytes or of the normalised text) or
                near-duplicate (cosine of embeddings of a text sample) of a document accepted
                earlier in this run.

`fail` in quality, completeness or uniqueness quarantines the document: it is moved to
<base>/quarantine/ (copied when the corpus is read-only) with a reason file, and it is not
chunked, embedded or indexed. The run goes on with the rest. Every run writes
<base>/reports/verification-<UTC stamp>.json (the contract in docs/CONTRACT.md, section 3) and
a .md. <base> is $DATA_DIR, else $VERIFY_DIR, else /tmp/verification. Thresholds are VERIFY_*
environment variables, so on Kubernetes they live in the ConfigMap.

Standalone, without a cluster:
    python -m pipeline.verify --corpus ../corpus/raw                # report only
    python -m pipeline.verify --corpus ../corpus/raw --no-embed     # skip the near-duplicate check
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
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import Document

EmbedFn = Callable[[list[str]], list[list[float]]]
MIN_TEXT_CHARS = 25          # same floor extract.py uses for "this page has a text layer"
YEAR = re.compile(r"(?<!\d)(\d{4})(?:\s*[-/]\s*(\d{2,4}))?(?!\d)")
SENTENCE_END = tuple('.!?)"\'”:;|')
PLAIN_TEXT = {".md", ".txt"}   # no running footers: a missing full stop at the end means cut off


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
    min_ocr_chars_per_page: int = 300    # ... i.e. fewer characters per page than this
    warn_ocr_share: float = 0.3
    warn_age_years: int = 3              # recency: a current document older than this warns
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
            near_dup_fail=_env("VERIFY_NEAR_DUP_FAIL", d.near_dup_fail),
            near_dup_warn=_env("VERIFY_NEAR_DUP_WARN", d.near_dup_warn),
        )


@dataclass
class Check:
    name: str
    status: str            # pass | warn | fail | skip
    detail: str
    blocks: bool = True    # a fail that blocks indexing (recency "newer version exists" does not)


@dataclass
class DocReport:
    source: str
    doc_id: str
    doc_version: str
    superseded: bool
    pages: int
    chars: int
    checks: list[Check] = field(default_factory=list)

    @property
    def action(self) -> str:
        return "quarantined" if any(c.status == "fail" and c.blocks for c in self.checks) else "indexed"

    @property
    def reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if c.status == "fail" and c.blocks]

    def as_json(self) -> dict[str, Any]:
        return {"source": self.source, "doc_id": self.doc_id, "doc_version": self.doc_version,
                "superseded": self.superseded, "pages": self.pages, "chars": self.chars,
                "checks": {c.name: {"result": "fail" if c.status == "fail" else "pass", "status": c.status,
                                    "reason": c.detail} for c in self.checks},
                "action": self.action, "reasons": self.reasons}


# ------------------------------------------------------------------------------- helpers

def full_text(doc: Document) -> str:
    return "\n".join(p.text for p in doc.pages)


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def text_hash(doc: Document) -> str:
    return hashlib.sha256(normalise(full_text(doc)).encode("utf-8")).hexdigest()


def file_hash(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(doc: Document, size: int = 1500) -> str:
    """The text sample that stands for a document in the near-duplicate comparison: the start of
    the first page and of the middle page (a whole 50-page document does not fit one embedding,
    and the start alone would match any two documents that share a cover page)."""
    if not doc.pages:
        return ""
    pages = [doc.pages[0], doc.pages[len(doc.pages) // 2]]
    return " ".join(normalise(p.text)[:size] for p in pages)


def doc_title(doc: Document) -> str:
    """First non-empty line of the first page, minus Markdown marks (the pipeline has no
    separate title field; this is what a reader would call the title)."""
    for page in doc.pages[:1]:
        for line in page.text.splitlines():
            line = line.strip().lstrip("#").strip()
            if len(line) >= 3:
                return line[:120]
    return ""


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
        return Check("quality", "fail", "no pages extracted")
    n = len(doc.pages)
    empty = sum(1 for p in doc.pages if len(p.text.strip()) < MIN_TEXT_CHARS) / n
    ocr = sum(1 for p in doc.pages if p.ocr) / n
    text = full_text(doc)
    garbled = sum(1 for ch in text if ch == "�" or (ord(ch) < 32 and ch not in "\n\t\r")) / max(len(text), 1)
    density = len(text) / n
    stats = f"empty pages {empty:.0%}, OCR pages {ocr:.0%}, garbled chars {garbled:.1%}, {density:.0f} chars/page"
    # A scan is not bad data; unreadable OCR is. Mostly-OCR fails only when the OCR also found
    # little text (the first real run quarantined a clean 26-page scan: see the report).
    thin_scan = ocr > t.max_ocr_share and density < t.min_ocr_chars_per_page
    if empty > t.max_empty_page_share or garbled > t.max_garbled_share or thin_scan:
        return Check("quality", "fail", stats)
    if empty > t.warn_empty_page_share or garbled > t.warn_garbled_share or ocr > t.warn_ocr_share:
        return Check("quality", "warn", stats)
    return Check("quality", "pass", stats)


def check_completeness(doc: Document, t: Thresholds, suffix: str = "") -> Check:
    """Required metadata (source, title, date or version, page count) and not truncated."""
    problems: list[str] = []
    warnings: list[str] = []
    if not doc.source:
        problems.append("source missing")
    if not doc.pages:
        problems.append("page count 0")
    if doc.doc_version.strip().lower() in ("", "unknown"):
        problems.append("doc_version unknown (recency and archiving need it)")
    if doc.pages and not doc_title(doc):
        problems.append("no title (the first page has no usable first line)")
    chars = sum(len(p.text.strip()) for p in doc.pages)
    if chars < t.min_total_chars:
        problems.append(f"only {chars} characters of text (need {t.min_total_chars})")
    last = next((ln.strip() for ln in reversed(full_text(doc).splitlines()) if ln.strip()), "")
    if len(last) > 20 and not last.endswith(SENTENCE_END):
        (problems if suffix.lower() in PLAIN_TEXT else warnings).append(
            f"text ends mid-sentence ({last[-40:]!r}): truncated?")
    if problems:
        return Check("completeness", "fail", "; ".join(problems))
    if warnings:
        return Check("completeness", "warn", "; ".join(warnings))
    return Check("completeness", "pass",
                 f"source, title, version {doc.doc_version}, {len(doc.pages)} page(s), {chars} characters")


def check_recency(doc: Document, doc_id: str, peers: dict[str, list[tuple[str, str, bool]]], t: Thresholds,
                  today: dt.date) -> Check:
    """peers: doc_id -> [(source, doc_version, superseded)] for every file of this run's corpus."""
    newer = [(s, v) for s, v, sup in peers.get(doc_id, []) if s != doc.source and not sup and v > doc.doc_version]
    if newer and not doc.superseded:
        s, v = max(newer, key=lambda x: x[1])
        return Check("recency", "fail", f"newer version {v} of doc_id {doc_id} exists ({s}): this one will be "
                     "archived (excluded from retrieval, restorable), not deleted", blocks=False)
    year = version_year(doc.doc_version)
    if doc.superseded:
        return Check("recency", "warn", f"marked superseded ({doc.doc_version}): kept, archived, filtered at query time")
    if year is None:
        return Check("recency", "skip", f"no year in doc_version {doc.doc_version!r}")
    age = today.year - year
    if age > t.warn_age_years:
        return Check("recency", "warn", f"newest version of its doc_id but {age} years old (warn above {t.warn_age_years})")
    return Check("recency", "pass", f"newest version of doc_id {doc_id}, version year {year}")


def check_uniqueness(doc: Document, fhash: str | None, seen_hash: dict[str, str], vec: list[float] | None,
                     seen_vec: list[tuple[str, list[float]]], t: Thresholds) -> Check:
    for h in (fhash, text_hash(doc)):
        if h and h in seen_hash:
            return Check("uniqueness", "fail", f"exact duplicate of {seen_hash[h]} (same content hash)")
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
                     today: dt.date | None = None, *, doc_ids: dict[str, str] | None = None,
                     peers: dict[str, list[tuple[str, str, bool]]] | None = None,
                     paths: dict[str, Path] | None = None) -> list[DocReport]:
    """Run the four checks on every document, in order; an earlier accepted document wins a
    duplicate pair. `doc_ids` maps source -> doc_id (default: the file stem), `peers` lists the
    versions of each doc_id on disk, `paths` gives the files (for the byte hash and the suffix)."""
    t = t or Thresholds.from_env()
    today = today or dt.date.today()
    doc_ids = doc_ids or {}
    paths = paths or {}
    peers = peers if peers is not None else _peers(docs, doc_ids)
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
        did = doc_ids.get(d.source) or Path(d.source).stem
        rep = DocReport(d.source, did, d.doc_version, d.superseded, len(d.pages), sum(len(p.text) for p in d.pages))
        p = paths.get(d.source)
        fh = file_hash(p)
        rep.checks = [check_quality(d, t), check_completeness(d, t, p.suffix if p else Path(d.source).suffix),
                      check_recency(d, did, peers, t, today),
                      check_uniqueness(d, fh, seen_hash, vectors[i] if vectors else None, seen_vec, t)]
        if note:
            rep.checks[-1].detail += f" [{note}]"
        if rep.action == "indexed":
            seen_hash[text_hash(d)] = d.source
            if fh:
                seen_hash[fh] = d.source
            if vectors:
                seen_vec.append((d.source, vectors[i]))
        reports.append(rep)
    return reports


def _peers(docs: list[Document], doc_ids: dict[str, str]) -> dict[str, list[tuple[str, str, bool]]]:
    out: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    for d in docs:
        out[doc_ids.get(d.source) or Path(d.source).stem].append((d.source, d.doc_version, d.superseded))
    return dict(out)


@dataclass
class Outcome:
    accepted: list[Any]                       # the entries that may be chunked (not quarantined)
    reports: list[DocReport]
    cache: dict[str, Document]                # path -> cleaned document, so extraction is not repeated


def verify_entries(entries: list[Any], *, extract_fn: Callable[..., Document], clean_fn: Callable[..., Document],
                   embed_fn: EmbedFn | None = None, only: set[str] | None = None, t: Thresholds | None = None,
                   today: dt.date | None = None) -> Outcome:
    """Verify corpus entries (objects with .path .doc_version .superseded .doc_id .source).
    `only` limits the work to new or changed sources; the others are not read at all but still
    count as versions of their doc_id. A file that cannot be extracted is a quality failure in
    the report, not a silent skip."""
    # When two files are duplicates the first one checked survives. Check the shortest file name first, so
    # the original (policy-2026.md) wins over its copy (policy-2026-copy.md, -final.md, -reworded.md).
    todo = sorted((e for e in entries if only is None or e.source in only), key=lambda e: (len(e.source), e.source))
    docs: list[Document] = []
    failed: list[DocReport] = []
    cache: dict[str, Document] = {}
    by_source = {e.source: e for e in todo}
    for e in todo:
        try:
            doc = clean_fn(extract_fn(str(e.path), e.doc_version, e.superseded))
        except Exception as exc:  # noqa: BLE001 - a corrupt file is exactly what this stage must report
            rep = DocReport(e.source, e.doc_id, e.doc_version, e.superseded, 0, 0)
            rep.checks = [Check("quality", "fail", f"extraction failed: {exc!r}"),
                          Check("completeness", "skip", "not checked: no text"),
                          Check("recency", "skip", "not checked: no text"),
                          Check("uniqueness", "skip", "not checked: no text")]
            failed.append(rep)
            continue
        docs.append(doc)
        cache[str(e.path)] = doc
    peers: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    for e in entries:
        peers[e.doc_id].append((e.source, e.doc_version, e.superseded))
    reports = verify_documents(docs, t, embed_fn, today, doc_ids={e.source: e.doc_id for e in entries},
                               peers=dict(peers), paths={e.source: e.path for e in todo})
    reports = failed + reports
    keep = {r.source for r in reports if r.action == "quarantined"}
    return Outcome([e for e in entries if e.source not in keep or e.source not in by_source], reports, cache)


def cached_fns(cache: dict[str, Document], extract_fn: Callable[..., Any], clean_fn: Callable[..., Any]
               ) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Extract and clean functions that reuse the documents verification already read, so a
    scanned 26-page PDF is not OCR'd twice."""
    cleaned = {id(d) for d in cache.values()}

    def extract(path: str, version: str, superseded: bool) -> Any:
        return cache.get(path) or extract_fn(path, version, superseded)

    def clean(doc: Any) -> Any:
        return doc if id(doc) in cleaned else clean_fn(doc)

    return extract, clean


# ------------------------------------------------------------------------------- outputs

def base_dir() -> Path:
    return Path(os.environ.get("DATA_DIR") or os.environ.get("VERIFY_DIR") or "/tmp/verification")


def manifest_problems(corpus_dir: str) -> list[str]:
    """Corpus level completeness: every file _index.json lists must exist."""
    index = Path(corpus_dir) / "_index.json"
    if not index.exists():
        return []
    entries = json.loads(index.read_text(encoding="utf-8"))
    return [f"manifest lists {e['file']} but the file is missing"
            for e in entries if not (Path(corpus_dir) / e["file"]).exists()]


def write_report(reports: list[DocReport], base: str | Path, t: Thresholds, extra: list[str] | None = None,
                 now: dt.datetime | None = None, unchanged: int = 0) -> tuple[Path, Path]:
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    out = Path(base) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    quarantined = [r for r in reports if r.action == "quarantined"]
    indexed = [r for r in reports if r.action == "indexed"]
    flagged = [r for r in indexed if any(c.status in ("warn", "fail") for c in r.checks)]
    data = {"run": stamp, "thresholds": asdict(t), "documents": len(reports), "indexed": len(indexed),
            "quarantined": len(quarantined), "indexed_with_flags": len(flagged), "unchanged_not_checked": unchanged,
            "corpus_problems": extra or [], "reports": [r.as_json() for r in reports]}
    jp, mp = out / f"verification-{stamp}.json", out / f"verification-{stamp}.md"
    jp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [f"# Verification report {stamp}", "",
             f"{len(reports)} documents checked ({unchanged} unchanged ones skipped): **{len(indexed)} indexed** "
             f"({len(flagged)} with flags), **{len(quarantined)} quarantined**.", ""]
    if extra:
        lines += ["Corpus problems:", *[f"- {e}" for e in extra], ""]
    lines += ["| Document | doc_id | Version | Action | quality | completeness | recency | uniqueness |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in reports:
        by = {c.name: c.status for c in r.checks}
        lines.append(f"| {r.source} | {r.doc_id} | {r.doc_version} | {r.action} | {by.get('quality')} | "
                     f"{by.get('completeness')} | {by.get('recency')} | {by.get('uniqueness')} |")
    lines += ["", "## Details", ""]
    for r in reports:
        lines.append(f"### {r.source} ({r.action})")
        lines += [f"- {c.name} [{c.status}]: {c.detail}" for c in r.checks]
        lines.append("")
    mp.write_text("\n".join(lines), encoding="utf-8")
    return jp, mp


def quarantine_files(reports: list[DocReport], paths: dict[str, Path], base: str | Path) -> list[Path]:
    """Move each quarantined original to <base>/quarantine/ with a <name>.reason.json. A read-only
    corpus (the Job mounts it that way) cannot be moved from: the file is copied, and the report says so."""
    qdir = Path(base) / "quarantine"
    moved = []
    for r in reports:
        if r.action != "quarantined":
            continue
        qdir.mkdir(parents=True, exist_ok=True)
        src = paths.get(r.source)
        how = "not found"
        if src and src.exists():
            try:
                shutil.move(str(src), str(qdir / r.source))
                how = "moved"
            except OSError:
                shutil.copy2(src, qdir / r.source)
                how = "copied (corpus is read-only)"
        (qdir / f"{r.source}.reason.json").write_text(json.dumps(
            {"source": r.source, "doc_id": r.doc_id, "how": how, "reasons": r.reasons}, indent=2))
        moved.append(qdir / r.source)
    return moved


def run_stage(entries: list[Any], *, extract_fn: Callable[..., Any], clean_fn: Callable[..., Any],
              embed_fn: EmbedFn | None, only: set[str] | None, corpus_dir: str, log: Callable[..., None]) -> Outcome:
    """The whole stage as the ingestion Job runs it: verify, write the report, quarantine."""
    t = Thresholds.from_env()
    outcome = verify_entries(entries, extract_fn=extract_fn, clean_fn=clean_fn, embed_fn=embed_fn, only=only, t=t)
    base = base_dir()
    unchanged = len(entries) - len(outcome.reports)
    jp, _ = write_report(outcome.reports, base, t, manifest_problems(corpus_dir), unchanged=unchanged)
    quarantine_files(outcome.reports, {e.source: e.path for e in entries}, base)
    for r in outcome.reports:
        log("verified", source=r.source, doc_id=r.doc_id, action=r.action,
            checks={c.name: c.status for c in r.checks}, reasons=r.reasons)
    log("verification summary", indexed=sum(r.action == "indexed" for r in outcome.reports),
        quarantined=sum(r.action == "quarantined" for r in outcome.reports), unchanged_not_checked=unchanged,
        report=str(jp))
    return outcome


def main() -> int:
    from .clean import clean
    from .config import get_settings
    from .embed import embed
    from .extract import extract
    from .run import discover_entries

    ap = argparse.ArgumentParser(description="Verify the documents in a corpus folder (report only).")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS_DIR", "/corpus"))
    ap.add_argument("--out", default=str(base_dir()), help="base folder: reports/ and quarantine/ go below it")
    ap.add_argument("--no-embed", action="store_true", help="skip the near-duplicate check")
    ap.add_argument("--quarantine", action="store_true", help="also move failed files to <out>/quarantine/")
    args = ap.parse_args()
    settings = get_settings()
    entries = discover_entries(args.corpus)
    t = Thresholds.from_env()
    outcome = verify_entries(entries, extract_fn=extract, clean_fn=clean, t=t,
                             embed_fn=None if args.no_embed else (lambda texts: embed(texts, settings)))
    jp, mp = write_report(outcome.reports, args.out, t, manifest_problems(args.corpus))
    if args.quarantine:
        quarantine_files(outcome.reports, {e.source: e.path for e in entries}, args.out)
    print(mp.read_text(encoding="utf-8"))
    print(f"written: {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
