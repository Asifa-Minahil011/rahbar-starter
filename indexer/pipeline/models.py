"""The shapes passed between stages. Metadata fields are required (section 3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Page:
    """Text of one page, as extraction produced it."""

    source: str              # file name, e.g. Academic_Rules_Handbook_2026.pdf
    page: int                # 1-based
    text: str
    ocr: bool = False        # True when this page had no text layer and was OCR'd


@dataclass
class Document:
    source: str
    doc_version: str         # e.g. "2026-06"; from corpus/sources.yaml
    superseded: bool         # True for an older version of a policy
    pages: list[Page] = field(default_factory=list)


@dataclass
class Chunk:
    id: str                  # stable: same document + same position gives the same id
    text: str
    source: str
    section: str             # the heading this chunk sits under, e.g. "5.4 Academic Warning"
    page: int
    doc_version: str
    superseded: bool
    # A2 (docs/CONTRACT.md section 1). Defaults keep every older call site working.
    doc_id: str = ""         # logical document; all versions of a policy share it
    content_hash: str = ""   # SHA-256 of the source file's bytes: change detection
    status: str = "active"   # "active" or "archived"; archived also sets superseded=True
    pinned: bool = False     # an operator override: ingestion will not change `status`
    ingested_at: str = ""    # UTC time of the run that wrote this chunk

    def metadata(self) -> dict[str, str | int | bool]:
        """Chroma metadata values must be str, int, float or bool."""
        return {
            "source": self.source,
            "section": self.section,
            "page": self.page,
            "doc_version": self.doc_version,
            "superseded": self.superseded,
            "doc_id": self.doc_id,
            "content_hash": self.content_hash,
            "status": self.status,
            "pinned": self.pinned,
            "ingested_at": self.ingested_at,
        }
