"""Guards the Batch B fixtures: each planted problem is really there, so an accidental edit
cannot silently turn a duplicate into a unique file or a truncated file into a complete one."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.extract import extract
from pipeline.run import _front_matter

B = Path(__file__).resolve().parents[2] / "corpus" / "batch_b"
pytestmark = pytest.mark.skipif(not B.exists(), reason="corpus/batch_b not present")


def sha(name: str) -> str:
    return hashlib.sha256((B / name).read_bytes()).hexdigest()


def body(name: str) -> str:
    return (B / name).read_text(encoding="utf-8").split("---\n", 2)[-1]


def test_answer_key_covers_every_file() -> None:
    key = json.loads((B / "planted.json").read_text(encoding="utf-8"))["files"]
    on_disk = {p.name for p in B.iterdir() if p.suffix in {".md", ".pdf"} and p.name != "README.md"}
    assert {k["file"] for k in key} == on_disk and len(key) >= 10


def test_two_exact_duplicates() -> None:
    assert sha("library-borrowing-policy-2026-copy.md") == sha("library-borrowing-policy-2026.md")
    assert sha("scholarship-and-financial-aid-policy-2026-final.md") == \
        sha("scholarship-and-financial-aid-policy-2026.md")


def test_one_near_duplicate_that_is_not_byte_identical() -> None:
    a, b = "library-borrowing-policy-2026.md", "library-borrowing-policy-2026-reworded.md"
    assert sha(a) != sha(b)
    assert difflib.SequenceMatcher(None, body(a), body(b)).ratio() > 0.9


def test_truncated_file_ends_mid_sentence_and_underdelivers_its_pages() -> None:
    text = body("transport-policy-2026-truncated.md").rstrip()
    assert not text.endswith((".", "!", "?")) and _front_matter(
        B / "transport-policy-2026-truncated.md")["pages"] == "3"


def test_corrupt_pdf_cannot_be_extracted() -> None:
    with pytest.raises(Exception):
        extract(str(B / "attendance-notice-corrupt.pdf"), "unknown", False)


def test_missing_metadata_file_has_no_front_matter() -> None:
    assert _front_matter(B / "research-ethics-guidelines.md") == {}


def test_newer_version_shares_the_doc_id_of_the_2022_policy() -> None:
    meta = _front_matter(B / "faculty-entrepreneurship-and-technical-start-ups-policy-2026-revision.md")
    assert meta["doc_id"] == "faculty-entrepreneurship-and-technical-start-ups-policy"
    assert meta["doc_version"] > "2022-05"          # string order, as in incremental.py


def test_injected_instruction_is_present() -> None:
    assert "ignore all previous instructions" in body("library-winter-break-notice-injected.md").lower()


def test_every_markdown_file_says_it_is_synthetic() -> None:
    for p in B.glob("*.md"):
        if p.name != "README.md":
            assert "Synthetic test document" in p.read_text(encoding="utf-8"), p.name
