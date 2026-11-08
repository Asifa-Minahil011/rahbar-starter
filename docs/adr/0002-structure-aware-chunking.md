# ADR 0002: Structure-aware chunking, tuned on the golden set

## Status
Accepted · 2026-11-08

## Context
The corpus mixes numbered rule clauses, long narrative handbooks, tables and OCR text. A fixed-size window cuts a
definition in half and can mix two clauses in one vector, so retrieval returns half an answer. The chunker is also
where exact terms (clause numbers) survive or die.

## Decision
`indexer/pipeline/chunk.py` splits by section first (a heading or numbered clause starts a new section), then packs
paragraphs up to a size in words, splits over-long paragraphs on sentences, and carries an overlap between
neighbouring chunks of the same section. Each chunk starts with its section label (clause number plus its first
words), which is also the `section` metadata and what a citation quotes. The size and overlap are not guessed:
`scripts/tune_chunking.py` measures recall@3 and recall@5 on the golden set for three configurations, and the
shipped values and the table are in docs/PIPELINE.md.

## Options considered
- **Fixed-size windows.** Simplest, but cuts definitions and ignores section boundaries. It is the stage B failure
  we break on purpose in FAILURE-ANALYSIS.md.
- **One chunk per page.** Pages cut clauses in half and mix sections.
- **Token-based sizes.** More exact for the model, but needs a tokenizer in the image; words are within about
  1.3x of tokens for English.

## Consequences
Citations point at a clause, not a page, and golden questions name clauses, so the golden set stays valid when
the size changes. Chunks in the rules documents are short (each clause is its own section), so size matters mostly
for narrative documents. Known cost: tables and contents fragments can land in the wrong section (see
PIPELINE.md, "what could not be recovered"); watch recall on table questions.
