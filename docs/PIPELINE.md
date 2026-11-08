# Indexing and retrieval pipeline

Indexing runs once, and again when the material changes (`indexer/`, a Job). Querying runs on every question
(`api/`, reads only).

## Extraction
`indexer/pipeline/extract.py`. PDFs go through pdfplumber page by page. A page with fewer than 25 characters of
text is rasterised at 300 dpi and OCR'd with Tesseract; those pages carry `ocr=True` and are counted in the
`extracted` log line.

OCR'd pages (indexing run log): start-ups policy 8 of 8 pages (a pure scan, `pdffonts` lists no fonts), 2023 rules
24 of 26, rules handbook 2026 5 of 58, student handbook 4 of 64.

Tables are cut out of the running text and re-rendered one line per row, with the column header repeated in every
cell, because a plain dump puts the headings far from the numbers and an embedding cannot connect them. A real
stored chunk from `fee-structure-page.html` shows the result: `| Program: BBA / BS | Fee: Rs. 12,000 |`, and the
refund table is stored as `| Timeline: Upto 10th day of commencement of classes | Percentage of Fee: 100% ...`.

Before and after for a table that did not work well: the grade-point table of the 2026 rules handbook (PDF page
12). Plain `pdfplumber` text of the page:

    Grade Points Grade Points
    A+ 4.00 C+ 2.33
    A 4.00 C 2.00
    A- 3.67 C- 1.67
    B+ 3.33 D+ 1.33
    B 3.00 D 1.00

What we stored (chunk of section "2.7 In CGPA calculation for the courses that", page 12):

    2.7 In CGPA calculation for the courses that 2.7 In CGPA calculation for the courses that have been repeated, | Grade | Points | Grade | Points | | A+ | 4.00 | C+ | 2.33 | | A | 4.00 | C | 2.00 | | A- | 3.67 | C- | 1.67 | | B+ | 3.33 | D+ | 1.33 | | B | 3.00 | D | 1.00 | | B- | 2.67 | F | 0.00 | | 6 | Undergraduate Academic Rules and Regulations 2026 | Grade Points only the most recent grade points are used, even if they are less than the earlier attempts.

Two problems are visible. The table has two column pairs with the same header names (Grade, Points, Grade, Points),
so the header cannot be repeated per cell and the renderer falls back to plain pipe-separated cells. And tables are
appended at the end of their page's text, so this table is stored under the last section on the page ("2.7 In CGPA calculation for the courses that")
and not under clause 2.3 where it belongs.

What could not be recovered, found by inspecting the real index:
- 43 of 769 chunks still carry a page-footer fragment (for example `| 22 | Undergraduate Academic Rules and
  Regulations 2026 |`), because the footer is rendered as a table row and its page number changes.
- The first chapter of the 2026 handbook has a duplicated text layer: heading lines repeat up to four times in
  one chunk (clauses 1.1, 1.17, 1.18).
- The handbook's two-column contents page leaks into clauses: clause 1.6's chunk contains `Use in Research 45
  ... Detecting Generative AI Use in Assessments 45`.
- The 2023 rules are 24 of 26 pages of OCR text. Its table of contents came out as garbage (`IMtrOdUCtION....`), so
  dot-leader lines are removed; OCR also misreads some digits, so any 2023 value used in a golden question was
  checked in the PDF.
- Text inside images and charts is not extracted.

## Cleaning
`indexer/pipeline/clean.py`. Rules applied: drop lines that repeat at the top or bottom of at least 40% of the
pages (running headers and footers, page counters), drop lone page numbers, drop table-of-contents lines (runs of
dots), re-join lines the PDF wrapped mid-sentence and undo hyphenation, drop OCR single-character junk.

Deliberately not stripped: clause numbers (`1.21`), heading lines, and table-row lines. Exact-term questions and
citations depend on them.

## Chunking
`indexer/pipeline/chunk.py`. Structure-aware: a heading or numbered clause starts a new section, paragraphs are
packed up to the size, an over-long paragraph is split on sentences, and each chunk starts with the last
`overlap` words of its predecessor in the same section. The size unit is words (whitespace-separated), not model
tokens, so it needs no tokenizer; for English that is roughly 1.3 tokens per word. Each chunk text begins with its
section label.

Section label: the clause number plus its first eight words (`1.21 Students are required to maintain 100% attendance`).
In the rules documents each clause is its own section, so chunks are short there (the 2023 edition averages about 60 words per chunk, the 2026 handbook about 77), and the size limit only matters for long narrative documents
(annual reports, student handbook).

Shipped: size 400, overlap 50 (words). Final indexing run: 769 chunks from 86,658 original words; stored words 97,775; storage amplification 1.128 (stored / original). 43 of the chunks still carry a page-footer fragment. Amplification rose from 1.078 to 1.128 after the chunker kept the full clause line in each chunk and prefixed each chunk with its section label.

Three configurations measured on the golden set (`python scripts/tune_chunking.py`), recall@5 against the real
retriever (`eval/tuning.json`):

TUNING-TABLE-GOES-HERE

Why the shipped configuration: WRITE-YOUR-OWN-JUSTIFICATION from the table (the rules documents are insensitive to
size because their clauses are short; say which config wins on the narrative documents and why you accepted its
cost in chunks stored and amplification).

## Embedding
Model `nomic-embed-text` via Ollama (`/api/embed`, batches of 16), vector dimension 768. Documents are embedded
with the prefix `search_document: ` and questions with `search_query: `; nomic trains the two as a pair. The pin
is `EMBED_MODEL` in the ConfigMap, read by both the indexer and the API. The indexer writes `embed_model`,
`dimension` and a random `index_id` into the Chroma collection metadata (`indexer/pipeline/store.py`); the API's
`/readyz` compares `embed_model` with its own and refuses to become ready on a mismatch (`api/app/routes/health.py`).
The new index is built in a scratch collection and renamed over the old one only after its count is verified.

## Semantic cache
FILL IN with Asifa's cache: thresholds tried, the false hit that ruled one out, and the one shipped. The cache keys
must include the collection's `index_id` so a re-index invalidates every cached answer.

## Evaluation method
`eval/run_eval.py`. A retrieved passage matches a golden one when the source file is equal and the golden section
is a prefix of the stored section on a word or dot boundary (`5.4` matches `5.4 Academic Warning` and `5.4.1`, not
`5.40`). Matching by section rather than chunk id keeps the golden set valid when the chunk size changes.
recall@k = relevant passages found in the top k / relevant passages, per question, averaged, and reported overall
and by question type. Citation validity = citations that match a passage retrieved for that request / all
citations. Abstention correctness = must-abstain questions answered `answered_from_context: false` / all of them.
`--retrieval-only` skips the model and measures recall only.
