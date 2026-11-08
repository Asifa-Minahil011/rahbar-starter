# Corpus

Source: FAST-NUCES public website (nu.edu.pk) · Documents: 9 · Pages: 281 (8 PDFs, 2 HTML pages)

Public documents only. We commit `corpus/MANIFEST.md` (URL and SHA-256 of each file) and the fetch script
`corpus/fetch.py`, not the files themselves (`make fetch` downloads them into `corpus/raw/`, which is gitignored).

| File | doc_version | superseded | Pages | OCR'd pages |
| --- | --- | --- | --- | --- |
| undergraduate-academic-rules-aug-2023.pdf | 2023-08 | **true** | 26 | 24 |
| academic-rules-handbook-jun-2026.pdf | 2026-06 | false | 58 | 5 |
| prospectus-2026-academic-rules-and-regulations.pdf | 2026 | false | 5 | 0 |
| student-handbook-jun-2026.pdf | 2026-06 | false | 64 | 4 |
| annual-report-2019-20.pdf | 2019-20 | false | 33 | 0 |
| annual-report-2023-24.pdf | 2023-24 | false | 85 | 0 |
| faculty-entrepreneurship-and-technical-start-ups-policy.pdf | 2022-05 | false | 8 | 8 |
| fee-structure-page.html | 2026-27 | false | 1 | 0 |
| grading-page.html | 2026 | false | 1 | 0 |

(Page and OCR counts are from the indexer's `extracted` log lines.)

| # | Required property | Document(s) that provide it | Evidence or notes |
| --- | --- | --- | --- |
| 1 | At least 6 documents, 40 pages | All nine files | 9 documents, 281 pages in total. |
| 2 | Two versions of one policy | `undergraduate-academic-rules-aug-2023.pdf` (old) and `academic-rules-handbook-jun-2026.pdf` (current) | The 2023 edition is indexed with `superseded: true` and `doc_version: 2023-08`; the retriever filters on `superseded: false` unless the question asks about an older edition (`api/app/retrieval/chroma.py`, `wants_older_version`). The two editions share clause numbers (clause 1.21 is the attendance rule in both), so without the filter the retriever returns the wrong year. The prospectus carries the same rules with different clause numbers. |
| 3 | Scanned, image-only PDF | `faculty-entrepreneurship-and-technical-start-ups-policy.pdf` (8 of 8 pages OCR'd); `undergraduate-academic-rules-aug-2023.pdf` (24 of 26 pages OCR'd) | Start-ups policy: `pdffonts` lists no fonts at all, so the file has no text layer and every page was OCR'd with Tesseract. 2023 rules: `pdffonts` lists only three fonts (Calibri, Calibri-Bold, a display font), which belong to the 2 pages with real text; the other 24 pages are images. OCR runs at 300 dpi (`indexer/pipeline/extract.py`). |
| 4 | A table with real information | The two annual reports (faculty, graduates, publications by campus); `fee-structure-page.html` (tuition per credit hour, refund timeline); `grading-page.html` (grade points) | Tables are re-rendered one row per line with the column header repeated in each cell (`render_table`), see PIPELINE.md for a before and after. VERIFY: confirm with `find_clause.py "Campus"` that rows survive. |
| 5 | Long definitions | `student-handbook-jun-2026.pdf` (64 pages, including the Generative AI policy) | Chunking is by section, so a definition is not cut in half by a fixed-size window. |
| 6 | A deliberately absent topic | Hostel rules | We left the prospectus's Transport and Hostel Facilities PDF out. A student plausibly asks how to get a hostel room. The index contains only passing mentions (one sentence in a lab-facilities rule, and the annual reports' objectives), found by scanning every chunk for "hostel". The golden set's two abstain questions are about hostels. |
| 7 | Exact-term content | The 2026 rules handbook: clause numbers (1.21 attendance, 4.24 withdrawal, 6.6 repeat limit), the CGPA thresholds, the fee page | Two golden questions name a clause number; the retriever gives exact clause or code matches a rank bonus (`HYBRID_BOOST`). |
