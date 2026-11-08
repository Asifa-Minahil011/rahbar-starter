# Failure analysis

We broke four of the six stages on purpose, one at a time, restored each, and recorded what the measurements showed (`scripts/failure_experiments.py`, `scripts/tune_chunking.py`; raw numbers in `eval/failure-*.json` and `eval/tuning-b.json`). Only stage F is the model's fault, so A to E are checked first.

Baseline, nothing broken: recall@5 92.9% (by type: exact 50.0%, injection 100.0%, ordinary 100.0%, version 100.0%).

## Stage B · Chunking
- **What we broke:** fixed 100-word chunks with no overlap (`tune_chunking.py --configs 100:0`), indexed into a scratch collection; the real collection was untouched.
- **Symptom (measured):** recall@5 92.9% and recall@3 92.9% with 1383 chunks, against 92.9% recall@5 for the shipped 400:50 (769 chunks). By type: exact 50.0%, injection 100.0%, ordinary 100.0%, version 100.0% vs exact 50.0%, injection 100.0%, ordinary 100.0%, version 100.0%.
- **Diagnosis, as if we had not caused it:** recall@5 by question type in `eval/results.json` drops for questions whose answer spans several sentences, and the retrieved chunk text in the `answered` log line ends mid-sentence. Chunk count and storage amplification in the indexer's `chunking summary` line jump when the size shrinks.
- **Remedy:** structure-aware chunking with overlap, size and overlap chosen from the tuning table (ADR 0002).

## Stage C · Retrieval
- **What we broke:** `TOP_K=1`.
- **Symptom (measured):** recall@5 fell from 92.9% to 85.7%; by type exact 50.0%, injection 0.0%, ordinary 100.0%, version 100.0%. A question with two relevant clauses can score at most one half.
- **Diagnosis, as if we had not caused it:** the `answered` log line shows `top_k: 1` and a single `chunk_ids` entry; the retrieval-only eval's recall by type is the number that moved.
- **Remedy:** `TOP_K=5` from the ConfigMap.

## Stage E · Source
- **What we broke:** `SUPERSEDED_FILTER=off`, so superseded editions are retrieved.
- **Symptom (measured):** recall@5 92.9% (baseline 92.9%); version questions retrieved from {"q15": ["academic-rules-handbook-jun-2026.pdf", "prospectus-2026-academic-rules-and-regulations.pdf", "undergraduate-academic-rules-aug-2023.pdf"], "q16": ["academic-rules-handbook-jun-2026.pdf", "undergraduate-academic-rules-aug-2023.pdf"]} with the filter off, against {"q15": ["academic-rules-handbook-jun-2026.pdf", "prospectus-2026-academic-rules-and-regulations.pdf"], "q16": ["academic-rules-handbook-jun-2026.pdf"]} normally. Both editions share clause numbers, so the 2023 edition competes for the same questions.
- **Diagnosis, as if we had not caused it:** the retrieved `source` and `doc_version` of the version questions in the log line show `undergraduate-academic-rules-aug-2023.pdf` (`superseded: true`) in the top k.
- **Remedy:** metadata filter on `superseded: false` unless the question asks about an older edition (`api/app/retrieval/chroma.py`); a prompt cannot fix this.

## Stage F · Generation
- **What we broke:** `TEMPERATURE=1.2`.
- **Symptom (measured):** citation validity 100.0% (baseline 100.0%), abstention correctness 100.0% (baseline 100.0%). Counters in this run: retries 3, fallbacks 1, invalid citations dropped 13. Retrieval recall is unchanged by temperature, which is how this stage is told apart from A to E.
- **Diagnosis, as if we had not caused it:** retrieval-only recall is normal while the generation counters (`rahbar_generation_retries_total`, `rahbar_fallbacks_total`, `rahbar_invalid_citations_total` in `/metrics`) and citation validity move.
- **Remedy:** `TEMPERATURE=0.2` from the ConfigMap, schema-constrained output, one retry and a safe fallback.
