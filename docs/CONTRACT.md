# RAGOps contract (Assignment 03)

Both members code against this file. Change it only through a PR that both approve.

## 1. Chunk metadata (Chroma, one record per chunk)

| Field | Type | Meaning |
|---|---|---|
| `doc_id` | str | slug of the logical document, e.g. `academic-rules-handbook` (the same for all versions of it) |
| `doc_version` | str | e.g. `2026-06` |
| `content_hash` | str | SHA-256 of the extracted document text (for change detection) |
| `status` | str | `active` or `archived` |
| `superseded` | bool | kept for the existing retriever filter; `true` for older versions |
| `source` | str | file name |
| `section` | str | heading the chunk sits under |
| `page` | int | 1-based page |
| `ingested_at` | str | UTC timestamp of the ingest run |

Chunk id: `{doc_id}-{doc_version}-{n:04d}`.
The retriever returns only `status == "active"` and `superseded == false`
(unless the question asks for an older version).
Collection metadata keeps: `embed_model`, `dimension`, `index_id`, `chunk_size`, `chunk_overlap`, `indexed_at`;
add `embed_version`.

## 2. Verification report (one JSON file per ingest run, in `/data/reports/`)

Per document: `doc_id`, and for each check (`quality`, `completeness`, `recency`, `uniqueness`)
`{"result": "pass|fail", "reason": "..."}`, plus `action` (`indexed|quarantined`).

## 3. Trace record (one JSON line per query)

`trace_id` (= API `request_id`), `ts`, `query`, `rewritten_query`, `embed_model`, `embed_version`,
`collection`, `index_id`, `retrieved` [{`chunk_id`, `doc_id`, `doc_version`, `score`}], `filters`,
`reranked_order`, `prompt_hash`, `prompt`, `llm_model`, `llm_params`, `response`, `citations`,
`guardrail_actions`, `latency_ms` {`embed`, `retrieve`, `generate`, `total`}, `cache` (`hit|miss`).
Written to stdout (so `kubectl logs` shows it) and appended to `$TRACE_DIR/traces.jsonl`.

## 4. Configuration keys (ConfigMap `rahbar-config`; env vars)

Existing: `EMBED_MODEL`, `CHROMA_COLLECTION`, `TOP_K`, `RELEVANCE_FLOOR`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `OLLAMA_URL`, `LLM_MODEL`.
New: `EMBED_VERSION`, `DATA_DIR` (`/data`: `corpus/`, `quarantine/`, `archive/`, `reports/`), `TRACE_DIR` (`/traces`),
`RELEASE_TAG`. Gate thresholds live in a separate ConfigMap `gate-config` (file `gate.yaml`).

## 5. Entry points and exit codes

| Command | Does | Exit |
|---|---|---|
| `python -m pipeline.run` | verify + incremental ingest (Job) | 0 ok, non-zero fail |
| `python eval/suite.py` | three-level suite, writes `eval/results/<timestamp>.json` | 0 |
| `python eval/gate.py` | applies `gate.yaml` to a result vs the baseline | 0 pass, 1 fail |
| `python scripts/trace_show.py <trace_id>` | prints context + prompt | 0 |
| `python scripts/trace_metrics.py --since 24h` | metrics from the trace log | 0 |
| `python eval/coverage.py` | live queries vs golden set | 0 ok, 1 below threshold |

## 6. Ownership

Asifa: `k8s/`, `store.py` (incremental, archive/restore), Batch B, PVCs, Jobs.
Manahil: `verify.py`, `eval/`, trace code in `api/`, coverage, golden set.
A change to a file you don't own goes through the owner's review.
