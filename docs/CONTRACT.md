# RAGOps contract (Assignment 03)

Both members code against this file. Change it only through a PR that both approve.

## 1. Chunk metadata (Chroma, one record per chunk)

| Field | Type | Meaning |
|---|---|---|
| `source` | str | file name; the unit of change for ingestion (update, delete, archive) |
| `doc_id` | str | logical document. Files with the same `doc_id` are versions of one document. Default: the file name without its extension |
| `doc_version` | str | e.g. `2026-06`; newest in string order is the active one |
| `content_hash` | str | SHA-256 of the **file bytes** (change detection) |
| `status` | str | `active` or `archived` |
| `superseded` | bool | `true` whenever `status == "archived"`; the API's retriever filters on it |
| `pinned` | bool | operator override: ingestion does not change `status` until `unpin` |
| `section` | str | heading the chunk sits under |
| `page` | int | 1-based page |
| `ingested_at` | str | UTC time of the run that wrote the chunk |

Chunk id: `{file-stem-slug}-{n:04d}` (unchanged; unique because the stem is per file).
The retriever returns only `superseded == false` unless the question asks for an older version.
Collection metadata: `embed_model`, `dimension`, `index_id` (new on every change; the semantic cache
keys on it), `chunk_count`, `chunk_size`, `chunk_overlap`, `indexed_at`; add `embed_version` (Part C/F).

## 2. Ingestion run (Job `indexer`, CronJob `reindex`)

* `python -m pipeline.run` is incremental: unchanged file = skipped (0 embedded); new = added;
  changed = its chunks replaced; removed from the corpus = its chunks deleted; older version of a
  `doc_id` = archived (metadata only).
* A full rebuild (`--full`, first run, other model/chunking, or pre-A2 index) re-embeds everything.
* Each run logs one `ingest summary` JSON line (counts, `orphan_sources`, `chunks_by_doc_id`) and,
  if `REPORT_DIR` is set, writes `ingest-<timestamp>.json` there.
* Archive / restore / rollback / counts: `python -m pipeline.manage ...` via `scripts/manage.sh`
  (the only writers to Chroma are indexer pods).

## 3. Verification report (A1; one JSON file per run in `$DATA_DIR/reports/`)

Per document: `source`, `doc_id`, and for each check (`quality`, `completeness`, `recency`,
`uniqueness`) `{"result": "pass|fail", "reason": "..."}`, plus `action` (`indexed|quarantined`).
Quarantined files are moved to `$DATA_DIR/quarantine/` and are not indexed.

## 4. Trace record (one JSON line per query)

`trace_id` (= API `request_id`), `ts`, `query`, `rewritten_query`, `embed_model`, `embed_version`,
`collection`, `index_id`, `retrieved` [{`chunk_id`, `doc_id`, `doc_version`, `score`}], `filters`,
`reranked_order`, `prompt_hash`, `prompt`, `llm_model`, `llm_params`, `response`, `citations`,
`guardrail_actions`, `latency_ms` {`embed`, `retrieve`, `generate`, `total`}, `cache` (`hit|miss`).
Written to stdout (so `kubectl logs` shows it) and appended to `$TRACE_DIR/traces.jsonl`.

## 5. Configuration keys (ConfigMap `rahbar-config`; env vars)

Existing: `EMBED_MODEL`, `CHROMA_COLLECTION`, `TOP_K`, `RELEVANCE_FLOOR`, `CHUNK_SIZE`,
`CHUNK_OVERLAP`, `OLLAMA_URL`, `LLM_MODEL`, `CORPUS_DIR`.
New: `EMBED_VERSION`, `REPORT_DIR`, `DATA_DIR` (`/data`: `corpus/`, `quarantine/`, `archive/`,
`reports/`), `TRACE_DIR` (`/traces`), `RELEASE_TAG`.
Gate thresholds live in a separate ConfigMap `gate-config` (file `gate.yaml`).

## 6. Entry points and exit codes

| Command | Does | Exit |
|---|---|---|
| `python -m pipeline.run` | verify + incremental ingest (Job) | 0 ok, non-zero fail |
| `scripts/manage.sh counts \| archive \| restore \| rollback \| unpin` | operate on the live index | 0 ok |
| `python eval/suite.py` | three-level suite, writes `eval/results/<timestamp>.json` | 0 |
| `python eval/gate.py` | applies `gate.yaml` to a result vs the baseline | 0 pass, 1 fail |
| `python scripts/trace_show.py <trace_id>` | prints context + prompt | 0 |
| `python scripts/trace_metrics.py --since 24h` | metrics from the trace log | 0 |
| `python eval/coverage.py` | live queries vs golden set | 0 ok, 1 below threshold |

## 7. Ownership

Asifa: `k8s/`, `indexer/pipeline/incremental.py`, `manage.py`, `store.py`, Batch B, PVCs, Jobs.
Manahil: `verify.py`, `eval/`, trace code in `api/`, coverage, golden set.
A change to a file you don't own goes through the owner's review.
