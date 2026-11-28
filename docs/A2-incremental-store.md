# A2: incremental update, archive and restore

Code: `indexer/pipeline/incremental.py` (sync), `manage.py` (operator commands), `run.py` (entry point).
Tests: `indexer/tests/test_incremental.py`. Cluster helper: `scripts/manage.sh`.

## How a run decides what to do
| Situation | Action | Chunks embedded |
|---|---|---|
| file unchanged (same SHA-256) | skipped, not even read | 0 |
| new file | extract, chunk, embed, add | its chunks |
| changed file | embed new chunks, upsert, delete only its stale ids | its chunks |
| file removed from the corpus | delete every chunk with that `source` | 0 |
| older version of a `doc_id` | `status=archived`, `superseded=true` (metadata only) | 0 |
| operator ran archive/restore/rollback | `pinned=true`: ingestion leaves the status alone | 0 |

A full rebuild (all chunks re-embedded, scratch collection and swap) happens only on the first run,
with `--full`, when `EMBED_MODEL`/`CHUNK_SIZE`/`CHUNK_OVERLAP` differ from the live index, or when the
index was built before A2 (no `content_hash`): so the first run after upgrading to this code is one
more full rebuild; the second run embeds 0.

## Versions
Two files with the same `doc_id` are versions of one document. Set `doc_id` in `_index.json`
(`{"file": ..., "doc_version": ..., "doc_id": ...}`) or in Markdown front matter. Without it,
`doc_id` is the file name without its extension, so every file is its own document.
Newest `doc_version` (string order) among files not marked `superseded` is active; the rest are archived.

## Demo commands (cluster)
    kubectl delete job indexer -n rahbar --ignore-not-found && kubectl apply -k k8s/overlays/dev
    kubectl logs job/indexer -n rahbar | grep "ingest summary"     # chunks_embedded: 0 on a rerun
    scripts/manage.sh counts                                       # chunks per doc_id, status, pin
    scripts/manage.sh counts --doc-id <deleted-doc>                # after deleting it: chunks 0
    scripts/manage.sh rollback <doc_id> --to <older doc_version>   # restore the old version
    scripts/manage.sh unpin <doc_id>                               # back to the automatic rule
