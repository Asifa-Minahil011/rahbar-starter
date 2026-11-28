# Batch B: documents with planted problems (assignment 03, Part A)

Twelve files, all synthetic (each says so). They go in the corpus folder next to Batch A
(the nine real documents downloaded by `corpus/fetch.py`) to test the verify stage (A1), the
incremental update (A2), the guardrail (D4) and data drift (E3).

| Planted problem | File(s) | Should be caught by |
| --- | --- | --- |
| exact duplicate x2 | `*-copy.md`, `*-final.md` | uniqueness (hash) |
| near-duplicate x1 | `library-borrowing-policy-2026-reworded.md` | uniqueness (embedding similarity) |
| truncated | `transport-policy-2026-truncated.md` | completeness |
| corrupt file | `attendance-notice-corrupt.pdf` | quality (extraction fails) |
| missing metadata | `research-ethics-guidelines.md` | completeness |
| newer version | `faculty-...-policy-2026-revision.md` | recency / A2 archives the old PDF |
| injected instruction | `library-winter-break-notice-injected.md` | not the verify stage: the guardrail (D4) |
| clean new documents x4 | library, scholarship, hostel, exam rules | nothing (must be indexed) |

`planted.json` is the answer key. Markdown files carry their metadata as front matter
(`source`, `title`, `date`, `doc_version`, `pages`, `doc_id`); the extractor strips it before chunking.

## Load it
Copy the files into the corpus folder the indexer reads (locally `corpus/raw/`, on the cluster
the `datalake` PVC at `/corpus`), then run the ingestion Job:

    kubectl run loader ...      # as in the Phase 0 loader pod
    kubectl cp corpus/batch_b/. rahbar/loader:/data/corpus/ -n rahbar
