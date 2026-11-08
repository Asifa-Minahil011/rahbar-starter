# Failure analysis

We broke four of the six stages on purpose, one at a time, restored each, and recorded what we saw. Every
experiment uses an environment variable, so no code changed. Run each against the real index, then restore.
Commands assume the API is started with the usual variables (see README) on port 8010.

How to read the numbers: `python eval/run_eval.py --api http://localhost:8010 --retrieval-only` gives recall@5 in
seconds; a full run adds citation validity and abstention correctness; the API's `/metrics` shows
`rahbar_fallbacks_total`, `rahbar_invalid_citations_total`, `rahbar_generation_retries_total` and
`rahbar_abstentions_total`; the `answered` log line shows `chunk_ids` and `scores` for every request.

Baseline before breaking anything: recall@5 = RECORD, citation validity = RECORD, abstention = RECORD.

## Stage B · Chunking
- **What we broke:** index into a scratch collection with fixed 100-word chunks and no overlap:
  `CHUNK_SIZE=100 CHUNK_OVERLAP=0 CHROMA_COLLECTION=rahbar_b python -m pipeline.run`, then start the API with
  `CHROMA_COLLECTION=rahbar_b`. (The real `rahbar` collection is untouched.)
- **Symptom:** RECORD recall@5 and which question types dropped (expect: long, multi-sentence definitions).
- **Diagnosis, as if we had not caused it:** RECORD the log line, metric or command that shows it; for example
  recall@5 by type in `eval/results.json`, and the retrieved chunk text ending mid-sentence.
- **Remedy:** structure-aware chunking with overlap (shipped config); restore `CHROMA_COLLECTION=rahbar`.

## Stage C · Retrieval
- **What we broke:** `TOP_K=1`.
- **Symptom:** RECORD (expect: recall@5 collapses to recall@1; answers lose the second relevant clause).
- **Diagnosis, as if we had not caused it:** RECORD (the `top_k` and `chunk_ids` fields of the `answered` log line).
- **Remedy:** restore `TOP_K=5`.

## Stage E · Source
- **What we broke:** `SUPERSEDED_FILTER=off`, then ask a version-sensitive golden question.
- **Symptom:** RECORD the source of the retrieved chunk (the 2023 edition) and what the answer said.
- **Diagnosis, as if we had not caused it:** RECORD (the retrieved `source` and `doc_version` in the log line; recall
  on the `version` questions in `eval/results.json`).
- **Remedy:** metadata filter on `superseded: false` unless the question asks about an older edition; restore
  `SUPERSEDED_FILTER=on`.

## Stage F · Generation
- **What we broke:** `TEMPERATURE=1.2`.
- **Symptom:** RECORD (expect: more validation retries and fallbacks, more invalid citations).
- **Diagnosis, as if we had not caused it:** RECORD the metric deltas of `rahbar_generation_retries_total`,
  `rahbar_fallbacks_total` and `rahbar_invalid_citations_total`, and citation validity in `eval/results.json`.
- **Remedy:** `TEMPERATURE=0.2`. Reminder: only F is the model's fault, so check A to E first.
