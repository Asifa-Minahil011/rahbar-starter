# ADR 0003: The semantic cache lives in Redis, keyed by the index version

## Status
Accepted · 2026-11-08

## Context
Repeated questions should not pay for retrieval and generation. A cached answer is served to everyone, so a wrong
entry, or an entry from an old index, is worse than no cache. The assignment allows Redis or a second Chroma
collection, and says to defend the choice.

## Decision
Redis (`api/app/services/cache.py`). An entry stores the question's unit vector and the validated response as
JSON, with a TTL. A lookup embeds the new question and returns the best entry of the current index if its cosine
similarity is at or above `CACHE_THRESHOLD` (ConfigMap). Every key contains the collection's `index_id`, which the
indexer writes into the collection metadata on each build, so a re-index invalidates every cached answer without
the indexer touching Redis (the NetworkPolicy allows Redis from the API only). Only grounded answers with
citations are stored; abstentions, fallbacks and anything that failed validation are never cached; any Redis
error is treated as a miss. The threshold was chosen with `scripts/cache_threshold.py` on paraphrases and
near-miss pairs (docs/PIPELINE.md, "Semantic cache").

## Options considered
- **A second Chroma collection.** Reuses an existing store, but Chroma is the system's single-writer vector
  store (written only by the indexer); letting the API write to it breaks that rule and its NetworkPolicy.
- **Exact-match cache.** Safe but almost never hits on natural-language questions.

## Consequences
Cheap hits, and automatic invalidation on re-index. The cache scans up to 1000 entries per lookup, which is fine
for a small cache and would need a vector index at scale. A threshold that is too low serves the wrong clause
(clause 4.23 versus 4.24 is the near miss we test), so it is set from measurements, not by feel.
