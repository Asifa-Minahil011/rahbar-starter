# Engineering notes


## 1. Indexing in API startup: what breaks with 3 replicas and one Chroma volume?

## 2. Readiness and liveness swapped, Chroma slow for 90 seconds: what happens?

## 3. What does a CPU-based HPA measure on a generation service, and what should scale it?

## 4. The indexer's peak memory, its limit, the margin, and the exit code when it is wrong

## 5. recall@5 by question type: where is it worst, and why?

Measured on the golden set (`eval/results.json`): overall recall@5 92.9%.

| Question type | Questions | recall@3 | recall@5 |
| --- | --- | --- | --- |
| exact | 2 | 50.0% | 50.0% |
| injection | 1 | 100.0% | 100.0% |
| ordinary | 9 | 100.0% | 100.0% |
| version | 2 | 100.0% | 100.0% |

It is worst for **exact** questions (50.0%, 2 questions) and best for injection (100.0%). Embeddings struggle with exact terms because a clause number or a code like `4.23` is a rare token whose vector sits almost on top of `4.24`: the embedding captures what a passage is about, not which exact number it carries, so neighbouring clauses are near-duplicates in vector space. The technique from the RAG lecture that addresses it is hybrid retrieval (exact matching next to vectors). We implemented a small version: `exact_terms` (`api/app/retrieval/chroma.py:29`) pulls clause numbers and course codes out of the question, and candidates that contain them get a rank bonus (`HYBRID_BOOST`, `chroma.py:86`). In the tuning run the exact-term boost changed recall@5 from 92.9% to 92.9%; exact questions alone: 50.0% without it, 50.0% with it.

## 6. EMBED_MODEL changed on the API only: what does your system do?

## 7. The failure that cost us more than an hour

**What happened.** I started the API and opened the chat page, but both `/healthz` and `/api/ask` returned `{"detail":"Not Found"}`, even though the indexer had just finished and the index was built.

**What I wrongly believed first.** I thought the API itself was broken, that the routes had not been registered, or that my retriever code had a bug. I spent about an hour checking the indexer output and the Chroma collection, which were all fine.

**What told me the truth.** The uvicorn terminal printed `[Errno 98] error while attempting to bind on address ('127.0.0.1', 8000): address already in use` right after "Application startup complete". Another process was already listening on port 8000, so my `curl` calls were reaching that process, which has no `/healthz` route, and not RAHBAR. I started the API on port 8010 instead and `{"status":"alive"}` came back immediately. The lesson: when a service answers with a 404 in a place you expect it to work, first check that your process really owns the port (`ss -ltnp | grep 8000`) and read the start-up log for bind errors before touching any code.
