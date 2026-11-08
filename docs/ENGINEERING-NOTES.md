# Engineering notes


## 1. Indexing in API startup: what breaks with 3 replicas and one Chroma volume?

## 2. Readiness and liveness swapped, Chroma slow for 90 seconds: what happens?

## 3. What does a CPU-based HPA measure on a generation service, and what should scale it?

## 4. The indexer's peak memory, its limit, the margin, and the exit code when it is wrong

## 5. recall@5 by question type: where is it worst, and why?

## 6. EMBED_MODEL changed on the API only: what does your system do?

## 7. The failure that cost us more than an hour

**What happened.** I started the API and opened the chat page, but both `/healthz` and `/api/ask` returned `{"detail":"Not Found"}`, even though the indexer had just finished and the index was built.

**What I wrongly believed first.** I thought the API itself was broken, that the routes had not been registered, or that my retriever code had a bug. I spent about an hour checking the indexer output and the Chroma collection, which were all fine.

**What told me the truth.** The uvicorn terminal printed `[Errno 98] error while attempting to bind on address ('127.0.0.1', 8000): address already in use` right after "Application startup complete". Another process was already listening on port 8000, so my `curl` calls were reaching that process, which has no `/healthz` route, and not RAHBAR. I started the API on port 8010 instead and `{"status":"alive"}` came back immediately. The lesson: when a service answers with a 404 in a place you expect it to work, first check that your process really owns the port (`ss -ltnp | grep 8000`) and read the start-up log for bind errors before touching any code.
