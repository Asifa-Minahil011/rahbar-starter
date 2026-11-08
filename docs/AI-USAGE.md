# AI usage


| Tool | What it wrote or shaped | What we changed afterwards, and why |
| --- | --- | --- |
| Claude (Anthropic) | First versions of the indexer stages (extract with OCR, clean, chunk, embed, store), the Chroma retriever, the Ollama and hosted providers, the answer service, the eval metric functions, the semantic cache, the tuning and experiment scripts, and their tests | I ran everything against the real corpus and pasted the output back. That showed clause text being dropped and table-of-contents text being indexed, so the chunker and cleaner were fixed. I checked the golden questions against the real chunks and reworded two (4.4 and 2.7), picked both version questions from real 2023 vs 2026 clauses, and resolved a merge conflict in chroma.py myself. |
| Claude (Anthropic) | Drafts of docs/CORPUS.md, PIPELINE.md, FAILURE-ANALYSIS.md and ADRs 0002 to 0004 | I replaced the placeholders with measured numbers and wrote the justifications and diagnoses in my own words. |
