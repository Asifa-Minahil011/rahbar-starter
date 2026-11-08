# ADR 0004: A pluggable LLM provider, and what data leaves the cluster

## Status
Accepted · 2026-11-08

## Context
RAHBAR needs a language model to turn retrieved passages into a cited answer. Models differ in where they run:
on a developer's laptop, inside the cluster, or at a hosted API. Our constraints: a student laptop (CPU only, 15 GB
RAM), minikube with 6 GB, no credit card, and a corpus that is public documents but questions that come from
users. We also needed the same code to run in CI with no network and no model.

## Decision
The model sits behind one interface (`api/app/providers/llm/base.py`: `generate(question, passages) -> Answer`),
chosen by `LLM_PROVIDER` in the ConfigMap. Implemented: `stub` (tests and CI), `ollama-host` (default: Ollama on the
laptop, reached at `host.minikube.internal:11434`), and `hosted` (an OpenAI-compatible free tier, key in a
Kubernetes Secret). Swapping a provider is one file plus one config value.

Embeddings are pinned separately (`EMBED_MODEL`), come from the same Ollama host, and are not swappable at runtime
(see ADR 0001 and `/readyz`).

Output is enforced by schema (`format=` JSON schema on Ollama, `response_format` on hosted) and then validated with
Pydantic; one retry, then a canned fallback (`api/app/services/answer.py`).

## Options considered
- **Ollama inside the cluster.** Needs at least 8 GB for minikube and a long startupProbe for model load. Rejected
  for memory; the interface still allows it as a bonus.
- **Hosted only.** Fast, but every question and the retrieved passages leave the cluster, and CI would need a key.
- **A single hard-wired provider.** Simplest, but untestable offline.

## Consequences
Easier: offline CI with the stub, and a faster hosted model for evaluation runs. Harder: three code paths to keep
consistent. Data governance: with `ollama-host` nothing leaves the laptop. With `hosted`, the user's question and the
retrieved passages are sent to a third party. That is acceptable only because the corpus is public university
documents; a corpus with private documents must use a local provider. The hosted key lives only in the
`rahbar-secret` Secret created from the command line, is never logged (`hosted.py` logs token counts only), and
is gitignored. Retrieved text is treated as data in the prompt (`providers/llm/prompt.py`), and model output is
never executed, evaluated, or used to build a query. Watch: generation on CPU takes 1 to 2 minutes per answer,
which is why evaluation has a retrieval-only mode.
