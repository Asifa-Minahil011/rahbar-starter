"""POST /api/ask. The Ingress sends /api/* here unchanged, so the prefix stays."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import AskRequest, AskResponse
from ..services.answer import _refs, answer_question
from ..services.cache import get_cache

router = APIRouter(prefix="/api")


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    return answer_question(body.question, state.retriever, state.provider, state.settings,
                           cache=get_cache(state))


@router.post("/retrieve")
def retrieve(body: AskRequest, request: Request) -> dict[str, object]:
    """Retrieval only: no model call. Used by `eval/run_eval.py --retrieval-only` to measure
    recall@k in seconds instead of minutes. Same retriever, same filter, same top_k."""
    state = request.app.state
    passages = state.retriever.retrieve(body.question, k=state.settings.top_k)
    return {"top_k": state.settings.top_k, "retrieved": [r.model_dump() for r in _refs(passages)]}
