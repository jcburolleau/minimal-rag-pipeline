"""Single retrieval entry point for the chat and the evals.

Every advanced technique is opt-in. With all flags off this is the same
history-aware vector search the chat has always run:

    question, history
      → standalone query   (condense follow-ups, or rewrite every question)
      → expand?            (alternative phrasings)
      → ground_query + vector search per query
      → reciprocal rank fusion
      → rerank?            (LLM, then keep k)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.lc import (
    create_history_aware_retriever,
    ground_query,
    llm,
    search,
    vectorstore,
)
from app.prompts import rewrite_prompt
from app.techniques.expand import query_expander, unique_queries
from app.techniques.fusion import reciprocal_rank_fusion
from app.techniques.rerank import LLMReranker
from app.techniques.rewrite import query_rewriter

SEARCH_CONCURRENCY = 4


@dataclass(frozen=True)
class RetrievalOptions:
    """Which techniques run. Defaults come from .env so the chat and the evals agree."""

    rewrite: bool = field(default_factory=lambda: settings.rewrite_query)
    expand: bool = field(default_factory=lambda: settings.expand_query)
    rerank: bool = field(default_factory=lambda: settings.rerank)
    k: int = field(default_factory=lambda: settings.top_k)
    candidate_k: int = field(default_factory=lambda: settings.candidate_k)
    expand_n: int = field(default_factory=lambda: settings.expand_n)


def retrieve(
    question: str,
    history: list[BaseMessage] | None = None,
    options: RetrievalOptions | None = None,
    store: Chroma | None = None,
) -> list[Document]:
    """Top-k chunks for a question, fresh on every call."""
    options = options or RetrievalOptions()
    messages = history or []
    chroma = store or vectorstore()
    model = llm(streaming=False, temperature=0)
    search_from = RunnableLambda(
        lambda query: _search_from(question, query, bool(messages), options, chroma, model)
    )
    if options.rewrite:
        chain = query_rewriter(model) | search_from
    else:
        chain = create_history_aware_retriever(llm(), search_from, rewrite_prompt())
    return chain.invoke({"input": question, "chat_history": messages})


def _search_from(
    question: str,
    query: str,
    has_history: bool,
    options: RetrievalOptions,
    chroma: Chroma,
    model: BaseChatModel,
) -> list[Document]:
    """Everything after the standalone query.

    Without history the user's own question is searched too and is what the
    re-ranker scores against. With history it may lean on earlier turns
    ("and its lead time?"), so the standalone query takes its place.
    """
    standalone = query.strip() or question
    anchor = standalone if has_history else question
    queries = [anchor, standalone]
    if options.expand:
        queries += query_expander(model, options.expand_n).invoke(anchor)
    queries = unique_queries(queries)

    per_query = max(options.candidate_k, options.k) if options.rerank else options.k
    searcher = RunnableLambda(lambda text: search(ground_query(text), per_query, chroma))
    results = searcher.batch(queries, config={"max_concurrency": SEARCH_CONCURRENCY})
    fused = reciprocal_rank_fusion(results)
    if options.rerank:
        reranker = LLMReranker(model=model, top_n=options.k)
        return list(reranker.compress_documents(fused, anchor))
    return fused[: options.k]
