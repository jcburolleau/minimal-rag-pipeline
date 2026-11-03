"""Query expansion: several phrasings of one question, each searched on its own."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from app.prompts import query_expand_prompt


class ExpandedQueries(BaseModel):
    queries: list[str] = Field(
        description="Alternative search queries for the knowledge base, each worded differently"
    )


def query_expander(model: BaseChatModel, n: int = 3) -> Runnable:
    """Question string in, up to `n` distinct alternative queries out."""
    return (
        RunnableLambda(lambda question: {"question": question})
        | query_expand_prompt().partial(n=str(n))
        | model.with_structured_output(ExpandedQueries)
        | RunnableLambda(lambda result: unique_queries(result.queries)[:n])
    ).with_retry(stop_after_attempt=3).with_config(run_name="query_expander")


def unique_queries(queries: list[str]) -> list[str]:
    """Drop blanks and case-insensitive duplicates, keeping first-seen order."""
    seen: set[str] = set()
    kept: list[str] = []
    for query in queries:
        text = query.strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            kept.append(text)
    return kept
