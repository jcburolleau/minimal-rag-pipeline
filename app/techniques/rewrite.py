"""Query rewriting: turn the user's question into one short, specific knowledge-base query."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from app.prompts import query_rewrite_prompt


def query_rewriter(model: BaseChatModel) -> Runnable:
    """`{"input", "chat_history"}` in, one search query out.

    Unlike the history-only condense step, this runs on every question.
    """
    return (
        query_rewrite_prompt() | model | StrOutputParser() | RunnableLambda(first_line)
    ).with_retry(stop_after_attempt=3).with_config(run_name="query_rewriter")


def first_line(text: str) -> str:
    """Models sometimes add a label, quotes, or a second line; keep only the query."""
    for line in text.strip().splitlines():
        line = line.strip().strip('"').strip("'").strip()
        if line:
            return line
    return ""
