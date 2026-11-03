from __future__ import annotations

from collections.abc import Iterator

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.lc import llm
from app.models import ChatTurn, Source
from app.prompts import answer_prompt, as_chat_history
from app.retrieve import RetrievalOptions, retrieve


def answer(message: str, history: list[ChatTurn]) -> Iterator[str]:
    history_messages = as_chat_history(history)
    documents = retrieve(message, history_messages)
    prompt_value = answer_prompt(_documents_to_sources(documents)).invoke(
        {"input": message, "chat_history": history_messages}
    )
    for chunk in llm().stream(prompt_value.to_messages()):
        text = chunk.content if isinstance(chunk.content, str) else ""
        if text:
            yield text


def answer_once(
    message: str,
    options: RetrievalOptions | None = None,
    store: Chroma | None = None,
) -> tuple[str, list[Document]]:
    """One-shot answer for evaluation. Same retrieval and prompt as the chat, with no history."""
    documents = retrieve(message, [], options, store)
    prompt_value = answer_prompt(_documents_to_sources(documents)).invoke(
        {"input": message, "chat_history": []}
    )
    response = llm().invoke(prompt_value.to_messages())
    text = response.content if isinstance(response.content, str) else ""
    return text, documents


def _documents_to_sources(documents: list[Document]) -> list[Source]:
    sources: list[Source] = []
    for doc in documents:
        meta = doc.metadata or {}
        sources.append(
            Source(
                id=str(meta.get("id") or doc.id or ""),
                doc_id=str(meta.get("doc_id") or ""),
                title=str(meta.get("title") or ""),
                doc_type=str(meta.get("doc_type") or ""),
                date=str(meta.get("date") or ""),
                section=str(meta.get("section") or ""),
                score=round(float(meta.get("score") or 0.0), 4),
                text=doc.page_content,
            )
        )
    return sources
