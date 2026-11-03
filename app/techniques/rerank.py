"""LLM re-ranking: reorder retrieved chunks by relevance to the question, then keep the top n."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.prompts import rerank_messages


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


class LLMReranker(BaseDocumentCompressor):
    """Drop-in `BaseDocumentCompressor`, so it also fits LangChain's compression retrievers."""

    model: BaseChatModel
    top_n: int = 5

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if len(documents) <= 1:
            return list(documents)[: self.top_n]
        ranker = self.model.with_structured_output(RankOrder).with_retry(stop_after_attempt=3)
        messages = rerank_messages(query, [doc.page_content for doc in documents])
        result = ranker.invoke(messages, config={"callbacks": callbacks, "run_name": "llm_reranker"})
        order = complete_order(result.order, len(documents))
        ranked: list[Document] = []
        for position, index in enumerate(order[: self.top_n], start=1):
            doc = documents[index]
            ranked.append(
                Document(
                    id=doc.id,
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rerank_position": position},
                )
            )
        return ranked


def complete_order(order: list[int], count: int) -> list[int]:
    """1-based ids from the model to 0-based indexes.

    Out-of-range and repeated ids are dropped; ids the model left out keep
    their retrieval order at the end.
    """
    seen: set[int] = set()
    indexes: list[int] = []
    for chunk_id in order:
        index = chunk_id - 1
        if 0 <= index < count and index not in seen:
            seen.add(index)
            indexes.append(index)
    indexes.extend(index for index in range(count) if index not in seen)
    return indexes
