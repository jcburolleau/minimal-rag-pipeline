from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings
from app.openai_client import api_key

BATCH = 32
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def persist_path(index_path: Path | None = None) -> Path:
    path = index_path or settings.index_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def client(index_path: Path | None = None) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(persist_path(index_path)))


def listed_collections(chroma_client: chromadb.PersistentClient | None = None) -> list[str]:
    connection = chroma_client or client()
    return [str(collection.name) for collection in connection.list_collections()]


def collection_exists(chroma_client: chromadb.PersistentClient | None = None) -> bool:
    return settings.table_name in listed_collections(chroma_client)


def row_count() -> int:
    if not collection_exists():
        return 0
    return int(client().get_collection(settings.table_name).count())


def embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embed_model, api_key=api_key())


def llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=api_key(),
        streaming=True,
    )


def vectorstore() -> Chroma:
    if not collection_exists():
        raise FileNotFoundError(
            f"Index collection '{settings.table_name}' is missing. Run ingest first."
        )
    return Chroma(
        collection_name=settings.table_name,
        embedding_function=embeddings(),
        persist_directory=str(persist_path()),
        collection_metadata=COLLECTION_METADATA,
    )


def overwrite_documents(documents: list[Document]) -> int:
    """Drop the northline collection and write documents through the LangChain store."""
    path = persist_path()
    chroma_client = client(path)
    if collection_exists(chroma_client):
        chroma_client.delete_collection(settings.table_name)

    embed = embeddings()
    store: Chroma | None = None
    for start in range(0, len(documents), BATCH):
        batch = documents[start : start + BATCH]
        ids = [str(doc.id) for doc in batch]
        if store is None:
            store = Chroma.from_documents(
                batch,
                embedding=embed,
                ids=ids,
                collection_name=settings.table_name,
                persist_directory=str(path),
                collection_metadata=COLLECTION_METADATA,
            )
        else:
            store.add_documents(batch, ids=ids)
        print(f"  {min(start + BATCH, len(documents))}/{len(documents)}")
    return len(documents)


class ScoredChromaRetriever(BaseRetriever):
    """Retriever that maps Chroma distances to UI-friendly 0–1 scores."""

    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any | None = None,
    ) -> list[Document]:
        pairs = vectorstore().similarity_search_with_score(query, k=self.k)
        documents: list[Document] = []
        for doc, distance in pairs:
            metadata = dict(doc.metadata)
            metadata.pop("source", None)
            metadata["score"] = round(1.0 / (1.0 + float(distance)), 4)
            documents.append(
                Document(
                    id=str(metadata.get("id") or doc.id or ""),
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )
        return documents


_COMPANY_DEIXIS = re.compile(
    r"\b(this company|the company|the organisation|the organization|"
    r"this org(?:anisation|anization)?)\b",
    re.IGNORECASE,
)


def ground_query(query: str) -> str:
    """Bind deictic questions ('this company') to Northline before embedding."""
    if "northline" in query.lower() or not _COMPANY_DEIXIS.search(query):
        return query
    return f"Northline Sensors SA — {query}"


def as_retriever() -> Runnable:
    return RunnableLambda(ground_query) | ScoredChromaRetriever(k=settings.top_k)


def create_history_aware_retriever(
    model: Runnable,
    retriever: Runnable,
    prompt: ChatPromptTemplate,
) -> Runnable:
    """LCEL retriever that rewrites follow-ups against chat history.

    Same contract as LangChain's classic helper: no history means the raw
    input is retrieved; with history the prompt+LLM produce a standalone query.
    """
    if "input" not in prompt.input_variables:
        raise ValueError(
            f"Expected `input` in the rewrite prompt, got {prompt.input_variables}"
        )
    return RunnableBranch(
        (
            lambda x: not x.get("chat_history", False),
            (lambda x: x["input"]) | retriever,
        ),
        prompt | model | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")
