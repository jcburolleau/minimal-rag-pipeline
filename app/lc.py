from __future__ import annotations

import re
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch
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


def collection_exists(
    chroma_client: chromadb.PersistentClient | None = None,
    collection_name: str | None = None,
) -> bool:
    return (collection_name or settings.collection_name) in listed_collections(chroma_client)


def row_count(collection_name: str | None = None) -> int:
    name = collection_name or settings.collection_name
    if not collection_exists(collection_name=name):
        return 0
    return int(client().get_collection(name).count())


def embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embed_model, api_key=api_key())


def llm(
    model: str | None = None,
    streaming: bool = True,
    temperature: float | None = None,
) -> ChatOpenAI:
    """Chat model. Technique calls (rewrite, expand, rerank, preprocess) use streaming=False, temperature=0."""
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=api_key(),
        streaming=streaming,
        temperature=temperature,
    )


def vectorstore(collection_name: str | None = None) -> Chroma:
    name = collection_name or settings.collection_name
    if not collection_exists(collection_name=name):
        raise FileNotFoundError(
            f"Index collection '{name}' is missing. Run ingest first."
        )
    return Chroma(
        collection_name=name,
        embedding_function=embeddings(),
        persist_directory=str(persist_path()),
        collection_metadata=COLLECTION_METADATA,
    )


def overwrite_documents(documents: list[Document], collection_name: str | None = None) -> int:
    """Drop the collection and write documents through the LangChain store."""
    name = collection_name or settings.collection_name
    path = persist_path()
    chroma_client = client(path)
    if collection_exists(chroma_client, name):
        chroma_client.delete_collection(name)

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
                collection_name=name,
                persist_directory=str(path),
                collection_metadata=COLLECTION_METADATA,
            )
        else:
            store.add_documents(batch, ids=ids)
        print(f"  {min(start + BATCH, len(documents))}/{len(documents)}")
    return len(documents)


def search(query: str, k: int, store: Chroma | None = None) -> list[Document]:
    """Top-k chunks with a 0–1 score. `query` is used as given."""
    pairs = (store or vectorstore()).similarity_search_with_score(query, k=k)
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
