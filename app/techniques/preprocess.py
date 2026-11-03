"""Document pre-processing: an LLM splits each document into headline + summary + original text chunks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.documents import with_ids_and_prefix
from app.prompts import PREPROCESS_HUMAN, PREPROCESS_SYSTEM, preprocess_prompt

# Corpus documents are 0.5–1.5k characters, roughly one markdown section per
# few hundred, so this asks for a handful of chunks per document.
AVERAGE_CHUNK_CHARS = 250
MAX_CONCURRENCY = 3


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )


class Chunks(BaseModel):
    chunks: list[Chunk]


class LLMChunker:
    """Splitter with the same `split_documents` contract as LangChain text splitters.

    Output chunks keep the markdown chunker's ids (`{doc_id}::{i}`), metadata and
    title prefix, so retrieval, evals and the answer prompt treat both alike.
    LLM output is cached per document under `cache_dir`, keyed on the model,
    the prompt and the document body, so re-embedding does not re-call the LLM.
    """

    def __init__(
        self,
        model: BaseChatModel,
        model_name: str,
        cache_dir: Path | None = None,
        max_concurrency: int = MAX_CONCURRENCY,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_concurrency = max_concurrency

    def split_documents(self, documents: list[Document]) -> list[Document]:
        parsed = {doc_id: chunks for doc_id, chunks in self._cached(documents)}
        pending = [doc for doc in documents if _doc_id(doc) not in parsed]
        if pending:
            print(f"Pre-processing {len(pending)} documents with {self.model_name} ({len(parsed)} cached)…")
            parsed.update(self._call_llm(pending))
        else:
            print(f"All {len(documents)} documents pre-processed from cache.")

        chunks: list[Document] = []
        for doc in documents:
            for index, chunk in enumerate(parsed[_doc_id(doc)]):
                chunks.append(_as_document(doc, chunk, index))
        return chunks

    def _call_llm(self, documents: list[Document]) -> dict[str, list[Chunk]]:
        chunker = (
            preprocess_prompt()
            | self.model.with_structured_output(Chunks)
        ).with_retry(stop_after_attempt=4)
        inputs = [_prompt_inputs(doc) for doc in documents]
        parsed: dict[str, list[Chunk]] = {}
        failures: list[str] = []
        for index, result in chunker.batch_as_completed(
            inputs,
            config={"max_concurrency": self.max_concurrency, "run_name": "llm_chunker"},
            return_exceptions=True,
        ):
            doc = documents[index]
            doc_id = _doc_id(doc)
            if isinstance(result, Exception) or not result.chunks:
                failures.append(f"{doc_id}: {result if isinstance(result, Exception) else 'no chunks'}")
                continue
            parsed[doc_id] = result.chunks
            self._save(doc, result.chunks)
            print(f"  {len(parsed)}/{len(documents)} {doc_id}: {len(result.chunks)} chunks")
        if failures:
            failed = "; ".join(failures)
            raise RuntimeError(
                f"Pre-processing failed for {failed}. "
                "Finished documents are cached; re-run ingest to retry the rest."
            )
        return parsed

    def _cache_key(self, doc: Document) -> str:
        digest = hashlib.sha256()
        for part in (self.model_name, PREPROCESS_SYSTEM, PREPROCESS_HUMAN, doc.page_content):
            digest.update(part.encode("utf-8") + b"\0")
        return digest.hexdigest()

    def _cache_file(self, doc: Document) -> Path | None:
        return self.cache_dir / f"{_doc_id(doc)}.json" if self.cache_dir else None

    def _cached(self, documents: list[Document]) -> list[tuple[str, list[Chunk]]]:
        hits: list[tuple[str, list[Chunk]]] = []
        for doc in documents:
            path = self._cache_file(doc)
            if path is None or not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("key") != self._cache_key(doc):
                    continue
                chunks = Chunks.model_validate(data).chunks
            except (ValueError, OSError):
                continue
            if chunks:
                hits.append((_doc_id(doc), chunks))
        return hits

    def _save(self, doc: Document, chunks: list[Chunk]) -> None:
        path = self._cache_file(doc)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"key": self._cache_key(doc), "chunks": [chunk.model_dump() for chunk in chunks]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _doc_id(doc: Document) -> str:
    return str((doc.metadata or {}).get("doc_id") or "doc")


def _prompt_inputs(doc: Document) -> dict[str, str]:
    meta = doc.metadata or {}
    return {
        "title": str(meta.get("title") or ""),
        "doc_type": str(meta.get("doc_type") or ""),
        "date": str(meta.get("date") or ""),
        "doc_id": _doc_id(doc),
        "how_many": str(len(doc.page_content) // AVERAGE_CHUNK_CHARS + 1),
        "document": doc.page_content,
    }


def _as_document(doc: Document, chunk: Chunk, index: int) -> Document:
    """The headline becomes the section, which `with_ids_and_prefix` puts in the title line."""
    text = f"{chunk.summary.strip()}\n\n{chunk.original_text.strip()}"
    metadata = {**doc.metadata, "section": chunk.headline.strip() or "Overview"}
    return with_ids_and_prefix(Document(page_content=text, metadata=metadata), index)
