from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "section"),
]


def split_frontmatter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    meta = yaml.safe_load(parts[1]) or {}
    return meta, parts[2].lstrip("\n")


class NorthlineMarkdownLoader(BaseLoader):
    """One Document per corpus markdown file.

    Stock TextLoader would drop YAML frontmatter. This keeps id/title/type/date
    on Document.metadata so the prompt inspector can still show them.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def lazy_load(self) -> Iterator[Document]:
        for file_path in sorted(self.path.glob("*.md")):
            raw = file_path.read_text(encoding="utf-8")
            meta, body = split_frontmatter(raw)
            doc_id = str(meta.get("id") or file_path.stem)
            yield Document(
                page_content=body,
                metadata={
                    "doc_id": doc_id,
                    "title": str(meta.get("title") or doc_id),
                    "doc_type": str(meta.get("doc_type") or "document"),
                    "date": str(meta.get("date") or ""),
                    "source": str(file_path),
                },
            )


def split_corpus(documents: list[Document]) -> list[Document]:
    """Header split, then markdown-aware recursive split for long sections."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    length_splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=150)

    headed: list[Document] = []
    for doc in documents:
        parts = header_splitter.split_text(doc.page_content)
        if not parts:
            parts = [Document(page_content=doc.page_content, metadata={})]
        for part in parts:
            text = part.page_content.strip()
            if not text:
                continue
            metadata = {**doc.metadata, **part.metadata}
            metadata["section"] = str(
                metadata.get("section") or metadata.get("h1") or "Overview"
            )
            headed.append(Document(page_content=text, metadata=metadata))

    split = length_splitter.split_documents(headed)
    counts: dict[str, int] = {}
    documents: list[Document] = []
    for doc in split:
        doc_id = str(doc.metadata.get("doc_id") or "doc")
        index = counts.get(doc_id, 0)
        counts[doc_id] = index + 1
        documents.append(_with_ids_and_prefix(doc, index))
    return documents


def _with_ids_and_prefix(doc: Document, index: int) -> Document:
    metadata = dict(doc.metadata)
    doc_id = str(metadata.get("doc_id") or "doc")
    section = str(metadata.get("section") or "Overview")
    title = str(metadata.get("title") or doc_id)
    doc_type = str(metadata.get("doc_type") or "document")
    date = str(metadata.get("date") or "")
    chunk_id = f"{doc_id}::{index}"
    prefix = f"{title} ({doc_type}, {date}) — {section}\n\n"
    text = doc.page_content.strip()
    if not text.startswith(prefix):
        text = prefix + text
    return Document(
        id=chunk_id,
        page_content=text,
        metadata={
            "id": chunk_id,
            "doc_id": doc_id,
            "title": title,
            "doc_type": doc_type,
            "date": date,
            "section": section,
        },
    )
