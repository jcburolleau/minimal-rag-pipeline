"""Load, split, and embed corpus markdown through LangChain into Chroma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from langchain_core.documents import Document  # noqa: E402

from app.config import collection_for, settings  # noqa: E402
from app.documents import NorthlineMarkdownLoader, split_corpus  # noqa: E402
from app.lc import llm, overwrite_documents  # noqa: E402
from app.techniques.preprocess import LLMChunker  # noqa: E402


def split(documents: list[Document], chunker: str) -> list[Document]:
    if chunker == "markdown":
        return split_corpus(documents)
    model_name = settings.preprocess_model or settings.chat_model
    llm_chunker = LLMChunker(
        llm(model=model_name, streaming=False, temperature=0),
        model_name,
        cache_dir=settings.preprocessed_path,
    )
    return llm_chunker.split_documents(documents)


def ingest(chunker: str = settings.chunker) -> None:
    documents = NorthlineMarkdownLoader(settings.docs_path).load()
    if not documents:
        raise SystemExit(f"No markdown documents in {settings.docs_path}. Run build_corpus first.")

    collection = collection_for(settings.table_name, chunker)
    chunks = split(documents, chunker)
    print(f"Embedding {len(chunks)} chunks with {settings.embed_model} into '{collection}'…")
    count = overwrite_documents(chunks, collection)
    print(f"Wrote {count} vectors to {settings.index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunker",
        choices=["markdown", "llm"],
        default=settings.chunker,
        help="markdown: header + length split (collection 'northline'). "
        "llm: LLM pre-processing into headline/summary/original chunks (collection 'northline_llm').",
    )
    ingest(parser.parse_args().chunker)


if __name__ == "__main__":
    main()
