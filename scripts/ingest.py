"""Load, split, and embed corpus markdown through LangChain into Chroma."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from app.config import settings  # noqa: E402
from app.documents import NorthlineMarkdownLoader, split_corpus  # noqa: E402
from app.lc import overwrite_documents  # noqa: E402


def ingest() -> None:
    documents = NorthlineMarkdownLoader(settings.docs_path).load()
    if not documents:
        raise SystemExit(f"No markdown documents in {settings.docs_path}. Run build_corpus first.")

    chunks = split_corpus(documents)
    print(f"Embedding {len(chunks)} chunks with {settings.embed_model}…")
    count = overwrite_documents(chunks)
    print(f"Wrote {count} vectors to {settings.index_path}")


def main() -> None:
    ingest()


if __name__ == "__main__":
    main()
