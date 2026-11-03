"""Reciprocal Rank Fusion: merge the ranked results of several queries into one list."""

from __future__ import annotations

from langchain_core.documents import Document

RRF_K = 60


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """A chunk ranked high by many queries wins. One input list comes back unchanged.

    Chunks are deduplicated by id; the copy with the best vector score is kept.
    """
    fused: dict[str, float] = {}
    best: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = _chunk_key(doc)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            if key not in best or _score(doc) > _score(best[key]):
                best[key] = doc
    ordered = sorted(fused, key=lambda key: fused[key], reverse=True)
    return [best[key] for key in ordered]


def _chunk_key(doc: Document) -> str:
    return str((doc.metadata or {}).get("id") or doc.id or doc.page_content)


def _score(doc: Document) -> float:
    return float((doc.metadata or {}).get("score") or 0.0)
