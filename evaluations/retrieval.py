"""Retrieval evaluation: MRR and nDCG against gold document ids."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_chroma import Chroma

from app.config import settings
from app.lc import ground_query, search, vectorstore
from evaluations.metrics import score_retrieval
from evaluations.tests import select_tests

CATEGORY_ORDER = (
    "lookup",
    "multi_hop",
    "comparison",
    "temporal",
    "aggregation",
    "unanswerable",
)


@dataclass
class RetrievalScores:
    mrr: float
    ndcg: float
    scored: int
    skipped: int
    by_category: list[tuple[str, int, float, float]] = field(default_factory=list)

    @property
    def mrr_percent(self) -> float:
        return self.mrr * 100

    @property
    def ndcg_percent(self) -> float:
        return self.ndcg * 100


def document_ids(documents) -> list[str]:
    return [str((doc.metadata or {}).get("doc_id") or "") for doc in documents]


def evaluate_retrieval(
    k: int,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    store: Chroma | None = None,
) -> RetrievalScores:
    tests = select_tests(limit)
    chroma = store or vectorstore()
    mrr_total = 0.0
    ndcg_total = 0.0
    scored = 0
    skipped = 0
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    total = len(tests)
    for index_no, test in enumerate(tests, start=1):
        if on_progress:
            on_progress(index_no, total)
        if not test.doc_ids:
            skipped += 1
            continue
        retrieved = document_ids(search(ground_query(test.question), k, chroma))
        mrr, ndcg = score_retrieval(test.match, test.doc_ids, retrieved)
        mrr_total += mrr
        ndcg_total += ndcg
        scored += 1
        grouped[test.category].append((mrr, ndcg))
    return RetrievalScores(
        mrr=(mrr_total / scored) if scored else 0.0,
        ndcg=(ndcg_total / scored) if scored else 0.0,
        scored=scored,
        skipped=skipped,
        by_category=_category_rows(grouped),
    )


def _category_rows(
    grouped: dict[str, list[tuple[float, float]]],
) -> list[tuple[str, int, float, float]]:
    def order(category: str) -> tuple[int, str]:
        try:
            return (CATEGORY_ORDER.index(category), category)
        except ValueError:
            return (len(CATEGORY_ORDER), category)

    rows = []
    for category in sorted(grouped, key=order):
        pairs = grouped[category]
        count = len(pairs)
        rows.append(
            (
                category,
                count,
                sum(item[0] for item in pairs) / count,
                sum(item[1] for item in pairs) / count,
            )
        )
    return rows


def format_report(scores: RetrievalScores, k: int) -> str:
    return (
        f"MRR {scores.mrr_percent:.1f}%\n"
        f"nDCG {scores.ndcg_percent:.1f}%\n"
        f"Scored {scores.scored} questions at k={k}. "
        f"Skipped {scores.skipped} with no gold document."
    )


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    k = int(sys.argv[2]) if len(sys.argv) > 2 else settings.top_k
    scores = evaluate_retrieval(k, limit, on_progress=lambda done, total: print(f"\r{done}/{total}", end=""))
    print()
    print(format_report(scores, k))
    for category, count, mrr, ndcg in scores.by_category:
        print(f"  {category:<14} n={count:<4} MRR {mrr * 100:5.1f}%  nDCG {ndcg * 100:5.1f}%")


if __name__ == "__main__":
    main()
