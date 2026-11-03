"""Retrieval evaluation: MRR and nDCG against gold document ids."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from langchain_chroma import Chroma

from app.config import collection_for, settings
from app.lc import vectorstore
from app.retrieve import RetrievalOptions, retrieve
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
    options: RetrievalOptions | None = None,
) -> RetrievalScores:
    tests = select_tests(limit)
    chroma = store or vectorstore()
    options = replace(options or RetrievalOptions(), k=k)
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
        retrieved = document_ids(retrieve(test.question, [], options, chroma))
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


def parse_args(description: str | None) -> argparse.Namespace:
    """`limit` and `k` stay positional; technique flags override .env for this run."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("limit", nargs="?", type=int, default=None)
    parser.add_argument("k", nargs="?", type=int, default=settings.top_k)
    parser.add_argument("--index", choices=["markdown", "llm"], default=settings.chunker)
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=settings.rewrite_query)
    parser.add_argument("--expand", action=argparse.BooleanOptionalAction, default=settings.expand_query)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=settings.rerank)
    parser.add_argument("--candidates", type=int, default=settings.candidate_k)
    return parser.parse_args()


def options_from_args(args: argparse.Namespace) -> RetrievalOptions:
    return RetrievalOptions(
        rewrite=args.rewrite,
        expand=args.expand,
        rerank=args.rerank,
        k=args.k,
        candidate_k=args.candidates,
    )


def index_store(index: str) -> Chroma:
    """`markdown` or `llm` to the matching Chroma collection."""
    return vectorstore(collection_for(settings.table_name, index))


def options_label(index: str, options: RetrievalOptions) -> str:
    enabled = [name for name in ("rewrite", "expand", "rerank") if getattr(options, name)]
    techniques = ", ".join(enabled) or "no techniques"
    if options.rerank:
        techniques += f" ({options.candidate_k} candidates)"
    return f"Index {index}, {techniques}"


def main() -> None:
    args = parse_args(__doc__)
    options = options_from_args(args)
    store = index_store(args.index)
    scores = evaluate_retrieval(
        args.k,
        args.limit,
        on_progress=lambda done, total: print(f"\r{done}/{total}", end=""),
        store=store,
        options=options,
    )
    print()
    print(options_label(args.index, options))
    print(format_report(scores, args.k))
    for category, count, mrr, ndcg in scores.by_category:
        print(f"  {category:<14} n={count:<4} MRR {mrr * 100:5.1f}%  nDCG {ndcg * 100:5.1f}%")


if __name__ == "__main__":
    main()
