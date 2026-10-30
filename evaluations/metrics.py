"""MRR and nDCG, same rank math as a keyword eval, with gold document ids as the hits."""

from __future__ import annotations

import math


def dcg(relevances: list[int], k: int) -> float:
    total = 0.0
    for index in range(min(k, len(relevances))):
        total += relevances[index] / math.log2(index + 2)
    return total


def ndcg(relevances: list[int], k: int) -> float:
    """Binary nDCG. Ideal ranking is the retrieved hits moved to the front."""
    gained = dcg(relevances, k)
    ideal = dcg(sorted(relevances, reverse=True), k)
    return gained / ideal if ideal > 0 else 0.0


def reciprocal_rank(relevances: list[int]) -> float:
    for rank, relevant in enumerate(relevances, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def score_retrieval(match: str, gold_ids: list[str], retrieved_ids: list[str]) -> tuple[float, float]:
    """Mean MRR and nDCG for one question.

    `any` treats a chunk as relevant when its document is in the gold set.
    `all` scores each gold document on its own, then averages, the same way a
    keyword eval averages across keywords.
    """
    gold = list(dict.fromkeys(gold_ids))
    if not gold:
        return 0.0, 0.0
    k = len(retrieved_ids)
    if match == "any":
        relevances = [1 if doc_id in gold else 0 for doc_id in retrieved_ids]
        return reciprocal_rank(relevances), ndcg(relevances, k)
    scores = [
        (
            reciprocal_rank([1 if doc_id == gold_id else 0 for doc_id in retrieved_ids]),
            ndcg([1 if doc_id == gold_id else 0 for doc_id in retrieved_ids], k),
        )
        for gold_id in gold
    ]
    return (
        sum(item[0] for item in scores) / len(scores),
        sum(item[1] for item in scores) / len(scores),
    )
