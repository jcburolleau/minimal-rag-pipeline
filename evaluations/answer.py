"""Answer evaluation: accuracy, completeness, and relevance against the reference answer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.chat import answer_once
from app.config import settings
from app.openai_client import api_key
from app.retrieve import RetrievalOptions
from evaluations.retrieval import (
    CATEGORY_ORDER,
    index_store,
    options_from_args,
    options_label,
    parse_args,
)
from evaluations.tests import TestQuestion, select_tests

JUDGE_SYSTEM = (
    "You are an expert evaluator assessing the quality of answers. "
    "Evaluate the generated answer by comparing it to the reference answer. "
    "Only give 5/5 scores for perfect answers."
)


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description=(
            "Concise feedback on the answer quality, comparing it to the reference "
            "answer and evaluating based on the retrieved context"
        )
    )
    accuracy: float = Field(
        ge=1,
        le=5,
        description=(
            "How factually correct is the answer compared to the reference answer? "
            "1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). "
            "An acceptable answer would score 3."
        ),
    )
    completeness: float = Field(
        ge=1,
        le=5,
        description=(
            "How complete is the answer in addressing all aspects of the question? "
            "1 (very poor - missing key information) to 5 (ideal - all the information "
            "from the reference answer is provided completely). Only answer 5 if ALL "
            "information from the reference answer is included."
        ),
    )
    relevance: float = Field(
        ge=1,
        le=5,
        description=(
            "How relevant is the answer to the specific question asked? "
            "1 (very poor - off-topic) to 5 (ideal - directly addresses question and "
            "gives no additional information). Only answer 5 if the answer is completely "
            "relevant to the question and gives no additional information."
        ),
    )


@dataclass
class AnswerScores:
    accuracy: float
    completeness: float
    relevance: float
    scored: int
    failed: int
    last_error: str = ""
    by_category: list[tuple[str, int, float, float, float]] = field(default_factory=list)

    @property
    def accuracy_percent(self) -> float:
        return self.accuracy / 5 * 100

    @property
    def completeness_percent(self) -> float:
        return self.completeness / 5 * 100

    @property
    def relevance_percent(self) -> float:
        return self.relevance / 5 * 100


def _judge() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=api_key(),
        temperature=0,
        streaming=False,
    ).with_structured_output(AnswerEval)


def _context(documents) -> str:
    if not documents:
        return "No passages were retrieved."
    blocks = []
    for index, doc in enumerate(documents, start=1):
        doc_id = str((doc.metadata or {}).get("doc_id") or "")
        blocks.append(f"[{index}] {doc_id}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def _judge_messages(test: TestQuestion, generated: str, documents) -> list:
    return [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=f"""Question:
{test.question}

Generated Answer:
{generated}

Reference Answer:
{test.answer}

Retrieved context:
{_context(documents)}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1. Scores compare the generated answer to the reference answer."""
        ),
    ]


def evaluate_answer(
    k: int,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    options: RetrievalOptions | None = None,
    store: Chroma | None = None,
) -> AnswerScores:
    tests = select_tests(limit)
    options = replace(options or RetrievalOptions(), k=k)
    judge = _judge()
    totals = {"accuracy": 0.0, "completeness": 0.0, "relevance": 0.0}
    scored = 0
    failed = 0
    last_error = ""
    grouped: dict[str, list[AnswerEval]] = defaultdict(list)
    total = len(tests)
    for index, test in enumerate(tests, start=1):
        if on_progress:
            on_progress(index, total)
        try:
            generated, documents = answer_once(test.question, options, store)
            verdict = judge.invoke(_judge_messages(test, generated, documents))
            if not isinstance(verdict, AnswerEval):
                verdict = AnswerEval.model_validate(verdict)
        except Exception as exc:
            failed += 1
            last_error = str(exc)
            continue
        totals["accuracy"] += verdict.accuracy
        totals["completeness"] += verdict.completeness
        totals["relevance"] += verdict.relevance
        scored += 1
        grouped[test.category].append(verdict)
    if scored == 0:
        return AnswerScores(0.0, 0.0, 0.0, 0, failed, last_error)
    return AnswerScores(
        accuracy=totals["accuracy"] / scored,
        completeness=totals["completeness"] / scored,
        relevance=totals["relevance"] / scored,
        scored=scored,
        failed=failed,
        last_error=last_error,
        by_category=_category_rows(grouped),
    )


def _category_rows(
    grouped: dict[str, list[AnswerEval]],
) -> list[tuple[str, int, float, float, float]]:
    def order(category: str) -> tuple[int, str]:
        try:
            return (CATEGORY_ORDER.index(category), category)
        except ValueError:
            return (len(CATEGORY_ORDER), category)

    rows = []
    for category in sorted(grouped, key=order):
        verdicts = grouped[category]
        count = len(verdicts)
        rows.append(
            (
                category,
                count,
                sum(item.accuracy for item in verdicts) / count,
                sum(item.completeness for item in verdicts) / count,
                sum(item.relevance for item in verdicts) / count,
            )
        )
    return rows


def format_report(scores: AnswerScores, k: int) -> str:
    return (
        f"Accuracy {scores.accuracy_percent:.1f}%\n"
        f"Completeness {scores.completeness_percent:.1f}%\n"
        f"Relevance {scores.relevance_percent:.1f}%\n"
        f"Scored {scores.scored} questions at k={k}. Failed {scores.failed}."
        + (f"\nLast error: {scores.last_error}" if scores.last_error else "")
    )


def main() -> None:
    args = parse_args(__doc__)
    options = options_from_args(args)
    scores = evaluate_answer(
        args.k,
        args.limit,
        on_progress=lambda done, total: print(f"\r{done}/{total}", end=""),
        options=options,
        store=index_store(args.index),
    )
    print()
    print(options_label(args.index, options))
    print(format_report(scores, args.k))
    for category, count, accuracy, completeness, relevance in scores.by_category:
        print(
            f"  {category:<14} n={count:<4} "
            f"accuracy {accuracy / 5 * 100:5.1f}%  "
            f"completeness {completeness / 5 * 100:5.1f}%  "
            f"relevance {relevance / 5 * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
