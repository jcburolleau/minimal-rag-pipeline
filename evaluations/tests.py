"""Load evaluations/tests.jsonl."""

from __future__ import annotations

from functools import cache
from typing import Literal

from pydantic import BaseModel, Field

from app.config import PIPELINE_ROOT

TESTS_PATH = PIPELINE_ROOT / "evaluations" / "tests.jsonl"


class TestQuestion(BaseModel):
    id: str
    category: str
    match: Literal["any", "all"]
    doc_ids: list[str] = Field(default_factory=list)
    question: str
    answer: str


@cache
def load_tests() -> tuple[TestQuestion, ...]:
    rows = []
    with TESTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(TestQuestion.model_validate_json(text))
    if not rows:
        raise RuntimeError(f"No questions in {TESTS_PATH}")
    return tuple(rows)


def select_tests(limit: int | None) -> list[TestQuestion]:
    tests = list(load_tests())
    if limit is None or limit >= len(tests):
        return tests
    return tests[: max(limit, 0)]
