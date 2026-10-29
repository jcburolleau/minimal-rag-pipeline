"""Gradio chat over the Northline Sensors knowledge base."""

from __future__ import annotations

from collections.abc import Iterator

import gradio as gr

from app.chat import answer
from app.models import ChatTurn


def _turns(history: list | None) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    for item in history or []:
        if isinstance(item, dict):
            role = str(item.get("role") or "")
            content = item.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if role in {"user", "assistant"} and str(content).strip():
                turns.append(ChatTurn(role=role, content=str(content)))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user, assistant = item[0], item[1]
            if user:
                turns.append(ChatTurn(role="user", content=str(user)))
            if assistant:
                turns.append(ChatTurn(role="assistant", content=str(assistant)))
    return turns


def respond(message: str, history: list) -> Iterator[str]:
    text = message.strip()
    if not text:
        yield "Ask a question about a product, supplier, or contract."
        return
    accumulated = ""
    try:
        for chunk in answer(text, _turns(history)):
            accumulated += chunk
            yield accumulated
    except Exception as exc:  # noqa: BLE001 — show retrieve/generate failures in the chat
        yield accumulated + (f"\n\n{exc}" if accumulated else str(exc))


def main() -> None:
    chat = gr.ChatInterface(
        respond,
        title="Northline Sensors — Field knowledge",
        description=(
            "Internal lookup over a fictional instrument catalogue, suppliers, and contracts. "
            "Answers come from retrieved passages, not from the model's memory."
        ),
    )
    chat.launch()


if __name__ == "__main__":
    main()
