from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.models import ChatTurn, Source

GROUNDED_SYSTEM = """You are the internal knowledge assistant for Northline Sensors SA, a fictional Lausanne company that builds environmental monitoring hardware.

Answer only from the retrieved passages below. If the passages do not contain the fact, say you do not have it in the knowledge base. Do not invent prices, lead times, suppliers, or contract terms. Follow-up questions still refer to Northline Sensors unless the user names another organization.

When two passages disagree, prefer the one with the newer date. Amendments supersede the original clause they change.

Do not mention retrieved passages, scores, or source titles. Answer the question directly.

Every document is fictional. Do not claim Northline Sensors is a real company.
"""

REWRITE_SYSTEM = """Given a chat history and the latest user question, which might reference context in the chat history, formulate a standalone search query for the Northline Sensors knowledge base.

Do not answer the question. Return only the search query as a single line. If the question is already standalone, return it unchanged. Spell out abbreviated officer titles such as CEO, CTO, and CFO. Follow-up questions still refer to Northline Sensors unless the user names another organization.
"""


def format_passages(sources: list[Source]) -> str:
    if not sources:
        return "No passages were retrieved."
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        header = (
            f"[{index}] {source.title} | {source.doc_type} | "
            f"{source.section} | {source.date} | score {source.score:.3f}"
        )
        blocks.append(f"{header}\n{source.text}")
    return "\n\n".join(blocks)


def rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def answer_prompt(sources: list[Source]) -> ChatPromptTemplate:
    system = GROUNDED_SYSTEM + "\n\nRetrieved passages:\n\n" + format_passages(sources)
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
        ]
    )


def as_chat_history(turns: list[ChatTurn]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in turns[-6:]:
        text = turn.content.strip()
        if not text:
            continue
        if turn.role == "user":
            messages.append(HumanMessage(content=text))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=text))
    return messages
