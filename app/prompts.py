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

KNOWLEDGE_BASE_SCOPE = """The Northline Sensors SA knowledge base holds the company overview and leadership, product sheets (SKU, list price, lead time, specifications, bill of materials), supplier profiles, customer contracts and their amendments, internal memos, and policies."""

QUERY_REWRITE_SYSTEM = f"""You are about to search a knowledge base to answer the user's latest question. {KNOWLEDGE_BASE_SCOPE}

Using the chat history for context, write one short, specific search query that is most likely to surface the passages that answer the latest question. Resolve references to earlier turns into the names they stand for. Keep product names, SKUs, supplier names, and document ids exactly as written. Spell out abbreviated officer titles such as CEO, CTO, and CFO. Follow-up questions still refer to Northline Sensors unless the user names another organization.

Do not answer the question. Return only the search query as a single line.
"""

QUERY_EXPAND_SYSTEM = f"""You write search queries for a knowledge base. {KNOWLEDGE_BASE_SCOPE}

Write {{n}} different search queries that together surface every passage needed to answer the question. Vary the wording and use synonyms. When the question compares several items or needs several facts, give each item or fact its own query. Keep product names, SKUs, supplier names, and document ids exactly as written. Spell out abbreviated officer titles such as CEO, CTO, and CFO.

Do not answer the question. Do not repeat the question word for word.
"""

RERANK_SYSTEM = """You are a document re-ranker for the Northline Sensors SA knowledge base.
You are given a question and chunks of text retrieved from the knowledge base. The chunks are in retrieval order, which is roughly by relevance, but you can improve on it.
Rank the chunks by how relevant they are to answering the question, most relevant first. When chunks disagree, the newer dated chunk is more relevant; amendments supersede the clause they change.
Reply only with the ranked chunk ids. Include every chunk id you were given exactly once.
"""

PREPROCESS_SYSTEM = """You split a document into overlapping chunks for a knowledge base.

The document is from the internal knowledge base of Northline Sensors SA, a fictional Lausanne company that builds environmental monitoring hardware. A chatbot will use these chunks to answer questions about the company.

Divide the document as you see fit, but return the entire document across the chunks: do not leave anything out. Make sure there are individual chunks that answer specific questions. Overlap neighbouring chunks by about 25% so the same text appears in more than one chunk.

For each chunk, give a headline, a summary, and the original text. The original text must be copied exactly from the document, not changed in any way.
"""

PREPROCESS_HUMAN = """Document title: {title}
Document type: {doc_type}
Document date: {date}
Document id: {doc_id}

This document should probably be split into at least {how_many} chunks, but use more or fewer as appropriate.

Here is the document:

{document}

Respond with the chunks."""


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


def query_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_REWRITE_SYSTEM),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
        ]
    )


def query_expand_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_EXPAND_SYSTEM),
            ("human", "{question}"),
        ]
    )


def rerank_messages(question: str, chunks: list[str]) -> list[BaseMessage]:
    """Chunk ids are 1-based, as the re-ranker replies with them."""
    blocks = "\n\n".join(
        f"# CHUNK ID: {index}\n\n{text}" for index, text in enumerate(chunks, start=1)
    )
    return [
        SystemMessage(content=RERANK_SYSTEM),
        HumanMessage(
            content=(
                f"The user has asked the following question:\n\n{question}\n\n"
                "Order all the chunks of text by relevance to the question, from most "
                "relevant to least relevant. Include all the chunk ids you are given.\n\n"
                f"Here are the chunks:\n\n{blocks}\n\n"
                "Reply only with the list of ranked chunk ids, nothing else."
            )
        ),
    ]


def preprocess_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", PREPROCESS_SYSTEM),
            ("human", PREPROCESS_HUMAN),
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
