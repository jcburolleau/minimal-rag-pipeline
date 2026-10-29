# Northline Sensors — local RAG

A local retrieval-augmented chat over a **fictional** Lausanne hardware company. Northline Sensors does not exist. The catalogue, suppliers, prices, and contracts are authored for this demo so the model cannot answer from memory.

A [Gradio](https://www.gradio.app/) chat calls a [LangChain](https://python.langchain.com/) LCEL chain: OpenAI embeddings, a history-aware Chroma retriever, a grounded `ChatPromptTemplate`, and streaming `ChatOpenAI`.

## Stack

- Gradio chat UI
- LangChain LCEL (`langchain`, `langchain-openai`, `langchain-chroma`) over OpenAI and local Chroma
- Chroma (local files under `data/chroma/`)
- OpenAI: `text-embedding-3-small` for embeddings, `gpt-4o-mini` for answers

## Prerequisites

- Python 3.12 (this repo pins it with `uv`)
- An `OPENAI_API_KEY` in `.env`

## Setup

```bash
cp .env.example .env
# set OPENAI_API_KEY in .env

uv sync
uv run python -m scripts.build_corpus
uv run python -m scripts.ingest
uv run python -m app.ui
```

Open the Gradio URL printed in the terminal (usually [http://127.0.0.1:7860](http://127.0.0.1:7860)).

## Example questions

- What is the unit price and lead time for the river gauge, and which supplier builds its ultrasonic sensor?
- Which supply contract renews first, and what is the liability cap?
- Did the 2025 amendment change the soil-probe lead time?

Turn **Search knowledge base** off and ask the same questions. The model is instructed to refuse company-specific facts.

## Corpus

Hand-written entities live in [`corpus/entities/`](corpus/entities/). [`scripts/build_corpus.py`](scripts/build_corpus.py) renders them to markdown in [`corpus/docs/`](corpus/docs/). Ingest loads those files as LangChain `Document`s, splits with `MarkdownHeaderTextSplitter` then `MarkdownTextSplitter`, and overwrites the `northline` Chroma collection through the LangChain store. Re-run ingest after any embed-model change.

Amendment **AMD-2025-01** (18 April 2025) changes the Rhine Silicon moisture-IC lead time from 45 days to 70 days. When two passages disagree, the chat prompt tells the model to prefer the newer date.

## Layout

```
app/          Gradio chat and LangChain pipeline
scripts/      corpus renderer and ingest
corpus/       YAML entities and generated markdown
data/chroma/  local Chroma collection (gitignored)
.env          OpenAI key and model names (gitignored)
```
