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

## Corpus

Hand-written entities live in [`corpus/entities/`](corpus/entities/). [`scripts/build_corpus.py`](scripts/build_corpus.py) renders them to markdown in [`corpus/docs/`](corpus/docs/). Ingest loads those files as LangChain `Document`s, splits with `MarkdownHeaderTextSplitter` then `MarkdownTextSplitter`, and overwrites the `northline` Chroma collection through the LangChain store. Re-run ingest after any embed-model change.

Amendment **AMD-2025-01** (18 April 2025) changes the Rhine Silicon moisture-IC lead time from 45 days to 70 days. When two passages disagree, the chat prompt tells the model to prefer the newer date.

## Advanced retrieval

Four techniques from the course examples in [`advanced_rag_techniques/`](advanced_rag_techniques/), rebuilt as independent LangChain modules in [`app/techniques/`](app/techniques/). All are off by default; [`app/retrieve.py`](app/retrieve.py) composes whichever are enabled, and the chat and both evals call it.

| Technique | Module | When | Switch |
|---|---|---|---|
| Document pre-processing | `preprocess.py` (`LLMChunker`) | ingest | `uv run python -m scripts.ingest --chunker llm`, then `CHUNKER=llm` |
| Query rewriting | `rewrite.py` (`query_rewriter`) | per question | `REWRITE_QUERY=true` |
| Query expansion | `expand.py` (`query_expander`) + `fusion.py` (RRF) | per question | `EXPAND_QUERY=true`, `EXPAND_N` |
| Re-ranking | `rerank.py` (`LLMReranker`) | per question | `RERANK=true`, `CANDIDATE_K` |

- **Pre-processing** asks the LLM to split each document into chunks with a headline, a summary, and the original text. It writes to a separate `northline_llm` collection, so both indexes coexist. LLM output is cached in `data/preprocessed/` (gitignored), so re-embedding does not re-call the LLM.
- **Rewriting** turns every question, with the chat history, into one short search query. Without history the original question is searched too.
- **Expansion** adds alternative queries. Results from all queries are merged with reciprocal rank fusion.
- **Re-ranking** pulls `CANDIDATE_K` chunks per query, has the LLM order them by relevance to the question, and keeps the top `TOP_K`.

Each per-question technique adds one `gpt-4o-mini` call. Choose them from the retrieval eval:

```bash
uv run python -m evaluations.retrieval                      # baseline (flags from .env)
uv run python -m evaluations.retrieval --rerank             # one technique
uv run python -m evaluations.retrieval --rewrite --expand --rerank --candidates 20
uv run python -m evaluations.retrieval --index llm          # pre-processed index
uv run python -m evaluations.answer 20 --rerank             # answer judge on 20 questions
uv run python -m evaluations.ui                             # same switches in Gradio
```

Command-line flags override `.env` for that run only; `--no-rerank` etc. turn a technique off.

The chat reads the switches from `.env` once, at startup, and lists which techniques are on under its title. Restart `uv run python -m app.ui` after changing them.

## Layout

```
app/          Gradio chat and LangChain pipeline
app/techniques/  pre-processing, rewrite, expansion, fusion, re-ranking
evaluations/  retrieval (MRR, nDCG) and answer (LLM judge) evals
scripts/      corpus renderer and ingest
corpus/       YAML entities and generated markdown
data/chroma/  local Chroma collections (gitignored)
data/preprocessed/  cached LLM pre-processing output (gitignored)
.env          OpenAI key and model names (gitignored)
```
