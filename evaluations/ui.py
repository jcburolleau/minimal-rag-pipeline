"""Gradio runner for retrieval and answer evaluation."""

from __future__ import annotations

import gradio as gr

from app.config import settings
from app.documents import CHUNK_OVERLAP, CHUNK_SIZE
from evaluations.answer import evaluate_answer
from evaluations.retrieval import evaluate_retrieval
from evaluations.tests import load_tests


def _retrieval_view(k: int, limit: int, progress=gr.Progress()) -> tuple[str, list[list[str]]]:
    try:
        scores = evaluate_retrieval(
            int(k),
            int(limit),
            on_progress=lambda done, total: progress(done / total, desc=f"Retrieval {done}/{total}"),
        )
    except Exception as exc:  # noqa: BLE001 — show the failure on the page
        return f"Retrieval evaluation failed.\n\n{exc}", []
    summary = (
        f"### MRR\n**{scores.mrr_percent:.1f}%**\n\n"
        f"### nDCG\n**{scores.ndcg_percent:.1f}%**\n\n"
        f"Scored {scores.scored} questions at k={int(k)}. "
        f"Skipped {scores.skipped} questions with no gold document."
    )
    rows = [
        [category, str(count), f"{mrr * 100:.1f}%", f"{ndcg * 100:.1f}%"]
        for category, count, mrr, ndcg in scores.by_category
    ]
    return summary, rows


def _answer_view(k: int, limit: int, progress=gr.Progress()) -> tuple[str, list[list[str]]]:
    try:
        scores = evaluate_answer(
            int(k),
            int(limit),
            on_progress=lambda done, total: progress(done / total, desc=f"Answer {done}/{total}"),
        )
    except Exception as exc:  # noqa: BLE001 — show the failure on the page
        return f"Answer evaluation failed.\n\n{exc}", []
    summary = (
        f"### Accuracy\n**{scores.accuracy_percent:.1f}%**\n\n"
        f"### Completeness\n**{scores.completeness_percent:.1f}%**\n\n"
        f"### Relevance\n**{scores.relevance_percent:.1f}%**\n\n"
        f"Mean scores {scores.accuracy:.2f} / {scores.completeness:.2f} / {scores.relevance:.2f} out of 5. "
        f"Scored {scores.scored} questions at k={int(k)}. Failed {scores.failed}."
    )
    if scores.last_error:
        summary += f"\n\nLast error: {scores.last_error}"
    rows = [
        [
            category,
            str(count),
            f"{accuracy / 5 * 100:.1f}%",
            f"{completeness / 5 * 100:.1f}%",
            f"{relevance / 5 * 100:.1f}%",
        ]
        for category, count, accuracy, completeness, relevance in scores.by_category
    ]
    return summary, rows


def main() -> None:
    total = len(load_tests())
    with gr.Blocks(title="Northline evaluation") as demo:
        gr.Markdown(
            f"""# Northline evaluation

Retrieval is **MRR** and **nDCG**. Answers are judged for **accuracy**, **completeness**, and **relevance** (1–5, shown as a percentage of 5).

This run uses `{settings.embed_model}` and `{settings.chat_model}`, chunks of {CHUNK_SIZE} characters with {CHUNK_OVERLAP} overlap. Top-k changes the ranking for this run. A new embedding model or chunk size needs a fresh ingest before the percentages are comparable.
"""
        )
        with gr.Row():
            k = gr.Slider(1, 20, value=settings.top_k, step=1, label="Top K")
            limit = gr.Slider(1, total, value=total, step=1, label="Questions")
        with gr.Row():
            retrieval_button = gr.Button("Run retrieval evaluation", variant="primary")
            answer_button = gr.Button("Run answer evaluation")
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Retrieval")
                retrieval_summary = gr.Markdown("Not run.")
                retrieval_table = gr.Dataframe(
                    headers=["Category", "Questions", "MRR", "nDCG"],
                    value=[],
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("## Answer")
                answer_summary = gr.Markdown(
                    "Not run. Each question calls the chat model, then the judge."
                )
                answer_table = gr.Dataframe(
                    headers=["Category", "Questions", "Accuracy", "Completeness", "Relevance"],
                    value=[],
                    interactive=False,
                )
        retrieval_button.click(
            _retrieval_view,
            inputs=[k, limit],
            outputs=[retrieval_summary, retrieval_table],
        )
        answer_button.click(
            _answer_view,
            inputs=[k, limit],
            outputs=[answer_summary, answer_table],
        )
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    main()
