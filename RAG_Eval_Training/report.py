# =============================================================================
# report.py — Rich Terminal Report for Evaluation Results
# =============================================================================
#
# RESPONSIBILITY: Pretty-print all evaluation results to the terminal.
# Uses the 'rich' library for tables, color-coded scores, and progress bars.
#
# WHY A SEPARATE REPORT MODULE?
# Separation of concerns. The evaluation module computes scores.
# This module formats and displays them. Swapping to a JSON export,
# a PDF report, or a dashboard is now a 1-file change.
# =============================================================================

from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


console = Console()


# ---------------------------------------------------------------------------
# 1. Score Color Coding
# ---------------------------------------------------------------------------
def _score_color(score: float) -> str:
    """Returns a Rich color string based on score thresholds."""
    if score >= 0.9:
        return "bold green"
    elif score >= 0.7:
        return "green"
    elif score >= 0.5:
        return "yellow"
    else:
        return "bold red"


def _score_label(score: float) -> str:
    """Returns a human-readable label for the score range."""
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.5:
        return "Needs Work"
    else:
        return "Poor"


# ---------------------------------------------------------------------------
# 2. Print Aggregate Scores
# ---------------------------------------------------------------------------
def print_aggregate_report(scores: Dict[str, float]) -> None:
    """Prints the aggregate RAG Triad evaluation scores."""

    console.print("\n")
    console.print(Panel(
        "[bold cyan]RAG TRIAD — AGGREGATE EVALUATION SCORES[/bold cyan]",
        box=box.DOUBLE,
        style="cyan",
    ))

    table = Table(
        title="Evaluation Results",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Metric", style="bold white", width=30)
    table.add_column("Score", justify="center", width=12)
    table.add_column("Rating", justify="center", width=14)
    table.add_column("What It Measures", style="dim", width=50)

    metric_descriptions = {
        "faithfulness":        "Answer is grounded in retrieved context (no hallucinations)",
        "answer_relevancy":    "Answer actually addresses the user's question",
        "context_precision":   "Retrieved chunks are relevant to the question",
        "context_recall":      "All necessary chunks were retrieved",
    }

    metric_display_names = {
        "faithfulness":        "Faithfulness",
        "answer_relevancy":    "Answer Relevance",
        "context_precision":   "Context Precision",
        "context_recall":      "Context Recall",
    }

    for metric_key, score in scores.items():
        color = _score_color(score)
        label = _score_label(score)
        display_name = metric_display_names.get(metric_key, metric_key)
        description = metric_descriptions.get(metric_key, "—")

        table.add_row(
            display_name,
            f"[{color}]{score:.3f}[/{color}]",
            f"[{color}]{label}[/{color}]",
            description,
        )

    console.print(table)

    # Print overall summary
    avg_score = sum(scores.values()) / len(scores)
    console.print(f"\n  [bold]Overall Average:[/bold] "
                  f"[{_score_color(avg_score)}]{avg_score:.3f} "
                  f"({_score_label(avg_score)})[/{_score_color(avg_score)}]\n")


# ---------------------------------------------------------------------------
# 3. Print Per-Question Breakdown
# ---------------------------------------------------------------------------
def print_per_question_report(per_sample_scores: List[Dict[str, Any]]) -> None:
    """Prints a detailed per-question breakdown of scores."""

    console.print(Panel(
        "[bold cyan]PER-QUESTION SCORE BREAKDOWN[/bold cyan]",
        box=box.DOUBLE,
        style="cyan",
    ))

    table = Table(
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", width=4, justify="center")
    table.add_column("Question", width=45)
    table.add_column("Faith.", width=9, justify="center")
    table.add_column("Rel.", width=9, justify="center")
    table.add_column("Prec.", width=9, justify="center")
    table.add_column("Rec.", width=9, justify="center")

    for i, sample in enumerate(per_sample_scores, 1):
        # Truncate long questions for display
        q = sample["question"]
        if len(q) > 42:
            q = q[:39] + "..."

        faith = sample["faithfulness"]
        rel = sample["answer_relevancy"]
        prec = sample["context_precision"]
        rec = sample["context_recall"]

        table.add_row(
            str(i),
            q,
            f"[{_score_color(faith)}]{faith:.2f}[/{_score_color(faith)}]",
            f"[{_score_color(rel)}]{rel:.2f}[/{_score_color(rel)}]",
            f"[{_score_color(prec)}]{prec:.2f}[/{_score_color(prec)}]",
            f"[{_score_color(rec)}]{rec:.2f}[/{_score_color(rec)}]",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# 4. Print Q&A Pairs (for manual inspection)
# ---------------------------------------------------------------------------
def print_qa_pairs(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
) -> None:
    """Prints each question, its retrieved context, and the generated answer."""

    console.print(Panel(
        "[bold cyan]QUESTION & ANSWER PAIRS (with Retrieved Context)[/bold cyan]",
        box=box.DOUBLE,
        style="cyan",
    ))

    for i, (q, a, ctx) in enumerate(zip(questions, answers, contexts), 1):
        console.print(f"\n[bold yellow]── Question {i} ──[/bold yellow]")
        console.print(f"[bold]Q:[/bold] {q}")
        console.print(f"\n[dim]Retrieved Context ({len(ctx)} chunk(s)):[/dim]")
        for j, chunk in enumerate(ctx, 1):
            # Show first 150 chars of each chunk
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            console.print(f"  [dim]  Chunk {j}:[/dim] {preview}")
        console.print(f"\n[bold green]A:[/bold green] {a}")
        console.print("[dim]" + "─" * 80 + "[/dim]")


# ---------------------------------------------------------------------------
# 5. Print Interview Prep Summary
# ---------------------------------------------------------------------------
def print_interview_prep_summary() -> None:
    """Prints a quick-reference card of key interview topics covered."""

    console.print(Panel(
        "[bold magenta]INTERVIEW PREP — KEY TOPICS COVERED IN THIS PROJECT[/bold magenta]",
        box=box.DOUBLE,
        style="magenta",
    ))

    topics = [
        ("Chunking Strategy",     "RecursiveCharacterTextSplitter vs alternatives; chunk_size & overlap tuning"),
        ("Embedding Choice",      "text-embedding-3-small vs large; dimensionality vs cost tradeoffs"),
        ("Retrieval Tuning",      "K selection; 'lost in the middle' problem; query rewriting (HyDE)"),
        ("Hallucination Control", "System prompt constraints; the 'I don't know' escape hatch"),
        ("The RAG Triad",         "Faithfulness, Answer Relevance, Context Precision/Recall"),
        ("LLM-as-Judge",          "Why lexical metrics (BLEU) fail; semantic evaluation via LLM"),
        ("Eval Frameworks",       "RAGAS vs DeepEval vs TruLens — when to use which"),
        ("Production Concerns",   "Idempotent ingestion; embedding drift; per-question monitoring"),
        ("Token Management",      "Context window budgets; cost optimization; truncation strategies"),
        ("Debugging Retrieval",   "Score logging; similarity thresholds; per-question score alerts"),
    ]

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Topic", style="bold magenta", width=28)
    table.add_column("Key Point", style="white", width=72)

    for topic, point in topics:
        table.add_row(topic, point)

    console.print(table)
