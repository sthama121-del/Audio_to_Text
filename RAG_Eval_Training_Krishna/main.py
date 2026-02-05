"""main.py

Runs:
1) RAG pipeline for each evaluation question
2) RAGAS evaluation (triad metrics)
3) Retrieval-only evaluation (hit-rate/MRR proxy)
4) Writes artifacts to outputs/

RUN:
  python main.py

If interviewers ask: "How do you evaluate RAG in production?"
Answer:
- Offline: golden set + RAGAS + retrieval metrics
- Online: per-query logging + sampling + drift alerts + regression tests in CI
"""

from __future__ import annotations
import os
import json
import statistics
from datetime import datetime
from typing import Dict, Any, List
from rich import print

from rag.config import Settings, validate
from rag.vectorstore import load_vectorstore
from rag.rag_pipeline import run_one
from eval.datasets import EVAL_QUESTIONS
from eval.ragas_eval import run_ragas
from eval.retrieval_eval import score_retrieval
from eval.report import write_json, write_markdown_report

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    settings = Settings()
    validate(settings)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    # Load vector store (must exist)
    if not os.path.exists(settings.CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store not found at {settings.CHROMA_DIR}. Run: python ingest.py"
        )
    vs = load_vectorstore(settings)
    print(f"[bold green]✓ Loaded vector store[/bold green] dir={settings.CHROMA_DIR} collection={settings.CHROMA_COLLECTION}")

    rows: List[Dict[str, Any]] = []
    latencies = []

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        q = item["question"]
        gt = item["ground_truth"]

        print(f"\n[bold]Q{i}/{len(EVAL_QUESTIONS)}[/bold] {q}")
        out = run_one(q, vs, settings)

        # Serialize retrieved chunks as plain text for evaluation frameworks
        retrieved_texts = [d.page_content for d in out.retrieved_docs]

        row = {
            "question": q,
            "ground_truth": gt,
            "answer": out.answer,
            "contexts": retrieved_texts,  # RAGAS expects list[str]
            "retrieved_texts": retrieved_texts,
            "retrieval_scores": out.retrieval_scores,
            "query_used": out.query_used,
            "refused_for_low_evidence": out.refused_for_low_evidence,
            "latency_ms": out.latency_ms,
        }
        rows.append(row)
        latencies.append(out.latency_ms)

        # Log retrieval scores (debugging retrieval is a superpower)
        if out.retrieval_scores:
            top = out.retrieval_scores[0]
            print(f"  top_score={top:.3f} refused={out.refused_for_low_evidence}")
        else:
            print("  (no retrieved docs)")

    # ---- Evals ----
    print("\n[bold]Running RAGAS...[/bold]")
    ragas_metrics = run_ragas(
        [{"question": r["question"], "answer": r["answer"], "contexts": r["contexts"], "ground_truth": r["ground_truth"]} for r in rows],
        settings,
    )

    print("[bold]Running retrieval metrics...[/bold]")
    retrieval_metrics = score_retrieval(settings, rows)

    metrics = {
        "ragas": ragas_metrics,
        "retrieval": {
            "hit_rate_at_k": retrieval_metrics.hit_rate_at_k,
            "mrr_at_k": retrieval_metrics.mrr_at_k,
            "mean_top1_score": retrieval_metrics.mean_top1_score,
        },
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p95": statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else (max(latencies) if latencies else 0.0),
            "max": max(latencies) if latencies else 0.0,
        },
    }

    tag = _now_tag()
    run_path = os.path.join(settings.OUTPUT_DIR, f"run_{tag}.json")
    metrics_path = os.path.join(settings.OUTPUT_DIR, f"metrics_{tag}.json")
    report_path = os.path.join(settings.OUTPUT_DIR, f"report_{tag}.md")

    run_payload = {"rows": rows, "settings": settings.__dict__}
    write_json(run_path, run_payload)
    write_json(metrics_path, metrics)
    write_markdown_report(report_path, run_payload, metrics)

    print(f"\n[bold green]✓ Wrote[/bold green] {run_path}")
    print(f"[bold green]✓ Wrote[/bold green] {metrics_path}")
    print(f"[bold green]✓ Wrote[/bold green] {report_path}\n")

    # Print a tiny console summary
    agg = metrics["ragas"]["aggregate"]
    print("[bold]RAGAS aggregate:[/bold]")
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}")

    print("\n[bold]Retrieval:[/bold]")
    for k, v in metrics["retrieval"].items():
        print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    main()
