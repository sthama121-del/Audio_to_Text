# eval/ragas_eval.py
from __future__ import annotations

import json
import os
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from ragas import evaluate

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag.pipeline_adapter import build_rag_pipeline
from eval.embeddings_shim import RagasEmbeddingsShim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "eval_questions.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_ragas_llm() -> Any:
    """
    RAGAS collections metrics require modern InstructorLLM.
    Build it via llm_factory + OpenAI client.
    """
    from ragas.llms import llm_factory
    from openai import OpenAI

    # Requires OPENAI_API_KEY in env
    openai_client = OpenAI()
    model = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
    return llm_factory(model, client=openai_client)


def _build_ragas_embeddings() -> Any:
    """
    Embeddings used by answer_relevancy (and sometimes other metrics).
    Wrap OpenAIEmbeddings with shim adding embed_query().
    Requires OPENAI_API_KEY in env.
    """
    from langchain_openai import OpenAIEmbeddings

    model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-large")
    base = OpenAIEmbeddings(model=model)
    return RagasEmbeddingsShim(base)


def _unwrap_metric(mod_or_obj: Any, attr_candidates: List[str]) -> Any:
    """
    In your install, ragas.metrics.collections.* exports modules.
    This unwraps the class from inside the module.
    """
    if isinstance(mod_or_obj, types.ModuleType):
        for name in attr_candidates:
            if hasattr(mod_or_obj, name):
                return getattr(mod_or_obj, name)
    return mod_or_obj


def _load_metrics(has_reference: bool, llm: Any, embeddings: Any) -> List[Any]:
    """
    Build INITIALISED metric objects explicitly with required args.
    This avoids all the API shape differences (module vs class vs object).
    """
    # Import from collections (preferred)
    from ragas.metrics.collections import (
        faithfulness as faithfulness_mod,
        answer_relevancy as answer_relevancy_mod,
        context_precision as context_precision_mod,
        context_recall as context_recall_mod,
    )

    Faithfulness = _unwrap_metric(faithfulness_mod, ["Faithfulness", "metric", "Metric", "faithfulness"])
    AnswerRelevancy = _unwrap_metric(answer_relevancy_mod, ["AnswerRelevancy", "metric", "Metric", "answer_relevancy"])
    ContextPrecision = _unwrap_metric(context_precision_mod, ["ContextPrecision", "metric", "Metric", "context_precision"])
    ContextRecall = _unwrap_metric(context_recall_mod, ["ContextRecall", "metric", "Metric", "context_recall"])

    # Construct explicitly (this fixes your error)
    metrics: List[Any] = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]

    if has_reference:
        metrics.extend(
            [
                ContextPrecision(llm=llm),
                ContextRecall(llm=llm),
            ]
        )

    # Validate
    for m in metrics:
        if not hasattr(m, "name"):
            raise TypeError(f"Metric did not initialise correctly: {m!r}")

    return metrics


def main() -> None:
    run_dir = OUTPUT_ROOT / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    debug = os.getenv("DEBUG_RAGAS", "0") == "1"

    rows = _read_jsonl(DATA_PATH)
    if not rows:
        raise ValueError(f"No rows found in {DATA_PATH}")

    rag = build_rag_pipeline()

    predictions: List[Dict[str, Any]] = []
    retrieval_contexts: List[Dict[str, Any]] = []

    for r in rows:
        q = r["question"]
        k = int(r.get("k", 4))

        result = rag.generate(q, k=k)

        if debug:
            print("Q:", q)
            print("contexts_len:", len(getattr(result, "contexts", [])))
            print("answer_preview:", (getattr(result, "answer", "") or "")[:100])
            print("metas_len:", len(getattr(result, "context_metadatas", [])))
            print("-" * 60)

        predictions.append({"question": q, "answer": getattr(result, "answer", "")})
        retrieval_contexts.append(
            {
                "question": q,
                "contexts": getattr(result, "contexts", []),
                "context_metadatas": getattr(result, "context_metadatas", []),
            }
        )

    _write_jsonl(run_dir / "predictions.jsonl", predictions)
    _write_jsonl(run_dir / "retrieval_contexts.jsonl", retrieval_contexts)

    dataset_rows: List[Dict[str, Any]] = []
    has_reference = False

    for i, r in enumerate(rows):
        q = r["question"]
        a = predictions[i]["answer"]
        ctxs = retrieval_contexts[i]["contexts"]

        row: Dict[str, Any] = {"question": q, "answer": a, "contexts": ctxs}

        gt = r.get("ground_truth") or r.get("reference") or r.get("ideal_answer")
        if gt and str(gt).strip():
            has_reference = True
            row["ground_truth"] = gt
            row["reference"] = gt

        dataset_rows.append(row)

    ds = Dataset.from_list(dataset_rows)

    # Build judge + embeddings
    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    metrics = _load_metrics(has_reference=has_reference, llm=ragas_llm, embeddings=ragas_embeddings)

    # evaluate (newer versions accept embeddings=; keep both ways safe)
    try:
        results = evaluate(ds, metrics=metrics, embeddings=ragas_embeddings)
    except TypeError:
        for m in metrics:
            if hasattr(m, "embeddings"):
                try:
                    m.embeddings = ragas_embeddings
                except Exception:
                    pass
        results = evaluate(ds, metrics=metrics)

    df = results.to_pandas()
    score_summary = df.mean(numeric_only=True).to_dict()

    with (run_dir / "ragas_scores.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": score_summary,
                "per_question": df.to_dict(orient="records"),
                "metrics_used": [getattr(m, "name", str(m)) for m in metrics],
                "has_reference": has_reference,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    console = Console(record=True, width=120)
    title = Text("RAGAS Evaluation Report", style="bold")
    subtitle = Text(f"Run: {run_dir.name}", style="dim")
    console.print(Panel.fit(Text.assemble(title, "\n", subtitle)))

    info = Text.assemble(
        ("Dataset rows: ", "bold"),
        (str(len(dataset_rows)), ""),
        ("\nReference present: ", "bold"),
        (str(has_reference), ""),
        ("\nMetrics used: ", "bold"),
        (", ".join([getattr(m, "name", str(m)) for m in metrics]), ""),
    )
    console.print(Panel(info, title="Run Info"))

    t = Table(title="Overall Metrics (Mean)")
    t.add_column("Metric", style="bold")
    t.add_column("Score", justify="right")

    for k, v in sorted(score_summary.items()):
        try:
            t.add_row(k, "nan" if pd.isna(v) else f"{float(v):.4f}")
        except Exception:
            t.add_row(k, str(v))
    console.print(t)

    if "faithfulness" in df.columns:
        worst = df.sort_values("faithfulness", ascending=True).head(5)
        wt = Table(title="Lowest Faithfulness Questions (Top 5)")
        wt.add_column("Idx", justify="right")
        wt.add_column("faithfulness", justify="right")
        wt.add_column("question")
        for idx, row in worst.iterrows():
            q = dataset_rows[int(idx)]["question"]
            wt.add_row(str(idx), f"{float(row['faithfulness']):.4f}", q)
        console.print(wt)

    (run_dir / "report.txt").write_text(console.export_text(), encoding="utf-8")
    (run_dir / "report.html").write_text(
        console.export_html(inline_styles=True), encoding="utf-8"
    )

    print(f"\n✅ Done. Outputs at: {run_dir}\n")


if __name__ == "__main__":
    main()
