"""eval/report.py

Pretty report generation (markdown) + machine-readable JSON metrics.

In production you'd also push these to:
- Prometheus (latency, error rates)
- a data warehouse (per-question scores over time)
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Dict, Any, List

def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def write_markdown_report(path: str, run_payload: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines = []
    lines.append(f"# RAG Eval Report\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")

    # Aggregate
    agg = metrics.get("ragas", {}).get("aggregate", {})
    lines.append("## Aggregate (RAGAS)\n")
    if agg:
        for k, v in agg.items():
            lines.append(f"- **{k}**: {v:.3f}\n")
    else:
        lines.append("- (no aggregate metrics)\n")

    # Retrieval
    r = metrics.get("retrieval", {})
    lines.append("\n## Retrieval (Embedding-proxy)\n")
    for k, v in r.items():
        lines.append(f"- **{k}**: {v:.3f}\n")

    # Latency
    lat = metrics.get("latency_ms", {})
    lines.append("\n## Latency\n")
    for k, v in lat.items():
        lines.append(f"- **{k}**: {v:.1f} ms\n")

    # Per-question details (short)
    lines.append("\n## Per-question (first 10)\n")
    for i, row in enumerate(run_payload.get("rows", [])[:10], 1):
        lines.append(f"\n### Q{i}: {row['question']}\n")
        lines.append(f"**Answer:** {row['answer']}\n")
        # show top 2 retrieved snippets
        chunks = row.get("retrieved_texts", [])[:2]
        for j, c in enumerate(chunks, 1):
            preview = c.replace("\n", " ")[:220]
            lines.append(f"- Chunk {j}: {preview}...\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
