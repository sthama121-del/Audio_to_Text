"""eval/retrieval_eval.py

Retrieval-only metrics (fast, cheap, no LLM judge):

- hit_rate@k: did we retrieve ANY chunk that looks like it contains the answer?
- mrr@k: where did the first good chunk appear? (early is better)

Because we don't have perfect "relevant doc IDs" in this toy dataset,
we use a practical proxy: similarity between each retrieved chunk and the
ground_truth answer (embedding cosine similarity).

INTERVIEW TIP:
This is exactly how many teams bootstrap retrieval evaluation quickly.
Later, you replace it with human labels (relevant/not relevant).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from rag.vectorstore import get_embeddings
from rag.config import Settings

@dataclass(frozen=True)
class RetrievalMetrics:
    hit_rate_at_k: float
    mrr_at_k: float
    mean_top1_score: float

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def score_retrieval(
    settings: Settings,
    results: List[Dict],
    similarity_threshold: float = 0.78,
) -> RetrievalMetrics:
    """results: list of dicts with keys: ground_truth, retrieved_texts, retrieval_scores"""
    emb = get_embeddings(settings)

    hits = []
    rr = []
    top1 = []

    for row in results:
        gt = row["ground_truth"]
        retrieved_texts = row["retrieved_texts"]
        scores = row["retrieval_scores"]
        top1.append(scores[0] if scores else 0.0)

        gt_vec = np.array(emb.embed_query(gt), dtype=np.float32)

        sims = []
        for t in retrieved_texts:
            v = np.array(emb.embed_query(t[:4000]), dtype=np.float32)  # cap for cost
            sims.append(_cos(gt_vec, v))

        # hit if any retrieved chunk is "close enough" to the ground-truth answer
        is_hit = any(s >= similarity_threshold for s in sims)
        hits.append(1.0 if is_hit else 0.0)

        # reciprocal rank of first "good" chunk
        rank = 0
        for i, s in enumerate(sims, 1):
            if s >= similarity_threshold:
                rank = i
                break
        rr.append(1.0 / rank if rank else 0.0)

    return RetrievalMetrics(
        hit_rate_at_k=float(np.mean(hits)) if hits else 0.0,
        mrr_at_k=float(np.mean(rr)) if rr else 0.0,
        mean_top1_score=float(np.mean(top1)) if top1 else 0.0,
    )
