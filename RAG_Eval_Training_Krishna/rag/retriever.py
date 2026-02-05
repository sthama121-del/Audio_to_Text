# rag/retriever.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document

# Chroma import (newer / fallback)
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma  # type: ignore

from langchain_openai import OpenAIEmbeddings


# ----------------------------
# Data model returned by retrieve()
# ----------------------------
@dataclass
class Retrieved:
    docs: List[Document]
    scores: List[float]      # normalized to [0,1], higher = more relevant
    query_used: str          # <-- IMPORTANT: fixes AttributeError in rag_pipeline.py


# ----------------------------
# Settings helpers
# ----------------------------
def _get_setting(settings: Any, key: str, default: Any = None) -> Any:
    """Support Settings object (attr) or dict settings."""
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(key, default)
    return getattr(settings, key, default)


def _ensure_embeddings(settings: Any) -> OpenAIEmbeddings:
    model = (
        _get_setting(settings, "EMBEDDING_MODEL", None)
        or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    return OpenAIEmbeddings(model=model)


def _ensure_vectorstore(vs: Any, settings: Any) -> Chroma:
    """
    vs may be:
      - None  -> build from env/settings
      - str   -> treat as persist_directory
      - Chroma-> use as-is
    """
    if isinstance(vs, Chroma):
        return vs

    embeddings = _ensure_embeddings(settings)

    collection = (
        _get_setting(settings, "CHROMA_COLLECTION", None)
        or os.getenv("CHROMA_COLLECTION", "rag")
    )
    persist_dir = (
        _get_setting(settings, "CHROMA_PERSIST_DIR", None)
        or os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    )

    if isinstance(vs, str) and vs.strip():
        persist_dir = vs.strip()

    return Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


# ----------------------------
# Score normalization
# ----------------------------
def _normalize_distance_to_relevance(dist: float) -> float:
    """
    similarity_search_with_score() returns a distance score.
    For cosine distance, lower is better (0 is perfect).

    We map distance -> relevance in [0,1].
    Heuristic:
      relevance = 1 / (1 + dist)
    This is monotonic and stable even if dist isn't strictly bounded.
    """
    if dist is None:
        return 0.0
    try:
        d = float(dist)
    except Exception:
        return 0.0
    if math.isnan(d) or math.isinf(d):
        return 0.0
    # clamp negatives to 0 just in case
    d = max(d, 0.0)
    rel = 1.0 / (1.0 + d)
    # hard clamp
    if rel < 0.0:
        return 0.0
    if rel > 1.0:
        return 1.0
    return rel


# ----------------------------
# Public API used by rag_pipeline.py
# ----------------------------
def retrieve(
    vs: Any,
    query: str,
    settings: Any,
    rewritten_query: Optional[str] = None,
) -> Retrieved:
    """
    Function-based retriever used by rag.rag_pipeline.

    Returns:
      Retrieved(
        docs=[Document,...],
        scores=[0..1,...],
        query_used="..."
      )
    """
    if not isinstance(query, str):
        query = str(query)

    query_used = (rewritten_query or query).strip()

    # top-k
    k = _get_setting(settings, "TOP_K", None)
    if k is None:
        k = _get_setting(settings, "k", 5)
    try:
        k = int(k)
    except Exception:
        k = 5
    k = max(1, k)

    store = _ensure_vectorstore(vs, settings)

    # Returns List[Tuple[Document, float]] where float is "distance"
    pairs: List[Tuple[Document, float]] = store.similarity_search_with_score(
        query_used, k=k
    )

    docs: List[Document] = []
    scores: List[float] = []

    for doc, dist in pairs:
        docs.append(doc)
        scores.append(_normalize_distance_to_relevance(dist))

    return Retrieved(docs=docs, scores=scores, query_used=query_used)


def format_context(
    retrieved_or_docs: Any,
    *,
    max_docs: int | None = None,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    Optional helper for pipeline prompting.

    Accepts:
      - Retrieved (has .docs)
      - list[Document]
      - list[dict]
    """
    docs = None

    if hasattr(retrieved_or_docs, "docs"):
        docs = getattr(retrieved_or_docs, "docs")
    elif isinstance(retrieved_or_docs, list):
        docs = retrieved_or_docs
    elif isinstance(retrieved_or_docs, dict):
        docs = retrieved_or_docs.get("docs")

    docs = docs or []
    if max_docs is not None:
        docs = docs[:max_docs]

    parts: List[str] = []
    for d in docs:
        text = getattr(d, "page_content", None)
        if text is None and isinstance(d, dict):
            text = d.get("page_content") or d.get("content") or d.get("text")
        if text is None:
            text = str(d)
        parts.append(text)

    return separator.join(parts)
