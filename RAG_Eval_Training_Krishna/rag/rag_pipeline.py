"""rag/rag_pipeline.py

End-to-end RAG logic:
1) (Optional) query rewrite (HyDE) for better recall
2) retrieve top-k chunks
3) decide whether we have enough evidence
4) generate answer grounded in context with citations
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from rag.config import Settings
from rag.retriever import retrieve, Retrieved  # NOTE: no format_context import
from rag.llm import get_generator_llm

SYSTEM_PROMPT = """You are a careful HR policy assistant.
Rules:
- Use ONLY the provided context. If the answer is not in context, say "I don't know based on the provided policy.".
- Be concise and specific.
- When you state a fact, cite the chunk like [Chunk 2].
"""

HYDE_PROMPT = """Write a short hypothetical answer to the user question.
This is used ONLY for retrieval (HyDE). Do NOT add extra unrelated content.
Question: {question}
Hypothetical answer:"""


@dataclass
class RAGResult:
    question: str
    query_used: str
    retrieved_docs: List[Document]
    retrieval_scores: List[float]
    context: str
    answer: str
    latency_ms: float
    refused_for_low_evidence: bool


def maybe_hyde_rewrite(question: str, settings: Settings) -> Optional[str]:
    if not settings.ENABLE_HYDE_QUERY_REWRITE:
        return None

    llm = get_generator_llm(settings)
    msg = HumanMessage(content=HYDE_PROMPT.format(question=question))
    resp = llm.invoke([msg])
    return str(resp.content).strip()


def generate_answer(question: str, context: str, settings: Settings) -> str:
    llm = get_generator_llm(settings)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
    ]
    resp = llm.invoke(messages)
    return str(resp.content).strip()


def run_one(
    question: str,
    vector_store: Any = None,
    settings: Optional[Settings] = None,
) -> RAGResult:
    """
    Function-based pipeline entrypoint.

    Compatibility:
      - vector_store can be None (retriever coerces it based on env/settings)
      - settings can be None (we construct Settings() which typically reads env)
    """
    if settings is None:
        settings = Settings()

    t0 = time.time()

    rewritten = maybe_hyde_rewrite(question, settings)
    ret: Retrieved = retrieve(vector_store, question, settings, rewritten_query=rewritten)

    # NOTE: we format from ret.docs, NOT importing format_context from retriever.py
    context = format_context(ret.docs)

    # Evidence gate: if nothing is relevant enough, force refusal.
    top_score = ret.scores[0] if ret.scores else 0.0
    refused = top_score < settings.MIN_RELEVANCE_SCORE

    if refused:
        answer = "I don't know based on the provided policy."
    else:
        answer = generate_answer(question, context, settings)

    latency_ms = (time.time() - t0) * 1000.0
    return RAGResult(
        question=question,
        query_used=ret.query_used,
        retrieved_docs=ret.docs,
        retrieval_scores=ret.scores,
        context=context,
        answer=answer,
        latency_ms=latency_ms,
        refused_for_low_evidence=refused,
    )


def format_context(
    docs: Any,
    *,
    max_docs: int | None = None,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    Local context formatter.

    Accepts:
      - list[Document]
      - list[dict]
      - anything iterable-ish

    Returns a single string context for prompting.
    """
    if docs is None:
        docs = []

    # If someone passes a Retrieved-like object, accept it
    if hasattr(docs, "docs"):
        docs = getattr(docs, "docs")

    if max_docs is not None:
        docs = list(docs)[:max_docs]

    parts: list[str] = []
    for d in docs:
        text = getattr(d, "page_content", None)
        if text is None and isinstance(d, dict):
            text = d.get("page_content") or d.get("content") or d.get("text")
        if text is None:
            text = str(d)
        parts.append(text)

    return separator.join(parts)
