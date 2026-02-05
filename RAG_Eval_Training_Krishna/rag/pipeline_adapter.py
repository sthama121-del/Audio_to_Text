# rag/pipeline_adapter.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from rag.config import Settings


@dataclass
class PipelineResult:
    """Return type expected by eval/ragas_eval.py (attribute access)."""
    question: str
    answer: str
    contexts: List[str]

    # Common extras used by RAG eval scripts
    response: str = ""
    context_metadatas: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)

    raw: Any = None  # optional for debugging


class _RAGPipelineWrapper:
    """
    Compatibility wrapper expected by eval/ragas_eval.py

    Supports:
      - rag.generate(q, k) -> PipelineResult (attribute access)

    Key fix:
      - Ensures we load the real persistent vector store (Chroma) by passing a
        vector_store path into run_one().
      - Builds contexts from retrieved docs (page_content), not from a single
        concatenated `result.context` string that can be empty.
    """

    def __init__(self, settings: Union[Dict, Settings, None] = None):
        self.settings: Settings = self._normalize_settings(settings)

    def generate(self, question: str, k: int = 4) -> PipelineResult:
        return self._run(question, k=k)

    def invoke(self, inp: Any) -> Dict[str, Any]:
        # Optional: keep invoke compatibility
        if isinstance(inp, dict):
            question = (
                inp.get("question")
                or inp.get("query")
                or inp.get("input")
                or inp.get("prompt")
                or ""
            )
        else:
            question = str(inp)

        pr = self._run(question, k=getattr(self.settings, "TOP_K", 4))
        return {
            "question": pr.question,
            "answer": pr.answer,
            "response": pr.response,
            "contexts": pr.contexts,
            "context_metadatas": pr.context_metadatas,
            "retrieval_scores": pr.retrieval_scores,
            "_raw": pr.raw,
        }

    def __call__(self, question: str) -> Dict[str, Any]:
        return self.invoke({"question": question})

    def _pick_vector_store_arg(self) -> Any:
        """
        run_one() supports vector_store as None | str | Chroma.
        We want it to actually load your persistent Chroma DB.
        """
        # Try several common config names; fall back to your repo's default.
        for attr in (
            "VECTOR_STORE_DIR",
            "VECTOR_STORE_PATH",
            "CHROMA_DIR",
            "CHROMA_PATH",
            "CHROMA_PERSIST_DIR",
            "PERSIST_DIR",
            "persist_directory",
        ):
            v = getattr(self.settings, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # Last resort (matches your repo layout)
        return "chroma_db"

    def _run(self, question: str, k: int = 4) -> PipelineResult:
        from rag.rag_pipeline import run_one  # function-based pipeline

        # Best-effort propagate top-k into Settings
        try:
            setattr(self.settings, "TOP_K", int(k))
        except Exception:
            pass
        try:
            setattr(self.settings, "k", int(k))
        except Exception:
            pass

        vector_store_arg = self._pick_vector_store_arg()

        result = run_one(
            question=question,
            vector_store=vector_store_arg,  # ✅ IMPORTANT: do not pass None
            settings=self.settings,
        )

        # From rag_pipeline.RAGResult
        answer = getattr(result, "answer", "") or ""
        retrieved_docs = getattr(result, "retrieved_docs", []) or []
        retrieval_scores = list(getattr(result, "retrieval_scores", []) or [])

        # ✅ Build contexts directly from retrieved docs (best for RAGAS)
        contexts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for d in retrieved_docs:
            if d is None:
                continue

            # LangChain Document
            if hasattr(d, "page_content"):
                txt = getattr(d, "page_content", "") or ""
                md = getattr(d, "metadata", {}) or {}
            else:
                # Custom doc types (best-effort)
                txt = getattr(d, "text", None) or getattr(d, "content", None) or str(d)
                md = getattr(d, "metadata", {}) or {}

            txt = txt.strip()
            if not txt:
                continue

            contexts.append(txt)
            metadatas.append(md)

        # Fallback: if retrieved_docs didn't produce contexts, try `result.context`
        if not contexts:
            context_str = getattr(result, "context", "") or ""
            context_str = context_str.strip()
            if context_str:
                contexts = [context_str]
                metadatas = [{"note": "fallback_from_result.context"}]

        # Keep scores aligned with contexts length (best effort)
        if retrieval_scores and len(retrieval_scores) != len(contexts):
            retrieval_scores = retrieval_scores[: len(contexts)]

        # If no contexts, you likely get "I don't know..." answers (expected)
        return PipelineResult(
            question=question,
            answer=answer,
            response=answer,
            contexts=contexts,
            context_metadatas=metadatas,
            retrieval_scores=retrieval_scores,
            raw=result,
        )

    @staticmethod
    def _normalize_settings(settings: Union[Dict, Settings, None]) -> Settings:
        if isinstance(settings, Settings):
            return settings
        if isinstance(settings, dict):
            try:
                return Settings(**settings)
            except Exception:
                s = Settings()
                for k, v in settings.items():
                    try:
                        setattr(s, k, v)
                    except Exception:
                        pass
                return s
        return Settings()


def build_rag_pipeline(settings: Optional[Union[Dict, Settings]] = None, *args, **kwargs):
    if settings is None:
        settings = {}

    if isinstance(settings, dict) and kwargs:
        settings = {**settings, **kwargs}

    return _RAGPipelineWrapper(settings)
