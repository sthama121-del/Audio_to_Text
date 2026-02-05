"""rag/config.py

Central configuration for the whole project.

INTERVIEW TIP:
- Interviewers love when config is centralized, validated, and easy to tune.
- In production you’d usually load this from env + a config service.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val

@dataclass(frozen=True)
class Settings:
    # -------- Data / storage --------
    PDF_PATH: str = _env("PDF_PATH", "data/HR_POLICY.pdf")  # you can override
    CHROMA_DIR: str = _env("CHROMA_DIR", "chroma_db")
    CHROMA_COLLECTION: str = _env("CHROMA_COLLECTION", "hr_policy")

    # -------- Chunking --------
    # Bigger chunks => fewer calls, but higher risk of mixing topics.
    # Smaller chunks => higher recall, but more tokens + more noise.
    CHUNK_SIZE: int = int(_env("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP: int = int(_env("CHUNK_OVERLAP", "150"))

    # -------- Retrieval --------
    RETRIEVAL_K: int = int(_env("RETRIEVAL_K", "5"))
    # If top similarity is below this, we trigger "I don't know" (hallucination control)
    MIN_RELEVANCE_SCORE: float = float(_env("MIN_RELEVANCE_SCORE", "0.45"))

    # HyDE: generate a "hypothetical answer" to improve retrieval recall (optional)
    ENABLE_HYDE_QUERY_REWRITE: bool = _env("ENABLE_HYDE_QUERY_REWRITE", "false").lower() == "true"

    # -------- Models --------
    OPENAI_API_KEY: str = _env("OPENAI_API_KEY", "") or ""
    OPENAI_EMBEDDING_MODEL: str = _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"

    ANTHROPIC_API_KEY: str = _env("ANTHROPIC_API_KEY", "") or ""
    ANTHROPIC_CHAT_MODEL: str = _env("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-20241022") or "claude-3-5-sonnet-20241022"

    # LLM-as-judge (RAGAS + qualitative checks)
    JUDGE_PROVIDER: str = _env("JUDGE_PROVIDER", "anthropic") or "anthropic"  # anthropic|openai
    JUDGE_MODEL: str = _env("JUDGE_MODEL", "claude-3-5-sonnet-20241022") or "claude-3-5-sonnet-20241022"

    # -------- Runtime / output --------
    OUTPUT_DIR: str = _env("OUTPUT_DIR", "outputs") or "outputs"
    RANDOM_SEED: int = int(_env("RANDOM_SEED", "7"))

def validate(settings: Settings) -> None:
    """Fail-fast config validation (interviewers LOVE this)."""
    missing = []
    if not settings.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY (needed for embeddings)")
    if not settings.ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY (needed for generation/judging)")
    if missing:
        raise RuntimeError(
            "Missing required environment variables:\n- " + "\n- ".join(missing)
        )
