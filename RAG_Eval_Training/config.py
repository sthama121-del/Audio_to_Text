# =============================================================================
# config.py — Centralized Configuration & Environment Management
# =============================================================================
#
# DESIGN DECISION: All tunables live in ONE place.
# Why? In production, you A/B test chunk sizes, swap models, or adjust
# retrieval K — you need to change ONE file, not hunt through 10 modules.
#
# INTERVIEW GOTCHA: "How do you manage secrets in production?"
# Answer: Environment variables loaded at runtime, NEVER hardcoded.
# In real deployments these come from AWS Secrets Manager or Vault,
# but .env is fine for local/training purposes.
# =============================================================================

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from hr_policy_evaluation_set import evaluation_set as EVAL_QUESTIONS

# Load .env from the project root (where this file lives)
load_dotenv()


# ---------------------------------------------------------------------------
# 1. API Keys — pulled from environment at import time
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")


# ---------------------------------------------------------------------------
# 2. Model Identifiers
# ---------------------------------------------------------------------------
# INTERVIEW GOTCHA: "Why text-embedding-3-small and not 'large'?"
# Answer: 'small' gives 1536-dim vectors at ~10x the throughput and
# lower cost. For a single HR PDF (< 50 chunks), latency & cost
# dominate; accuracy differences are negligible at this scale.
# 'large' (3072-dim) only pays off on massive, diverse corpora.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "claude-sonnet-4-5-20250929")


# ---------------------------------------------------------------------------
# 3. Chunking Parameters (Ingestion Tuning)
# ---------------------------------------------------------------------------
# INTERVIEW GOTCHA: "How do you pick chunk_size?"
# Rule of thumb: chunk_size should fit comfortably inside the embedding
# model's token window (text-embedding-3-small supports 8191 tokens).
# 800 chars ≈ 200 tokens — well under the limit, leaving room for
# overlap and metadata.
#
# INTERVIEW GOTCHA: "Why overlap?"
# Answer: Sentences that span chunk boundaries would otherwise be
# split and lost. 10-15% overlap is the industry sweet spot.
# Too much overlap → bloated index, redundant retrieval.
# Too little → broken context at boundaries.
CHUNK_SIZE: int = 800          # Characters per chunk
CHUNK_OVERLAP: int = 100       # ~12.5% overlap — industry standard


# ---------------------------------------------------------------------------
# 4. Retrieval Parameters
# ---------------------------------------------------------------------------
# INTERVIEW GOTCHA: "Why K=3 and not K=5 or K=10?"
# Answer: More chunks = more noise in the context window, which
# INCREASES hallucination risk (see Faithfulness metric).
# K=3 is the sweet spot for focused document QA. Bump to 5 only if
# your questions provably need multi-chunk synthesis.
RETRIEVAL_K: int = 5


# ---------------------------------------------------------------------------
# 5. ChromaDB Settings
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME: str = "hr_policy_docs"
# Using in-memory for training/interview purposes.
# Production swap: replace with PersistentClient(path="./chroma_db")
CHROMA_PERSIST: bool = False


# ---------------------------------------------------------------------------
# 6. Evaluation Test Questions (Golden Dataset)
# ---------------------------------------------------------------------------
# INTERVIEW GOTCHA: "Where do ground-truth answers come from?"
# Answer: In production, a domain expert (e.g., HR manager) writes
# them. For training, we author them manually from known document content.
# Using LLM-generated ground truths is circular and biases scores UP.
#
# 10 questions across 4 difficulty tiers:
#   Tier 1 (Q1–Q3)  → Simple Factual Lookup
#   Tier 2 (Q4–Q6)  → Multi-Detail Extraction
#   Tier 3 (Q7–Q8)  → Cross-Section Synthesis
#   Tier 4 (Q9–Q10) → Out-of-Scope Detection

# ---------------------------------------------------------------------------
# 7. Validation helper
# ---------------------------------------------------------------------------
def validate_config() -> None:
    """
    Called at startup. Fails fast if critical env vars are missing.
    INTERVIEW GOTCHA: "How do you prevent silent failures from missing keys?"
    Answer: Validate at startup, not at first use. Fail loud, fail early.
    """
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            f"Copy .env.example to .env and fill in your keys."
        )
    print("[config] ✓ All required environment variables are set.")
