"""rag/llm.py

All model clients live here so swapping providers is easy.

We use:
- OpenAI embeddings (vector space for retrieval)
- Anthropic chat model for answer generation (and optionally judging)

INTERVIEW TIP:
Pluggability is a sign of good engineering.
"""

from __future__ import annotations
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from rag.config import Settings

def get_generator_llm(settings: Settings):
    return ChatAnthropic(
        model=settings.ANTHROPIC_CHAT_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        temperature=0.0,  # deterministic for evals
        max_tokens=600,
    )

def get_judge_llm(settings: Settings):
    if settings.JUDGE_PROVIDER.lower() == "openai":
        return ChatOpenAI(
            model=settings.JUDGE_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )
    # default anthropic
    return ChatAnthropic(
        model=settings.JUDGE_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        temperature=0.0,
        max_tokens=800,
    )
