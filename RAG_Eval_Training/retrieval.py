# =============================================================================
# retrieval.py — Semantic Search & Context Retrieval
# =============================================================================
#
# RESPONSIBILITY: Given a user question, find the most relevant chunks
# from the vector store.
#
# This module is the BRIDGE between ingestion and generation.
# A bad retriever makes a good LLM useless — "garbage in, garbage out."
#
# INTERVIEW GOTCHA: "What is the single most important component of a RAG
# pipeline?" Many interviewers expect the answer: RETRIEVAL.
# The LLM is only as good as the context you give it.
# =============================================================================

from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import config


# ---------------------------------------------------------------------------
# 1. Basic Similarity Retrieval
# ---------------------------------------------------------------------------
def retrieve_chunks(
    vector_store: Chroma,
    query: str,
    k: int = None,
) -> List[Document]:
    """
    Performs a cosine-similarity search and returns the top-K chunks.

    Args:
        vector_store: The populated ChromaDB vector store.
        query: The user's natural language question.
        k: Number of chunks to retrieve. Defaults to config.RETRIEVAL_K.

    Returns:
        A list of Document objects, ordered by descending relevance.

    SCENARIO COVERAGE — Why cosine similarity?
    ─────────────────────────────────────────────
    Cosine similarity measures the ANGLE between two vectors, not their
    magnitude. This is critical for embeddings because:
      - Two semantically similar sentences may have different lengths
        (and thus different vector magnitudes).
      - Cosine ignores magnitude and focuses purely on directional alignment.
    Euclidean distance is an alternative but is sensitive to magnitude —
    it would incorrectly penalize short sentences.

    INTERVIEW GOTCHA: "What if the query is very short (e.g., 'PTO')?"
    Answer: Short queries produce low-information embeddings. The embedding
    of "PTO" is ambiguous — it could mean time off, paid time off, or even
    a PTO (power take-off) in engineering contexts.
    Solution: Query rewriting or expansion. Expand "PTO" to
    "What is the paid time off policy?" before embedding.
    This is called "Hypothetical Document Embedding" (HyDE) in the literature.
    """
    if k is None:
        k = config.RETRIEVAL_K

    if not query or query.strip() == "":
        raise ValueError("Query cannot be empty.")

    # similarity_search returns Documents sorted by relevance (best first)
    results = vector_store.similarity_search(query=query, k=k)

    if not results:
        print("[retrieval] ⚠ No relevant chunks found for this query.")
    else:
        print(f"[retrieval] ✓ Retrieved {len(results)} chunk(s) for query.")

    return results


# ---------------------------------------------------------------------------
# 2. Retrieval with Scores (for Debugging & Evaluation)
# ---------------------------------------------------------------------------
def retrieve_chunks_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = None,
) -> List[Tuple[Document, float]]:
    """
    Same as retrieve_chunks, but also returns the similarity score for each chunk.
    Useful for debugging retrieval quality and understanding WHY a chunk was
    (or wasn't) returned.

    INTERVIEW GOTCHA: "How do you debug a retrieval miss?"
    Answer: Log the scores. If the top score is < 0.5, the query likely has
    no semantic match in your corpus. If scores are clustered (e.g., all ~0.7),
    your chunks may be too similar to each other (chunking is too granular).

    NOTE: ChromaDB returns DISTANCE (L2 by default), not similarity.
    Lower distance = more similar. We convert to a 0-1 score for clarity.
    Score = 1 / (1 + distance)  — this maps [0, ∞) distance to (0, 1] score.
    """
    if k is None:
        k = config.RETRIEVAL_K

    if not query or query.strip() == "":
        raise ValueError("Query cannot be empty.")

    # similarity_search_with_score returns (Document, distance) tuples
    results_with_dist = vector_store.similarity_search_with_score(query=query, k=k)

    # Convert L2 distance to a normalized similarity score
    scored_results = []
    for doc, distance in results_with_dist:
        similarity_score = 1.0 / (1.0 + distance)
        scored_results.append((doc, round(similarity_score, 4)))

    return scored_results


# ---------------------------------------------------------------------------
# 3. Format Context for the LLM Prompt
# ---------------------------------------------------------------------------
def format_context(chunks: List[Document]) -> str:
    """
    Joins retrieved chunks into a single context string for the LLM prompt.

    INTERVIEW GOTCHA: "Does the ORDER of chunks in the context matter?"
    Answer: YES — this is called the "lost in the middle" problem.
    Research (Liu et al., 2023) shows LLMs pay most attention to the FIRST
    and LAST items in a long context. Relevant info buried in the middle
    gets ignored. Best practice: put the highest-scoring chunk FIRST.
    Our retrieve_chunks already returns results sorted best-first, so
    this join preserves that order.

    INTERVIEW GOTCHA: "Should you include chunk metadata in the prompt?"
    Answer: Sometimes. If your chunks have section headers or page numbers
    as metadata, including them helps the LLM produce more precise citations.
    For this training project, we keep it simple. In production, add:
        [Source: Section 3, Page 7]\n{chunk.page_content}
    """
    if not chunks:
        return ""

    # Number each chunk so the LLM can reference them if needed
    numbered_chunks = [
        f"[Chunk {i + 1}]\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    ]

    return "\n\n---\n\n".join(numbered_chunks)
