# =============================================================================
# evaluation.py — RAG Triad Evaluation Suite (using RAGAS)
# =============================================================================
#
# RESPONSIBILITY: Score the RAG pipeline's output across 4 key dimensions
# that form the "RAG Evaluation Triad" — the standard interviewers look for.
#
# THE RAG TRIAD (+ Context Recall):
# ┌─────────────────────────────────────────────────────────────────────┐
# │  1. FAITHFULNESS        — Is the answer grounded in the context?    │
# │     (No hallucinations. Every claim in the answer must be            │
# │      traceable to a retrieved chunk.)                                │
# │                                                                     │
# │  2. ANSWER RELEVANCE    — Does the answer actually address the      │
# │     question asked? (A correct but off-topic answer scores low.)    │
# │                                                                     │
# │  3. CONTEXT PRECISION   — Are the retrieved chunks relevant to      │
# │     answering the question? (Measures retrieval QUALITY.)           │
# │                                                                     │
# │  4. CONTEXT RECALL      — Did we retrieve ALL the chunks needed     │
# │     to answer the question? (Measures retrieval COMPLETENESS.)      │
# │     Requires ground_truth to compute.                               │
# └─────────────────────────────────────────────────────────────────────┘
#
# WHY RAGAS OVER OTHER FRAMEWORKS?
# ─────────────────────────────────
#   - RAGAS:      Best for pipeline testing & benchmarking (we use this)
#   - DeepEval:   Best for CI/CD integration (pytest-style)
#   - TruLens:    Best for production monitoring & iterative feedback
#   - Deepchecks: Best for continuous monitoring in production
#
# RAGAS is the right choice HERE because:
#   a) It implements the exact Triad metrics interviewers expect
#   b) It's LLM-based evaluation (uses an LLM judge), so scores are
#      semantically meaningful, not just lexical matches
#   c) It works offline — no external dashboard required
#
# INTERVIEW GOTCHA: "What's the difference between Faithfulness and Groundedness?"
# Answer: They're often used interchangeably, but technically:
#   - Faithfulness: The answer contains NO claims not supported by context
#   - Groundedness: Every claim in the answer CAN BE TRACED to a source
# Groundedness is stricter (requires explicit traceability).
# RAGAS's Faithfulness metric is closer to groundedness in practice.
#
# INTERVIEW GOTCHA: "Why do these metrics use an LLM as a judge?"
# Answer: Traditional metrics like BLEU/ROUGE measure LEXICAL overlap
# (word-for-word matching). They fail badly for paraphrases.
# "Employees get 15 vacation days" and "Staff are entitled to 15 days off"
# are semantically identical but lexically different — BLEU scores them low.
# LLM-as-judge understands semantics, making it far more accurate for QA.
# =============================================================================

from typing import List, Dict, Any
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ---------------------------------------------------------------------------
# FIX: Manually wrap langchain OpenAIEmbeddings and assign to each metric
# BEFORE calling evaluate().
#
# WHY THE PREVIOUS FIXES FAILED:
#   Fix attempt 1: Passed embeddings=embedding_factory(..., client=client)
#     into evaluate(). This created a MODERN BaseRagasEmbedding object which
#     has embed_text() but NOT embed_query(). However, the AnswerRelevancy
#     metric's calculate_similarity() method hardcodes a call to
#     self.embeddings.embed_query(). So the modern object crashes there.
#
#   Fix attempt 2: Same as above — embedding_factory with a client always
#     returns the modern interface. Same crash.
#
# WHY THIS FIX WORKS:
#   RAGAS's evaluate() only sets embeddings on a metric if
#   metric.embeddings is None. By assigning embeddings DIRECTLY onto each
#   metric object before calling evaluate(), we skip that entire code path.
#   We use LangchainEmbeddingsWrapper which exposes embed_query() and
#   embed_documents() — exactly what calculate_similarity() expects.
#   The underlying langchain OpenAIEmbeddings DOES have embed_query() —
#   the bug was never in langchain, it was in which wrapper RAGAS used.
#
# INTERVIEW GOTCHA: "Why set embeddings on the metric directly instead of
# passing it to evaluate()?"
# Answer: evaluate() propagates embeddings to metrics, but only if the
# metric.embeddings is None. More importantly, evaluate()'s auto-creation
# path can return the wrong type. Setting it directly is explicit and
# immune to RAGAS version changes — a key production principle.
# ---------------------------------------------------------------------------
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

import config


# ---------------------------------------------------------------------------
# 1. Build and assign embeddings to metrics (called once before evaluate)
# ---------------------------------------------------------------------------
def _build_metrics_with_embeddings() -> list:
    """
    Creates fresh metric instances with embeddings pre-assigned.

    We create NEW metric instances each time (not reusing the module-level
    singletons like `answer_relevancy`) because RAGAS metrics are stateful —
    once embeddings are set, they persist. Reusing singletons across runs
    could cause subtle bugs if config changes between runs.

    The model matches config.EMBEDDING_MODEL (text-embedding-3-small) so
    query embeddings live in the same vector space as the stored chunks.
    """
    # Create the langchain embeddings object — this one HAS embed_query()
    lc_embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )

    # Wrap it with RAGAS's LangchainEmbeddingsWrapper so RAGAS can use it
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    # Import the actual metric CLASSES (not the singleton instances)
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall

    # Create fresh instances and assign embeddings directly
    metrics = [
        Faithfulness(),
        AnswerRelevancy(embeddings=ragas_embeddings),   # ← needs embeddings
        ContextPrecision(),
        ContextRecall(),
    ]

    return metrics


# ---------------------------------------------------------------------------
# 2. Prepare Evaluation Dataset
# ---------------------------------------------------------------------------
def prepare_eval_dataset(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dataset:
    """
    Converts our evaluation data into the format RAGAS expects.

    RAGAS requires a HuggingFace Dataset with these exact column names:
      - "question"      : The user's question (str)
      - "answer"        : The LLM's generated answer (str)
      - "contexts"      : List of retrieved chunk texts (List[str])
      - "ground_truth"  : The expected correct answer (str)

    INTERVIEW GOTCHA: "Why does RAGAS need ground_truth for some metrics
    but not others?"
    Answer: It depends on what's being measured:
      - Faithfulness: Compares answer vs. contexts only. No ground truth needed.
      - Answer Relevance: Compares answer vs. question only. No ground truth needed.
      - Context Precision: Compares contexts vs. ground truth.
      - Context Recall: Compares contexts vs. ground truth.
    Ground truth is the "oracle" — it tells us what the CORRECT answer is
    so we can judge whether retrieval found the right evidence.
    """
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    dataset = Dataset.from_dict(eval_data)
    print(f"[evaluation] ✓ Prepared evaluation dataset with {len(dataset)} samples")
    return dataset


# ---------------------------------------------------------------------------
# 3. Run RAGAS Evaluation
# ---------------------------------------------------------------------------
def run_ragas_evaluation(eval_dataset: Dataset) -> Dict[str, float]:
    """
    Executes the full RAG Triad evaluation using RAGAS.

    Returns a dictionary of metric_name → score (0.0 to 1.0).

    SCENARIO COVERAGE — What do the scores MEAN?
    ────────────────────────────────────────────
    │ Score Range │ Interpretation                                    │
    │ 0.9 – 1.0   │ Excellent. Production-ready.                      │
    │ 0.7 – 0.9   │ Good. Minor issues to investigate.                │
    │ 0.5 – 0.7   │ Moderate. Needs improvement in this dimension.    │
    │ < 0.5       │ Poor. Significant pipeline issues.                │

    INTERVIEW GOTCHA: "A candidate gets Faithfulness=0.95 but
    Answer Relevance=0.6. What does that tell you?"
    Answer: The LLM is faithfully quoting the context, but it's answering
    the WRONG question. Likely cause: retrieval returned irrelevant chunks,
    and the LLM faithfully summarized them. Fix: improve retrieval, not generation.

    INTERVIEW GOTCHA: "Context Precision=0.9 but Context Recall=0.5. Why?"
    Answer: We're retrieving HIGH-QUALITY chunks (precision is good), but
    we're MISSING chunks (recall is low). Likely causes:
      a) K is too small — increase RETRIEVAL_K
      b) The answer requires info spread across chunks that are too far
         apart in embedding space — try query rewriting or multi-step retrieval.
    """
    print("\n[evaluation] Running RAGAS evaluation (this calls the LLM judge)...")
    print("[evaluation] Metrics: Faithfulness, Answer Relevance, "
          "Context Precision, Context Recall")

    # Build fresh metrics with embeddings pre-assigned.
    # This is the key fix — we never let evaluate() create embeddings itself.
    metrics = _build_metrics_with_embeddings()

    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
        )

        # Extract scores safely
        scores = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            try:
                val = results[metric_name]
                scores[metric_name] = float(val) if val == val else 0.0  # NaN check
                if val != val:
                    print(f"[evaluation] ⚠ {metric_name}: returned NaN. Defaulting to 0.0.")
            except (KeyError, TypeError):
                scores[metric_name] = 0.0
                print(f"[evaluation] ⚠ {metric_name}: not found in results. Defaulting to 0.0.")

        print("[evaluation] ✓ Evaluation complete.")
        return scores

    except Exception as e:
        print(f"[evaluation] ✗ RAGAS evaluation failed: {e}")
        raise RuntimeError(
            f"RAGAS evaluation failed: {e}\n"
            f"Common causes: missing API key for LLM judge, "
            f"network timeout, or malformed dataset."
        )


# ---------------------------------------------------------------------------
# 4. Per-Sample Scoring (Granular Debugging)
# ---------------------------------------------------------------------------
def score_per_sample(eval_dataset: Dataset) -> List[Dict[str, Any]]:
    """
    Runs evaluation and returns PER-QUESTION scores (not just averages).

    This is critical for debugging — aggregate scores hide which specific
    questions are failing. In production, you'd alert on individual
    question scores dropping below a threshold.

    INTERVIEW GOTCHA: "How do you monitor RAG quality in production?"
    Answer: Per-question scoring + threshold alerts. If a specific question
    consistently scores < 0.5 on faithfulness, that's a signal that either:
      a) The relevant chunk was deleted/modified (data drift)
      b) The question pattern changed (query drift)
      c) The model regressed (model drift)
    """
    # Fresh metrics again — don't reuse from the aggregate run
    metrics = _build_metrics_with_embeddings()

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )

    # Convert to per-row breakdown
    per_sample = []
    for i in range(len(eval_dataset)):
        sample_scores = {"question": eval_dataset[i]["question"]}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            try:
                val = results[metric_name][i]
                sample_scores[metric_name] = float(val) if val == val else 0.0
            except (KeyError, IndexError, TypeError):
                sample_scores[metric_name] = 0.0

        per_sample.append(sample_scores)

    return per_sample
