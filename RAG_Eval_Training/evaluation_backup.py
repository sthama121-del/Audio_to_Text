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
# From your research notes, the landscape includes:
#   - RAGAS:      Best for pipeline testing & benchmarking (we use this)
#   - DeepEval:   Best for CI/CD integration (pytest-style)
#   - TruLens:    Best for production monitoring & iterative feedback
#   - Deepchecks: Best for continuous monitoring in production
#   - Future AGI: Best for workflow debugging & agent tracing
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

import config


# ---------------------------------------------------------------------------
# 1. Prepare Evaluation Dataset
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
        (Is the answer supported by what we retrieved?)
      - Answer Relevance: Compares answer vs. question only. No ground truth needed.
        (Does the answer address what was asked?)
      - Context Precision: Compares contexts vs. ground truth.
        (Were the RIGHT chunks retrieved?)
      - Context Recall: Compares contexts vs. ground truth.
        (Were ALL needed chunks retrieved?)
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
# 2. Run RAGAS Evaluation
# ---------------------------------------------------------------------------
def run_ragas_evaluation(eval_dataset: Dataset) -> Dict[str, float]:
    """
    Executes the full RAG Triad evaluation using RAGAS.

    Returns a dictionary of metric_name → score (0.0 to 1.0).

    SCENARIO COVERAGE — What do the scores MEAN?
    ────────────────────────────────────────────
    All RAGAS metrics return a score between 0.0 and 1.0:

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

    NOTE ON LLM JUDGE: RAGAS uses an LLM internally to judge faithfulness
    and relevance. By default it uses OpenAI's GPT-4. You can configure it
    to use a different model if needed (see RAGAS docs for custom LLM config).
    """
    print("\n[evaluation] Running RAGAS evaluation (this calls the LLM judge)...")
    print("[evaluation] Metrics: Faithfulness, Answer Relevance, "
          "Context Precision, Context Recall")

    # Define which metrics to compute
    metrics = [
        faithfulness,        # Answer grounded in context?
        answer_relevancy,    # Answer addresses the question?
        context_precision,   # Retrieved chunks are relevant?
        context_recall,      # All needed chunks retrieved?
    ]

    try:
        # Run the evaluation
        # RAGAS's evaluate() returns a pandas-like result object
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
        )

        scores = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric_name in results._scores_dict:
                val = results._repr_dict.get(metric_name, 0.0)
                scores[metric_name] = float(val) if val == val else 0.0  # NaN check
                if val != val:
                    print(f"[evaluation] ⚠ {metric_name}: returned NaN (embed_query issue). Defaulting to 0.0.")
            else:
                scores[metric_name] = 0.0

        print(f"[evaluation] ✓ Evaluation complete.")
        return scores

    except Exception as e:
        print(f"[evaluation] ✗ RAGAS evaluation failed: {e}")
        print("[evaluation] Falling back to manual scoring...")
        # Return a structured error so the pipeline doesn't crash
        raise RuntimeError(
            f"RAGAS evaluation failed: {e}\n"
            f"Common causes: missing API key for LLM judge, "
            f"network timeout, or malformed dataset."
        )


# ---------------------------------------------------------------------------
# 3. Per-Sample Scoring (Granular Debugging)
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
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )

    # Convert to per-row breakdown
    per_sample = []
    for i in range(len(eval_dataset)):
        sample_scores = {"question": eval_dataset[i]["question"]}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = results[metric_name][i] if metric_name in results._scores_dict else 0.0
            sample_scores[metric_name] = float(val) if val == val else 0.0  # NaN check

        per_sample.append(sample_scores)

    return per_sample
