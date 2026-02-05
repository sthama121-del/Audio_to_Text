# =============================================================================
# main.py — Retrieve + Generate + Evaluate
# =============================================================================
#
# PREREQUISITE: Run ingest.py first if chroma_db/ doesn't exist yet.
#   python ingest.py
#
# WHAT IT DOES:
#   1. Loads the existing vector store from disk (no ingestion)
#   2. Retrieves + Generates answers for all eval questions
#   3. Runs RAGAS evaluation
#   4. Prints the report
# =============================================================================

import os
import sys
from typing import List

import chromadb
from langchain_community.vectorstores import Chroma

import config
from ingestion import get_embedding_model
from retrieval import retrieve_chunks_with_scores, format_context
from generation import generate_answer
from RAG_Eval_Training.evaluation_backup import prepare_eval_dataset, run_ragas_evaluation, score_per_sample
import report


def load_vector_store() -> Chroma:
    if not os.path.exists("./chroma_db"):
        raise FileNotFoundError(
            "Vector store not found. Run 'python ingest.py' first."
        )

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        client=chroma_client,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
    )

    count = chroma_client.get_collection(config.CHROMA_COLLECTION_NAME).count()
    print(f"[pipeline] ✓ Loaded vector store ({count} chunks) from disk")
    return vector_store


def run_retrieve_and_generate(vector_store, eval_questions: List[dict]):
    print("\n" + "=" * 60)
    print(" PHASE 1: RETRIEVAL + GENERATION")
    print("=" * 60)

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(eval_questions, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[pipeline] Processing question {i}/{len(eval_questions)}...")
        print(f"[pipeline] Q: {question[:80]}...")

        scored_chunks = retrieve_chunks_with_scores(vector_store, question)

        print("[pipeline] Retrieval scores:")
        for j, (doc, score) in enumerate(scored_chunks, 1):
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"[pipeline]   Chunk {j}: score={score:.4f} | \"{preview}...\"")

        retrieved_docs = [doc for doc, _ in scored_chunks]
        context_text = format_context(retrieved_docs)
        answer = generate_answer(question, context_text)

        questions.append(question)
        answers.append(answer)
        contexts.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(ground_truth)

    return questions, answers, contexts, ground_truths


def run_evaluation(questions, answers, contexts, ground_truths):
    print("\n" + "=" * 60)
    print(" PHASE 2: RAGAS EVALUATION")
    print("=" * 60)

    eval_dataset = prepare_eval_dataset(questions, answers, contexts, ground_truths)
    aggregate_scores = run_ragas_evaluation(eval_dataset)

    print("\n[evaluation] Running per-sample breakdown...")
    per_sample = score_per_sample(eval_dataset)

    return aggregate_scores, per_sample


def main():
    config.validate_config()

    vector_store = load_vector_store()

    questions, answers, contexts, ground_truths = run_retrieve_and_generate(
        vector_store, config.EVAL_QUESTIONS
    )

    report.print_qa_pairs(questions, answers, contexts)

    aggregate_scores, per_sample_scores = run_evaluation(
        questions, answers, contexts, ground_truths
    )

    print("\n" + "=" * 60)
    print(" PHASE 3: EVALUATION REPORT")
    print("=" * 60)

    report.print_aggregate_report(aggregate_scores)
    report.print_per_question_report(per_sample_scores)
    report.print_interview_prep_summary()

    print("\n[pipeline] ✓ Pipeline complete.")


if __name__ == "__main__":
    main()
