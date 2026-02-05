# RAG Evaluation Training Project
**End-to-End RAG Pipeline with Full Triad Evaluation Suite**

A production-grade, modular RAG (Retrieval-Augmented Generation) pipeline built for both technical interview preparation and real deployment learning. Uses ChromaDB, OpenAI embeddings, Anthropic Claude for generation, and the RAGAS framework for evaluation.

---

## Project Structure

```
RAG_Eval_Training/
├── .env.example        ← Copy to .env, fill in your API keys
├── requirements.txt    ← All Python dependencies
├── config.py           ← ALL tunables in one place (chunk size, K, models)
├── hr_policy.txt       ← Synthetic HR document (the knowledge base)
├── ingestion.py        ← Load → Chunk → Embed → Store
├── retrieval.py        ← Semantic search + context formatting
├── generation.py       ← Claude API call with grounded prompting
├── evaluation.py       ← RAGAS Triad: Faithfulness, Relevance, Precision, Recall
├── report.py           ← Rich terminal report rendering
├── main.py             ← Master orchestrator (run this)
└── README.md           ← This file
```

---

## Quick Start

```bash
# 1. Navigate to project
cd /Users/Sri/Documents/Audio_to_Text/RAG_Eval_Training

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and ANTHROPIC_API_KEY

# 5. Run the pipeline
python main.py
```

---

## What Runs When You Execute `main.py`

The pipeline executes in 4 sequential phases:

**Phase 1 — Ingestion:** `hr_policy.txt` is loaded, split into overlapping chunks using `RecursiveCharacterTextSplitter` (800 chars, 100 char overlap), embedded via `text-embedding-3-small`, and stored in an in-memory ChromaDB collection.

**Phase 2 — Retrieval + Generation:** For each of 4 evaluation questions, the pipeline retrieves the top-3 most semantically similar chunks (with similarity scores logged for debugging), formats them into a context block, and sends the question + context to Claude Sonnet via the Anthropic API. The system prompt explicitly constrains the model to answer only from the provided context.

**Phase 3 — Evaluation:** The RAGAS framework scores the pipeline across the RAG Triad (Faithfulness, Answer Relevance, Context Precision, Context Recall) using an LLM judge. Both aggregate and per-question scores are computed.

**Phase 4 — Report:** A color-coded terminal report displays all scores, Q&A pairs with retrieved context previews, and an interview prep summary card.

---

## The 4 Evaluation Questions (and Why)

| # | Question | Tests |
|---|----------|-------|
| 1 | Remote work policy details | Factual recall — specific numbers and rules |
| 2 | Current stock price | Out-of-scope detection — the "I don't know" handler |
| 3 | Disciplinary escalation steps | Multi-step reasoning — requires synthesizing across sections |
| 4 | Parental leave benefits | Edge case — multiple benefit types in one answer |

---

## Key Interview Topics Covered (in the code comments)

Every module contains block comments explaining the *why* behind each technical decision. Here's a summary of what interviewers typically ask about:

**Chunking:** Why `RecursiveCharacterTextSplitter`? How to pick `chunk_size` and `overlap`? What is embedding drift and how does overlap mitigate it?

**Embeddings:** Why `text-embedding-3-small` over `large`? What are the dimensionality and cost tradeoffs? Should you create a new client per request?

**Retrieval:** Why cosine similarity? What is the "lost in the middle" problem? How do you debug a retrieval miss? What is HyDE (Hypothetical Document Embedding)?

**Generation:** How do you prevent hallucination via prompt engineering? Why constrain the LLM with a system prompt? What's the "I don't know" escape hatch pattern?

**Evaluation:** What is the RAG Triad? Why LLM-as-judge over BLEU/ROUGE? When do you need ground truth and when don't you? How do you interpret score combinations (e.g., high faithfulness + low relevance)?

**Frameworks:** RAGAS vs DeepEval vs TruLens — what each is best for and when to use which.

---

## Swapping Components (Production Readiness)

| Want to change... | Edit this file | What to change |
|---|---|---|
| Vector store (FAISS) | `ingestion.py` | Swap `chromadb.Client()` for FAISS index |
| Embedding model | `config.py` | Change `EMBEDDING_MODEL` |
| Generation model | `config.py` | Change `GENERATION_MODEL` |
| Evaluation framework | `evaluation.py` | Swap RAGAS calls for DeepEval |
| Source document | `main.py` | Change `hr_policy_path` to your PDF |
| Chunk strategy | `config.py` | Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` |

---

## API Keys Required

- **OpenAI** (`OPENAI_API_KEY`) — for `text-embedding-3-small` embeddings and RAGAS's LLM judge
- **Anthropic** (`ANTHROPIC_API_KEY`) — for Claude Sonnet answer generation
