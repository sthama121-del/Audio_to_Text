# RAG Eval Training — Krishna Edition

This repo is a **hands-on interview-training harness** for RAG + RAG Evaluation.

You will:
1) Ingest `data/HR_POLICY.pdf` into a local **Chroma** vector DB using **OpenAI embeddings**
2) Run a **RAG pipeline** (retrieve → augment → generate) using **Anthropic** for generation
3) Run **RAG evaluation** (RAG Triad + retrieval metrics + latency/cost logging)

---

## 0) Prereqs

- Python **3.9+**
- API keys:
  - `OPENAI_API_KEY` (embeddings)
  - `ANTHROPIC_API_KEY` (generation + optional LLM-judge)

> Note: If you see `LibreSSL` warnings on macOS (urllib3), prefer a Python built against OpenSSL:
> - install Python from python.org OR
> - `brew install openssl@3` and use a brew Python.

---

## 1) Setup (recommended)

```bash
cd RAG_Eval_Training_Krishna

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your keys
```

---

## 2) Ingest the PDF → build vector DB

```bash
python ingest.py
```

This creates:
- `./chroma_db/` (persistent local DB)
- `outputs/ingestion_manifest.json` (idempotency + stats)

---

## 3) Run RAG + Eval

```bash
python main.py
```

Outputs:
- `outputs/run_*.json` (Q/A + retrieved chunks + metadata)
- `outputs/report_*.md` (human-readable report)
- `outputs/metrics_*.json` (machine-readable metrics)

---

## 4) What you can tweak quickly (for interviews)

Open `rag/config.py` and try:

- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `RETRIEVAL_K`
- `ENABLE_HYDE_QUERY_REWRITE` (HyDE for better recall)
- `MIN_RELEVANCE_SCORE` (hallucination control: answer only if evidence is strong)
- `JUDGE_PROVIDER` and `JUDGE_MODEL`

---

## Interview-ready talking points

- **RAG Triad**: Faithfulness, Answer Relevance, Context Precision, Context Recall
- **Retrieval metrics**: hit-rate@k, MRR, coverage, “lost-in-the-middle”
- **Production eval**: online monitoring, drift (data/query/model), per-question alerts
- **Debug loop**: inspect retrieved chunks → adjust chunking → adjust retriever → rerun eval

Enjoy — and keep it ruthless: *measure, don’t guess*.
