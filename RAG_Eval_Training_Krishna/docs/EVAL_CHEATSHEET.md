# RAG Evaluation Cheat Sheet (Interview-ready)

## 1) Retrieval failures (can't find the right evidence)
Symptoms:
- Low context_recall: missing relevant chunks
- Low context_precision: retrieving noisy/irrelevant chunks

Fixes:
- chunking: size/overlap, semantic chunking
- retriever: top_k, MMR, metadata filters
- query: rewrite / HyDE, add reranker

## 2) Generation failures (evidence exists but answer is wrong)
Symptoms:
- Low faithfulness: hallucinations / unsupported claims
- Low answer_relevancy: off-topic / rambling

Fixes:
- prompt: "answer ONLY from context", enforce citations
- temperature: lower
- refusal: "I don't know" if context missing
- format: bullet/short answer templates

## 3) Why component-level eval?
End-to-end output-only evaluation hides whether the failure is in retrieval or generation.
Component metrics show where to tune first.
