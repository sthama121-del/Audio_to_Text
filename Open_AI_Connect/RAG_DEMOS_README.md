# RAG Demos

Two versions to show ers:

## 1. Minimal RAG Demo (~50 lines)
**File:** `minimal_rag_demo.py`

**What it shows:**
- Core RAG concept: Retrieve → Augment → Generate
- Simple keyword-based retrieval (no embeddings)
- Perfect for explaining RAG

**Run:**
```bash
cd /Users/Sri/Documents/Audio_to_Text/Open_AI_Connect
# Edit line 11: add your OpenAI API key
python minimal_rag_demo.py
```

## 2. Production RAG Demo (~80 lines)
**File:** `production_rag_demo.py`

**What it shows:**
- Real vector embeddings (text-embedding-3-small)
- Cosine similarity search
- In-memory vector store
- Semantic search (not keyword matching)

**Run:**
```bash
cd /Users/Sri/Documents/Audio_to_Text/Open_AI_Connect
# Edit line 12: add your OpenAI API key
python production_rag_demo.py
```

## Setup

1. **Navigate to directory:**
```bash
cd /Users/Sri/Documents/Audio_to_Text/Open_AI_Connect
```

2. **Download both demo files** from outputs to this directory

3. **Add your API key:**
Edit line with `'your-key-here'` in both files

4. **Install dependencies:**
```bash
pip install openai numpy
```

5. **Run:**
```bash
python minimal_rag_demo.py        # Simple version
python production_rag_demo.py     # With embeddings
```

##  Talking Points

### Minimal Demo (50 lines)
- "This shows the core RAG concept in the simplest form"
- "Retrieval is keyword-based for clarity, not production-ready"
- "Shows how context prevents hallucination"
- "Notice the 'I don't know' for out-of-scope questions"

### Production Demo (80 lines)
- "This uses real embeddings for semantic search"
- "Vector similarity finds relevant chunks, not just keywords"
- "This is how production RAG systems work"
- "The vector store would be ChromaDB/Pinecone in production"

## Key  Questions to Prepare

**Q: Why RAG instead of fine-tuning?**
A: RAG is dynamic (update knowledge without retraining), cheaper, and more transparent.

**Q: What's the biggest challenge in RAG?**
A: Retrieval quality. If you retrieve wrong chunks, the LLM has no chance.

**Q: How do you evaluate RAG systems?**
A: Faithfulness (no hallucinations), Answer Relevance, Context Precision/Recall

**Q: What's the difference between this and your full project?**
A: This is a proof of concept. The full project adds:
- PDF ingestion and chunking
- Persistent vector store (ChromaDB)
- Evaluation framework (RAGAS)
- Web interface
- Production error handling

## Expected Output

### Minimal Demo:
```
Q: How many vacation days do I get?
A: 24 days of paid vacation per year.

Q: What's the remote work policy?
A: Remote work is allowed 3 days per week with manager approval.

Q: What's the stock price?
A: I don't know.
```

### Production Demo:
```
Building vector store...
✓ Embedded 4 documents

Q: How many vacation days do I get?
A: You get 24 days of paid vacation per year.

Q: Can I work from home?
A: Yes, remote work is allowed 3 days per week with manager approval.

Q: What's the stock price?
A: I don't know.
```

## Time Required

- **Minimal demo:** 2-3 minutes to explain and run
- **Production demo:** 5 minutes (includes embedding time)
- Perfect for 30-minute technical s
