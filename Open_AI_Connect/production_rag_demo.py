"""
Production-Ready RAG Demo (with real embeddings)
Shows proper vector search instead of keyword matching
~80 lines with embeddings
"""

from openai import OpenAI
import numpy as np
import os

os.environ['OPENAI_API_KEY'] = 'your-key-here'  # Replace with actual key
client = OpenAI()

# Knowledge base
KNOWLEDGE_BASE = [
    "Our company offers 24 days of paid vacation per year.",
    "Remote work is allowed 3 days per week with manager approval.",
    "Health insurance covers employee and immediate family members.",
    "Parental leave is 12 weeks paid for primary caregiver.",
]

def get_embedding(text: str) -> list:
    """Get embedding vector from OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class VectorStore:
    """Simple in-memory vector store"""
    
    def __init__(self, documents: list):
        print("Building vector store...")
        self.documents = documents
        self.embeddings = [get_embedding(doc) for doc in documents]
        print(f"✓ Embedded {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 2) -> list:
        """Semantic search: find most similar documents"""
        query_embedding = get_embedding(query)
        
        # Calculate similarity scores
        scores = [
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self.embeddings
        ]
        
        # Get top_k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return top documents
        return [self.documents[i] for i in top_indices]

def rag_query(vector_store: VectorStore, question: str) -> str:
    """
    RAG Pipeline with Vector Search
    1. Embed query
    2. Search vector store for similar chunks
    3. Generate answer using retrieved context
    """
    # RETRIEVE: Semantic search
    context_chunks = vector_store.search(question, top_k=2)
    context = "\n".join(context_chunks)
    
    # AUGMENT: Build prompt with context
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
    
    # GENERATE: Get answer from LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

# Demo
if __name__ == "__main__":
    # Build vector store (done once)
    vector_store = VectorStore(KNOWLEDGE_BASE)
    
    # Ask questions
    questions = [
        "How many vacation days do I get?",
        "Can I work from home?",
        "What's the stock price?",  # Out of scope
    ]
    
    print("\n" + "="*80)
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {rag_query(vector_store, q)}")
        print("-" * 80)
