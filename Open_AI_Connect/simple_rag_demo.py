from dotenv import load_dotenv
load_dotenv()

"""
Minimal RAG (Retrieval-Augmented Generation) Demo
Perfect for technical interviews - shows core RAG concept in ~50 lines
"""

from openai import OpenAI
import os

client = OpenAI()

# Step 1: Knowledge Base (in production, this would be a vector database)
KNOWLEDGE_BASE = [
    "Our company offers 24 days of paid vacation per year.",
    "Remote work is allowed 3 days per week with manager approval.",
    "Health insurance covers employee and immediate family members.",
    "Parental leave is 12 weeks paid for primary caregiver.",
]

def simple_retrieval(query: str, top_k: int = 2) -> str:
    """
    Simple keyword-based retrieval (in production: use embeddings + vector search)
    Returns the most relevant chunks from knowledge base
    """
    # For demo: just return chunks that contain query words
    query_words = query.lower().split()
    scored_chunks = []
    
    for chunk in KNOWLEDGE_BASE:
        score = sum(1 for word in query_words if word in chunk.lower())
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by relevance and return top_k
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    retrieved = [chunk for _, chunk in scored_chunks[:top_k]]
    
    return "\n".join(retrieved)

def rag_query(question: str) -> str:
    """
    RAG Pipeline: Retrieve → Augment → Generate
    """
    # Step 1: RETRIEVE relevant context
    context = simple_retrieval(question)
    
    # Step 2: AUGMENT the prompt with retrieved context
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
    
    # Step 3: GENERATE answer using LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

# Demo
if __name__ == "__main__":
    questions = [
        "How many vacation days do I get?",
        "What's the remote work policy?",
        "What's the stock price?",  # Out of scope - should say "I don't know"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {rag_query(q)}")
        print("-" * 80)