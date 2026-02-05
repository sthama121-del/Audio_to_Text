from dotenv import load_dotenv
load_dotenv()

"""
RAG Demo with Vector Database (ChromaDB)
Shows production-style RAG with embeddings and vector search in ~70 lines
"""

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import os

# Initialize OpenAI client for answer generation
client = OpenAI()

# =============================================================================
# Step 1: Define Knowledge Base
# =============================================================================
# In production, this would come from PDFs, databases, or APIs
# For demo purposes, we use a simple list of strings
KNOWLEDGE_BASE = [
    "Our company offers 24 days of paid vacation per year.",
    "Remote work is allowed 3 days per week with manager approval.",
    "Health insurance covers employee and immediate family members.",
    "Parental leave is 12 weeks paid for primary caregiver.",
]

# =============================================================================
# Step 2: Create Vector Database with ChromaDB
# =============================================================================
print("Building vector database...")

# Create ChromaDB client (in-memory database for this demo)
# In production, use PersistentClient to save to disk
chroma_client = chromadb.Client()

# Create embedding function that will convert text to vectors
# Uses OpenAI's text-embedding-3-small model (1536 dimensions)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Create a collection (similar to a table in traditional databases)
# The collection stores both the text and its vector embeddings
collection = chroma_client.create_collection(
    name="hr_policies",
    embedding_function=openai_ef
)

# Insert documents into the vector database
# ChromaDB automatically creates embeddings for each document
collection.add(
    documents=KNOWLEDGE_BASE,  # The actual text content
    ids=[f"doc_{i}" for i in range(len(KNOWLEDGE_BASE))]  # Unique IDs for each doc
)
print(f"✓ Vector database ready with {len(KNOWLEDGE_BASE)} documents\n")

# =============================================================================
# Step 3: Vector Retrieval Function
# =============================================================================
def vector_retrieval(query: str, top_k: int = 2) -> str:
    """
    Performs semantic search using vector similarity.
    
    How it works:
    1. Query text gets converted to an embedding vector
    2. ChromaDB calculates cosine similarity between query vector and all document vectors
    3. Returns the top_k most similar documents
    
    This is better than keyword search because it understands meaning:
    - "Can I work remotely?" matches "Remote work is allowed" (different words, same meaning)
    - "How many days off?" matches "24 days of paid vacation" (semantic understanding)
    """
    # Query the vector database
    # ChromaDB automatically embeds the query and finds similar documents
    results = collection.query(
        query_texts=[query],  # Can search multiple queries at once
        n_results=top_k       # Number of most similar documents to return
    )
    
    # Extract the retrieved document texts
    # results is a dict with 'documents', 'distances', 'ids', etc.
    retrieved_docs = results['documents'][0]  # [0] because we only sent one query
    
    # Join the retrieved chunks with newlines for the LLM prompt
    return "\n".join(retrieved_docs)

# =============================================================================
# Step 4: RAG Query Function
# =============================================================================
def rag_query(question: str) -> str:
    """
    Complete RAG Pipeline: Retrieve → Augment → Generate
    
    This is the core of RAG:
    1. RETRIEVE: Find relevant context from vector database
    2. AUGMENT: Build a prompt that includes the context
    3. GENERATE: Let the LLM answer using only the provided context
    """
    
    # STEP 1: RETRIEVE relevant context using vector search
    # This finds the most semantically similar documents to the question
    context = vector_retrieval(question)
    
    # STEP 2: AUGMENT the prompt with retrieved context
    # The system instruction constrains the LLM to only use the provided context
    # This prevents hallucination - the LLM can't make up information
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
    
    # STEP 3: GENERATE answer using LLM
    # The LLM reads the context and generates a grounded answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",           # Fast and cost-effective model
        messages=[{"role": "user", "content": prompt}],
        temperature=0                   # 0 = deterministic, no creativity
    )
    
    # Extract and return the text response
    return response.choices[0].message.content

# =============================================================================
# Step 5: Demo Execution
# =============================================================================
if __name__ == "__main__":
    # Test questions covering different scenarios
    questions = [
        "How many vacation days do I get?",      # Direct fact retrieval
        "What's the remote work policy?",        # Policy lookup
        "Can I work remotely?",                  # Semantic variation - tests vector search
        "What's the stock price?",               # Out of scope - should say "I don't know"
    ]
    
    # Run RAG pipeline for each question
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {rag_query(q)}")
        print("-" * 80)

