# =============================================================================
# app.py — Flask Web API for HR Policy Chatbot
# =============================================================================
#
# RESPONSIBILITY: Expose the RAG pipeline as a REST API endpoint.
#
# This allows the frontend to send questions and receive answers without
# running the full evaluation pipeline.
#
# DESIGN DECISION: We initialize the vector store ONCE at startup (singleton)
# rather than rebuilding it per request. This is critical for production —
# embedding 50 chunks takes ~2-3 seconds, which would make every user query
# unacceptably slow.
# =============================================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

import config
from ingestion import run_ingestion
from retrieval import retrieve_chunks, format_context
from generation import generate_answer

# ---------------------------------------------------------------------------
# 1. Flask App Initialization
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for frontend requests

# ---------------------------------------------------------------------------
# 2. Global Vector Store (initialized once at startup)
# ---------------------------------------------------------------------------
# INTERVIEW GOTCHA: "Why not rebuild the vector store per request?"
# Answer: Vector store initialization involves:
#   - Loading the PDF
#   - Chunking the text
#   - Calling OpenAI embeddings API (~50 chunks × 100ms = 5s)
#   - Building the index
# Doing this per request would make EVERY query take 5+ seconds.
# In production, the vector store is built once and persists (or cached).
vector_store = None


def initialize_vector_store():
    """
    Initializes the vector store at application startup.
    
    PRODUCTION NOTE: In a real deployment, you'd:
    1. Check if a persisted vector store exists on disk
    2. If yes, load it (instant)
    3. If no, run ingestion and persist it
    4. Set up a background job to re-ingest on document updates
    """
    global vector_store
    
    # Path to the HR policy document
    # MODIFY THIS if your document is elsewhere
    hr_policy_path = os.path.join(os.path.dirname(__file__), "HR_POLICY.pdf")
    
    if not os.path.exists(hr_policy_path):
        print(f"[app] ERROR: HR policy document not found at {hr_policy_path}")
        print("[app] Please ensure HR_POLICY.pdf is in the same directory as app.py")
        return False
    
    try:
        print("\n[app] Initializing vector store (this may take a moment)...")
        vector_store = run_ingestion(hr_policy_path)
        print("[app] ✓ Vector store ready!")
        return True
    except Exception as e:
        print(f"[app] ✗ Failed to initialize vector store: {e}")
        return False


# ---------------------------------------------------------------------------
# 3. API Endpoints
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Serve the frontend HTML."""
    return send_from_directory('static', 'index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Main API endpoint: receives a question, returns an answer.
    
    Request format:
        POST /ask
        Content-Type: application/json
        {
            "question": "What is the remote work policy?"
        }
    
    Response format:
        {
            "question": "What is the remote work policy?",
            "answer": "According to the HR policy...",
            "contexts": ["Chunk 1 text...", "Chunk 2 text..."],
            "num_chunks": 3
        }
    
    Error response:
        {
            "error": "Error message here"
        }
    """
    global vector_store
    
    # Validate vector store is initialized
    if vector_store is None:
        return jsonify({
            "error": "Vector store not initialized. Please check server logs."
        }), 500
    
    # Parse request
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "error": "Question cannot be empty"
            }), 400
        
        print(f"\n[app] Received question: {question}")
        
        # Step 1: Retrieve relevant chunks
        retrieved_docs = retrieve_chunks(
            vector_store=vector_store,
            query=question,
            k=config.RETRIEVAL_K
        )
        
        # Step 2: Format context for LLM
        context = format_context(retrieved_docs)
        
        # Step 3: Generate answer
        answer = generate_answer(
            question=question,
            context=context
        )
        
        # Extract chunk texts for frontend display
        chunk_texts = [doc.page_content for doc in retrieved_docs]
        
        print(f"[app] ✓ Answer generated ({len(answer)} chars)")
        
        return jsonify({
            "question": question,
            "answer": answer,
            "contexts": chunk_texts,
            "num_chunks": len(chunk_texts)
        })
    
    except Exception as e:
        print(f"[app] ✗ Error processing question: {e}")
        return jsonify({
            "error": f"Failed to process question: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "vector_store_initialized": vector_store is not None
    })


# ---------------------------------------------------------------------------
# 4. Application Startup
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Validate configuration
    config.validate_config()
    
    # Initialize vector store
    if not initialize_vector_store():
        print("\n[app] ✗ FATAL: Could not initialize vector store. Exiting.")
        exit(1)
    
    print("\n" + "=" * 60)
    print(" HR POLICY CHATBOT API SERVER")
    print("=" * 60)
    print(f" Frontend: http://localhost:5000")
    print(f" API endpoint: POST http://localhost:5000/ask")
    print(f" Health check: GET http://localhost:5000/health")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True  # Set to False in production
    )
