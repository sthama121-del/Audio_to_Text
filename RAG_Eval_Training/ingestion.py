# =============================================================================
# ingestion.py — Document Loading, Chunking & Vector Store Population
# =============================================================================
#
# RESPONSIBILITY: Take a raw document → produce searchable vector embeddings.
#
# Pipeline:  raw file  →  Document objects  →  chunked Documents  →  ChromaDB
#
# This module is intentionally separated from retrieval and generation so that
# you can swap out any stage independently — a key principle interviewers test.
# =============================================================================

import os
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import config


# ---------------------------------------------------------------------------
# 1. Document Loader
# ---------------------------------------------------------------------------
def load_document(file_path: str) -> List[Document]:
    """
    Loads a document from disk. Supports .txt and .pdf formats.

    INTERVIEW GOTCHA: "What happens if the file doesn't exist?"
    Answer: We fail IMMEDIATELY with a clear error, not silently return [].
    Silent failures in pipelines are the #1 cause of production bugs.

    SCENARIO COVERAGE — Why support both .txt and .pdf?
    In a real HR bot, the source is almost always a PDF (scanned or digital).
    We include .txt support so this training project works without a PDF
    dependency. The swap is a 1-line change — that's the point of the abstraction.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Document not found at: {file_path}\n"
            f"Make sure HR_POLICY.PDF is in the same directory as this script."
        )

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        # PyPDFLoader handles multi-page PDFs and returns one Document per page.
        # INTERVIEW GOTCHA: "What about scanned PDFs?"
        # Answer: PyPDFLoader only works on digitally-native PDFs.
        # For scanned PDFs, you'd need an OCR step (e.g., pytesseract or
        # AWS Textract) BEFORE this loader. Always ask the client about
        # the PDF source before choosing a loader.
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use .txt or .pdf")

    docs = loader.load()
    print(f"[ingestion] ✓ Loaded {len(docs)} document section(s) from '{os.path.basename(file_path)}'")
    return docs


# ---------------------------------------------------------------------------
# 2. Text Splitting (Chunking)
# ---------------------------------------------------------------------------
def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Splits raw documents into smaller, overlapping chunks.

    SCENARIO COVERAGE — Why RecursiveCharacterTextSplitter over others?
    ─────────────────────────────────────────────────────────────────────
    LangChain offers several splitters. Here's why Recursive is the gold standard:

    │ Splitter                      │ Strategy                  │ Best For           │
    │ CharacterTextSplitter         │ Single delimiter (\n\n)   │ Simple, uniform text│
    │ RecursiveCharacterTextSplitter│ Tries \n\n → \n → " " → ""│ Natural language   │ ← WE USE THIS
    │ TokenTextSplitter            │ Token count (tiktoken)    │ Token-budget control│
    │ MarkdownHeaderTextSplitter   │ # headers                 │ Markdown docs      │

    Recursive tries to split on paragraph breaks first (\n\n). If a paragraph
    is still too long, it falls back to single newlines, then spaces, then
    character-by-character. This preserves semantic coherence FAR better than
    a blind character cut.

    INTERVIEW GOTCHA: "What's the risk of chunk_size being too small?"
    Answer: Each chunk becomes a separate retrieval candidate. Too-small chunks
    lose context (a sentence without its surrounding paragraph is meaningless).
    Too-large chunks waste the LLM's context window on irrelevant text.
    800 chars ≈ 200 tokens is the sweet spot for document QA.

    INTERVIEW GOTCHA: "What is 'embedding drift' and how does chunking affect it?"
    Answer: Embedding drift occurs when the semantic meaning of a chunk shifts
    over time (e.g., policy updates). If your chunk boundaries cut a sentence
    in half, the embedding of that fragment drifts away from its true meaning.
    Overlap helps mitigate this by ensuring boundary sentences appear in TWO
    chunks, increasing the chance one will be retrieved correctly.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # These separators are tried IN ORDER. The splitter picks the first
        # one that produces chunks within chunk_size.
        separators=["\n\n", "\n", " ", ""],
        # length_function defaults to len() which counts characters.
        # For token-precise chunking, swap to: length_function=tiktoken_len
        length_function=len,
    )

    chunks = splitter.split_documents(docs)
    print(f"[ingestion] ✓ Split into {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding Model Initialization
# ---------------------------------------------------------------------------
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Returns a configured OpenAI embedding model instance.

    INTERVIEW GOTCHA: "Should you create a new embedding client per request?"
    Answer: NO. The client is stateless and thread-safe. Create it ONCE and
    reuse it. Re-creating it per call wastes TCP handshakes and connection
    pool setup. This is why it's a module-level factory, not inline.

    INTERVIEW GOTCHA: "What's the dimensionality of text-embedding-3-small?"
    Answer: 1536 dimensions. This matters for:
      - Memory: 1536 float32 = 6KB per vector. At 50 chunks = 300KB (trivial).
      - Similarity: cosine similarity is O(d), so higher dims = slower search.
        At 1536 dims and <1000 vectors, this is imperceptible.
        At millions of vectors, you'd want quantization or a dedicated ANN index.
    """
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )


# ---------------------------------------------------------------------------
# 4. Vector Store Creation & Population
# ---------------------------------------------------------------------------
def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Creates an in-memory ChromaDB vector store and populates it with chunks.

    INTERVIEW GOTCHA: "Why ChromaDB over FAISS for this use case?"
    ─────────────────────────────────────────────────────────────────
    │ Feature            │ ChromaDB              │ FAISS                │
    │ Setup              │ Zero-config, Python   │ Requires manual idx  │
    │ Metadata filtering │ Built-in              │ Not native           │
    │ Persistence        │ Optional (toggle)     │ Manual save/load     │
    │ Production scale   │ Up to ~1M vectors     │ Billions of vectors  │
    │ Best for           │ Prototyping, <1M docs │ Large-scale prod     │

    For an HR document bot (<1000 chunks), ChromaDB's ease of use wins.
    In production at scale, FAISS (or Milvus/Weaviate) is the right call.

    INTERVIEW GOTCHA: "What happens if you ingest the same document twice?"
    Answer: ChromaDB deduplicates by ID. If no ID is provided, it generates
    UUIDs — meaning duplicates WILL be stored. In production, always assign
    deterministic IDs (e.g., hash of chunk text) to prevent bloat.
    """
    embedding_model = get_embedding_model()

    # In-memory client: data lives only while the process runs.
    # For persistence, use: chromadb.PersistentClient(path="./chroma_db")
    #chroma_client = chromadb.Client()  # ephemeral, in-memory
    #Persistent client: data saved to disk, survives restarts.
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    #chcek if collection already exists and has data
    existing = chroma_client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    if existing.count() > 0:
        print(f"[ingestion] Vector store already populated with {existing.count()} vectors. Skipping ingestion.")
        return Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory="./chroma_db"
        )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=chroma_client,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )   


    print(f"[ingestion] ✓ Vector store populated with {len(chunks)} embedded chunks")
    return vector_store


# ---------------------------------------------------------------------------
# 5. Master Ingestion Pipeline
# ---------------------------------------------------------------------------
def run_ingestion(file_path: str) -> Chroma:
    """
    Orchestrates the full ingestion pipeline: load → chunk → embed → store.

    Returns the populated vector store, ready for retrieval.

    INTERVIEW GOTCHA: "How would you make this idempotent?"
    Answer: Check if the vector store already contains docs from this file
    (via metadata filtering). If yes, skip re-ingestion. This prevents
    duplicate vectors on restart — critical in long-running services.
    """
    print("\n" + "=" * 60)
    print(" PHASE 1: INGESTION")
    print("=" * 60)

    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    vector_store = build_vector_store(chunks)

    return vector_store
