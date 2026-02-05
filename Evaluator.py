"""
RAG + Retrieval Evaluation + Multi-agent (CrewAI) demo (TEACHING VERSION)

What this script demonstrates (interview mapping):
1) Ingestion: load docs from disk (TextLoader)
2) Chunking: split long docs into retrieval-friendly chunks (RecursiveCharacterTextSplitter)
3) Embeddings: convert each chunk into vectors (OpenAIEmbeddings)
4) Vector Store: store vectors and do similarity search (FAISS)
5) Retrieval Evaluation: Recall@K, MRR@K, nDCG@K
6) Grounded Answering: answer only from retrieved context + cite chunk ids
7) Multi-agent Orchestration: Retriever -> Answerer -> Critic (CrewAI)
8) Basic observability: latency measurement (time.time)

Run steps:
- Put your .txt files in ./docs
- Install deps
- Set provider credentials (e.g., OPENAI_API_KEY) if using OpenAI-compatible models

Install:
  pip install langchain langchain-community langchain-text-splitters faiss-cpu crewai numpy
Optional:
  pip install langchain-openai
"""

# -----------------------
# Imports
# -----------------------

import os                 # Used for filesystem operations like listing files in a folder
import time               # Used for timing (latency measurement)
import math               # Used for logarithms in nDCG calculation
import numpy as np        # Used for averaging metrics across examples
from typing import List, Dict, Tuple  # Type hints improve readability and interview clarity

# LangChain loaders/splitters/vector store
from langchain_community.document_loaders import TextLoader  # Loads text files into LangChain Document objects
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits docs into chunks using smart separators
from langchain_community.vectorstores import FAISS           # In-memory/local vector index with similarity search
from langchain.schema import Document                        # The basic LangChain document structure

# Try importing OpenAI-compatible embeddings + chat model
# This lets the script run only if the package is installed.
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False


# CrewAI (multi-agent orchestration)
from crewai import Agent, Task, Crew, Process


# =========================================================
# 1) LOAD DOCUMENTS (INGESTION)
# =========================================================
def load_txt_docs(folder: str) -> List[Document]:
    """
    PURPOSE:
      Read .txt files from a folder and convert them into LangChain Document objects.

    WHY INTERVIEWERS CARE:
      This is your ingestion layer: "How do you load data into the pipeline?"

    INPUT:
      folder: path like "./docs"

    OUTPUT:
      docs: List[Document], each Document has:
        - page_content: the text
        - metadata: typically includes 'source' file path from the loader
    """
    docs: List[Document] = []  # Create an empty list to collect Documents

    # os.listdir(folder) returns file names in that folder (not full paths)
    for fn in os.listdir(folder):
        # Only process .txt files (skip pdf, docx, etc in this demo)
        if fn.lower().endswith(".txt"):

            # Build full path like "./docs/file1.txt"
            full_path = os.path.join(folder, fn)

            # TextLoader reads the file and converts to Document(s)
            # encoding="utf-8" avoids common unicode issues
            loader = TextLoader(full_path, encoding="utf-8")

            # loader.load() returns a list[Document]
            # We extend our docs list with these loaded docs
            docs.extend(loader.load())

    return docs  # Return all loaded documents


# =========================================================
# 2) CHUNK DOCUMENTS (CHUNKING)
# =========================================================
def chunk_docs(docs: List[Document], chunk_size: int = 900, overlap: int = 150) -> List[Document]:
    """
    PURPOSE:
      Split long documents into smaller chunks so retrieval works well.

    WHY INTERVIEWERS CARE:
      Chunking strongly affects retrieval metrics and hallucination rates.
      Recursive chunking is a solid default.

    KEY PARAMETERS:
      chunk_size: max characters per chunk (token-based is better in prod, but ok for demo)
      overlap: repeated characters between adjacent chunks to preserve context continuity
    """
    # RecursiveCharacterTextSplitter tries separators in order to avoid cutting sentences abruptly.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,              # Maximum size of each chunk (in characters here)
        chunk_overlap=overlap,              # Shared text between chunks
        separators=["\n\n", "\n", ". ", " ", ""]
        # order matters:
        # 1) split by paragraphs
        # 2) then new lines
        # 3) then sentence-ish breaks
        # 4) then spaces
        # 5) finally brute-force if nothing else works
    )

    # This takes the list of docs and returns smaller Document chunks
    chunks = splitter.split_documents(docs)

    # Add our own metadata: chunk_id
    # In interviews: mention metadata is used for filters (doc_type, region, ACL tags)
    for i, d in enumerate(chunks):
        # d.metadata originally has things like {"source": ".../file.txt"}
        # we merge existing metadata with our new key chunk_id
        d.metadata = {**d.metadata, "chunk_id": i}

    return chunks


# =========================================================
# 3) BUILD VECTOR INDEX (EMBEDDINGS + VECTOR STORE)
# =========================================================
def build_faiss_index(chunks: List[Document]):
    """
    PURPOSE:
      Convert chunks into embeddings (vectors) and store them in FAISS.

    WHY INTERVIEWERS CARE:
      This is "indexing" — the core of RAG retrieval.

    OUTPUT:
      vs: a FAISS vector store that can do similarity_search(query, k)
    """
    # If OpenAI embeddings are not available, fail with a clear error.
    if not OPENAI_OK:
        raise RuntimeError(
            "Install langchain-openai OR swap in another embedding provider (HuggingFace, etc.)."
        )

    # Create embedding model object (this generates a vector for each chunk)
    # In Azure OpenAI, you’d configure endpoint/deployment via env vars.
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    # Build FAISS index from the list of chunk Documents
    # This:
    # - embeds each chunk text
    # - stores the vectors in FAISS
    # - stores mapping from vector -> original chunk Document
    vs = FAISS.from_documents(chunks, emb)

    return vs


# =========================================================
# 4) RETRIEVAL EVALUATION METRICS
# =========================================================

def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """
    PURPOSE:
      Recall@K: Did we retrieve at least one relevant chunk in top K?

    INTERVIEW EXPLANATION:
      If any relevant chunk appears in top K results, recall@K = 1 else 0.
      Average across many queries gives overall recall.
    """
    topk = set(retrieved_ids[:k])   # take first k retrieved ids
    rel = set(relevant_ids)         # the ground-truth relevant ids
    return 1.0 if len(topk & rel) > 0 else 0.0  # intersection non-empty => hit


def mrr_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """
    PURPOSE:
      MRR@K: Mean Reciprocal Rank - how early the first relevant result appears.

    INTERVIEW EXPLANATION:
      If first relevant item is at rank r, score = 1/r.
      If no relevant item in top K, score = 0.
    """
    rel = set(relevant_ids)

    # enumerate gives (index, value), start=1 so rank starts at 1 not 0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """
    PURPOSE:
      nDCG@K: normalized Discounted Cumulative Gain - evaluates ranking quality
      especially when multiple results can be relevant.

    EXPLANATION:
      - DCG rewards relevant items but discounts lower ranks using log2
      - nDCG = DCG / IDCG (ideal DCG)
    """
    rel = set(relevant_ids)

    def dcg(ids: List[int]) -> float:
        score = 0.0
        for i, cid in enumerate(ids[:k], start=1):
            gain = 1.0 if cid in rel else 0.0
            # log2(i+1) discount: rank 1 discount is log2(2)=1
            score += gain / math.log2(i + 1)
        return score

    # Ideal DCG assumes the best possible ordering: all relevant appear at the top
    # If there are R relevant documents, ideal has min(R, K) ones then zeros.
    ideal_gains = [1] * min(len(rel), k) + [0] * max(0, k - min(len(rel), k))

    idcg = 0.0
    for i, g in enumerate(ideal_gains[:k], start=1):
        idcg += g / math.log2(i + 1)

    # If no relevant documents exist, idcg=0 => define nDCG as 0 to avoid divide by zero
    return 0.0 if idcg == 0 else dcg(retrieved_ids) / idcg


def evaluate_retrieval(vs, eval_set: List[Dict], k: int = 5) -> Dict[str, float]:
    """
    PURPOSE:
      Run retrieval for each query in eval_set and compute average metrics.

    eval_set format (you create this manually in real projects):
      [
        {"query": "...", "relevant_chunk_ids": [1, 5, 9]},
        ...
      ]

    OUTPUT:
      dict like {"Recall@5": 0.8, "MRR@5": 0.6, "nDCG@5": 0.7}
    """
    recalls, mrrs, ndcgs = [], [], []

    for ex in eval_set:
        query = ex["query"]

        # similarity_search returns top-k Documents for the query
        docs = vs.similarity_search(query, k=k)

        # Convert retrieved docs into chunk ids so we can score them
        retrieved_ids = [d.metadata.get("chunk_id") for d in docs]

        # Compute per-query metric values
        recalls.append(recall_at_k(retrieved_ids, ex["relevant_chunk_ids"], k))
        mrrs.append(mrr_at_k(retrieved_ids, ex["relevant_chunk_ids"], k))
        ndcgs.append(ndcg_at_k(retrieved_ids, ex["relevant_chunk_ids"], k))

    # Average across all queries (numpy mean)
    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"MRR@{k}": float(np.mean(mrrs)),
        f"nDCG@{k}": float(np.mean(ndcgs)),
    }


# =========================================================
# 5) GROUNDED ANSWERING (CITATIONS)
# =========================================================

def format_context_with_citations(docs: List[Document]) -> str:
    """
    PURPOSE:
      Build a single context string with chunk ids and sources.
      This lets the model cite evidence in answers.

    INTERVIEW VALUE:
      Shows groundedness + auditability.
    """
    blocks = []

    for d in docs:
        # Pull chunk_id from metadata (we added it during chunking)
        cid = d.metadata.get("chunk_id", "NA")

        # The loader typically puts the file path in metadata["source"]
        src = d.metadata.get("source", "unknown")

        # Create a chunk label + the chunk content
        blocks.append(f"[chunk:{cid} | src:{src}]\n{d.page_content}")

    # Separate chunks so the model can distinguish them
    return "\n\n---\n\n".join(blocks)


def answer_question(vs, question: str, top_k: int = 5) -> Tuple[str, List[int], float]:
    """
    PURPOSE:
      Retrieve top_k relevant chunks and ask the LLM to answer ONLY from that context.

    OUTPUT:
      answer_text: model response
      retrieved_ids: which chunk ids were retrieved
      latency: time taken (basic observability)
    """
    start = time.time()  # record start time for latency tracking

    # Step 1: retrieve relevant chunks
    retrieved_docs = vs.similarity_search(question, k=top_k)

    # Store retrieved ids for transparency/debugging
    retrieved_ids = [d.metadata.get("chunk_id") for d in retrieved_docs]

    # Step 2: construct a context with citations
    context = format_context_with_citations(retrieved_docs)

    # Ensure we have a chat model provider
    if not OPENAI_OK:
        raise RuntimeError("Install langchain-openai or swap in another chat model provider.")

    # Step 3: instantiate the chat model
    # temperature low => less creative => better factual grounding
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Step 4: prompt instructs strict grounding + refusal if missing evidence
    prompt = f"""
You are a factual assistant.
Answer ONLY using the provided context.
If the context does not contain the answer, say: "I don't have enough information in the documents."
Cite chunk ids you used, like [chunk:12].

QUESTION:
{question}

CONTEXT:
{context}
"""

    # Step 5: invoke the model
    resp = llm.invoke(prompt).content

    # Step 6: compute latency
    latency = time.time() - start

    return resp, retrieved_ids, latency


# =========================================================
# 6) CREWAI MULTI-AGENT FLOW
# =========================================================

def run_crewai_flow(question: str, vs) -> str:
    """
    PURPOSE:
      Demonstrate multi-agent workflow:
      - Retriever agent fetches context
      - Answerer agent writes grounded answer
      - Critic agent validates citations and groundedness

    INTERVIEW VALUE:
      Shows agent orchestration (CrewAI) + quality enforcement.
    """

    # Agent 1: Retriever (search expert)
    retriever = Agent(
        role="Retriever",
        goal="Retrieve the most relevant chunks and provide citations.",
        backstory="You are expert in search and retrieval evaluation.",
        allow_delegation=False
    )

    # Agent 2: Answerer (grounded writer)
    answerer = Agent(
        role="Answerer",
        goal="Write a grounded answer strictly from retrieved context and cite chunks.",
        backstory="You never hallucinate. You refuse if evidence is missing.",
        allow_delegation=False
    )

    # Agent 3: Critic (strict validator)
    critic = Agent(
        role="Critic",
        goal="Check if answer is grounded and cites chunks. If not, demand fixes.",
        backstory="You are strict about factuality and missing citations.",
        allow_delegation=False
    )

    # "Tool" function to retrieve context (CrewAI can call functions assigned to Tasks)
    def retrieve_tool(q: str) -> str:
        docs = vs.similarity_search(q, k=5)            # retrieve top 5 chunks
        return format_context_with_citations(docs)     # return context string

    # Task 1: retrieval task executes the function to get context
    t1 = Task(
        description=f"Retrieve context for question: {question}",
        agent=retriever,
        expected_output="A context block with chunk citations.",
        function=lambda: retrieve_tool(question)  # runs retrieval_tool(question)
    )

    # Task 2: answer task uses previous task output as input in Crew pipeline
    t2 = Task(
        description="Using retrieved context, write final grounded answer with citations.",
        agent=answerer,
        expected_output="Answer with [chunk:id] citations and refusal if missing evidence."
    )

    # Task 3: critic checks for groundedness/citations
    t3 = Task(
        description="Review answer for groundedness and citation correctness; propose corrections.",
        agent=critic,
        expected_output="Pass/Fail + required fixes."
    )

    # Assemble crew and run sequentially (t1 -> t2 -> t3)
    crew = Crew(
        agents=[retriever, answerer, critic],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )

    # kickoff executes tasks in order and returns final combined output
    return crew.kickoff()


# =========================================================
# 7) MAIN ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    # Folder where you keep .txt docs
    DOCS_DIR = "./docs"

    # --- Ingestion
    docs = load_txt_docs(DOCS_DIR)

    # --- Chunking
    chunks = chunk_docs(docs, chunk_size=900, overlap=150)

    # --- Index build (embeddings + FAISS)
    vs = build_faiss_index(chunks)

    # --- Minimal evaluation dataset (toy)
    # In real projects, you manually label which chunk ids contain the answer.
    eval_set = [
        {"query": "What is the return policy?", "relevant_chunk_ids": [1, 2]},
        {"query": "How to escalate a work order?", "relevant_chunk_ids": [10]},
    ]

    # Run retrieval evaluation metrics
    metrics = evaluate_retrieval(vs, eval_set, k=5)
    print("Retrieval metrics:", metrics)

    # --- Ask a question (RAG answering)
    q = "Explain the SOP steps for handling a damaged line."
    ans, ids, latency = answer_question(vs, q, top_k=5)

    print("\nLatency(sec):", round(latency, 3))
    print("Retrieved chunk ids:", ids)
    print("\nAnswer:\n", ans)

    # --- Multi-agent orchestration demo
    print("\n--- CrewAI flow output ---\n")
    print(run_crewai_flow(q, vs))
