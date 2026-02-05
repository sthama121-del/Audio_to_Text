"""rag/vectorstore.py

Creates/loads Chroma and performs idempotent ingestion.

Why idempotency matters (interview):
- if ingestion is re-run, you must avoid duplicating chunks (double indexing).
- best practice: store a content hash per chunk and upsert.
"""

from __future__ import annotations
import json
import os
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from rag.config import Settings

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

@dataclass
class IngestionStats:
    total_chunks: int
    newly_added: int
    skipped_existing: int

def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )

def load_vectorstore(settings: Settings) -> Chroma:
    client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    vs = Chroma(
        client=client,
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=get_embeddings(settings),
    )
    return vs

def ingest_chunks(settings: Settings, chunks: List[Document]) -> IngestionStats:
    """Upsert-like behavior using deterministic IDs.

    Chroma doesn't support true upsert for langchain wrapper,
    but we can emulate by:
      - using stable ids (hash)
      - checking existing ids
      - adding only missing ones

    This is good enough for training + small corpora.
    """
    vs = load_vectorstore(settings)

    ids = []
    metadatas = []
    texts = []

    for c in chunks:
        # Stable per-chunk id. Include page + content to reduce collisions.
        chunk_id = _sha256(f"{c.metadata.get('source')}|{c.metadata.get('page')}|{c.page_content}")
        ids.append(chunk_id)
        metadatas.append({**c.metadata, "chunk_id": chunk_id})
        texts.append(c.page_content)

    # Fetch existing IDs in batches (Chroma API limitation).
    # If the collection is empty, this is fast.
    existing = set()
    try:
        col = vs._collection
        # peek returns some docs, but we need ids. We'll use get with where_document not.
        # Chroma can get by ids; if many, chunk the requests.
        batch = 200
        for i in range(0, len(ids), batch):
            sub = ids[i:i+batch]
            res = col.get(ids=sub, include=[])
            for _id in res.get("ids", []):
                existing.add(_id)
    except Exception:
        # If something changes in Chroma API, we fall back to "no existing".
        existing = set()

    to_add = [(i, t, m) for i, t, m in zip(ids, texts, metadatas) if i not in existing]

    if to_add:
        add_ids, add_texts, add_metas = zip(*to_add)
        vs.add_texts(list(add_texts), metadatas=list(add_metas), ids=list(add_ids))

    stats = IngestionStats(
        total_chunks=len(chunks),
        newly_added=len(to_add),
        skipped_existing=len(chunks) - len(to_add),
    )
    return stats

def write_ingestion_manifest(settings: Settings, stats: IngestionStats) -> None:
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(settings.OUTPUT_DIR, "ingestion_manifest.json")
    payload = {
        "total_chunks": stats.total_chunks,
        "newly_added": stats.newly_added,
        "skipped_existing": stats.skipped_existing,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
