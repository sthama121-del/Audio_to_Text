"""rag/chunking.py

Chunking is the #1 hidden lever in RAG quality.

INTERVIEW TOPICS:
- chunk_size & overlap trade-off
- semantic vs character chunking
- "lost in the middle" (long contexts hide the relevant bit)
"""

from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # This preserves metadata (like page numbers) per chunk.
    return splitter.split_documents(docs)
