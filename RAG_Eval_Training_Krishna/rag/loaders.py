"""rag/loaders.py

Loads documents from source(s).

In interviews, mention:
- PDFs can be messy (hyphenation, headers/footers, page numbers).
- For production: clean text + keep page metadata for citations/audit.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document

@dataclass(frozen=True)
class LoadedDoc:
    docs: List[Document]
    num_pages: int

def load_pdf(path: str) -> LoadedDoc:
    reader = PdfReader(path)
    docs: List[Document] = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Keep metadata so we can cite page numbers later.
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path, "page": page_idx + 1},
            )
        )

    return LoadedDoc(docs=docs, num_pages=len(reader.pages))
