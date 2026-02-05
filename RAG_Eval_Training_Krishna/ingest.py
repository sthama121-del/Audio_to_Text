"""ingest.py

One-time (idempotent) ingestion of the HR policy PDF into Chroma.

RUN:
  python ingest.py

What this teaches you (interview coverage):
- document loading
- chunking strategy
- embeddings + vector storage
- idempotent ingestion (no duplicates)
"""

from __future__ import annotations
import os
from rich import print

from rag.config import Settings, validate
from rag.loaders import load_pdf
from rag.chunking import chunk_documents
from rag.vectorstore import ingest_chunks, write_ingestion_manifest

def main():
    settings = Settings()
    validate(settings)

    loaded = load_pdf(settings.PDF_PATH)
    print(f"[bold green]✓ Loaded PDF[/bold green] pages={loaded.num_pages} path={settings.PDF_PATH}")

    chunks = chunk_documents(loaded.docs, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    print(f"[bold green]✓ Chunked[/bold green] chunks={len(chunks)} chunk_size={settings.CHUNK_SIZE} overlap={settings.CHUNK_OVERLAP}")

    stats = ingest_chunks(settings, chunks)
    print(f"[bold green]✓ Ingested[/bold green] newly_added={stats.newly_added} skipped_existing={stats.skipped_existing}")

    write_ingestion_manifest(settings, stats)
    print(f"[bold green]✓ Wrote[/bold green] outputs/ingestion_manifest.json")

if __name__ == "__main__":
    main()
