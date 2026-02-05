# eval/embeddings_shim.py
from __future__ import annotations

from typing import Any, List
import asyncio


class RagasEmbeddingsShim:
    """
    RAGAS expects embeddings with:
      - embed_query(text: str) -> List[float]
      - embed_documents(texts: List[str]) -> List[List[float]]

    Some OpenAIEmbeddings versions expose only embed_documents(), so we adapt.
    """

    def __init__(self, base_embeddings: Any):
        self.base = base_embeddings

    def embed_query(self, text: str) -> List[float]:
        # If underlying supports embed_query, use it.
        if hasattr(self.base, "embed_query") and callable(getattr(self.base, "embed_query")):
            return self.base.embed_query(text)

        # Fallback: embed_documents([text]) -> [[vec]]
        vecs = self.base.embed_documents([text])
        if not vecs or not isinstance(vecs, list) or not vecs[0]:
            raise ValueError("embed_documents returned empty embedding for query")
        return vecs[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.base.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        if hasattr(self.base, "aembed_query") and callable(getattr(self.base, "aembed_query")):
            return await self.base.aembed_query(text)

        if hasattr(self.base, "aembed_documents") and callable(getattr(self.base, "aembed_documents")):
            vecs = await self.base.aembed_documents([text])
            return vecs[0]

        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self.base, "aembed_documents") and callable(getattr(self.base, "aembed_documents")):
            return await self.base.aembed_documents(texts)
        return await asyncio.to_thread(self.embed_documents, texts)
