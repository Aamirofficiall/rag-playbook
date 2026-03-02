"""Using a custom vector store with rag-playbook.

Shows how to plug in your own BaseVectorStore implementation and use it
with any RAG pattern.

Usage:
    python examples/custom_vector_store.py
"""

from __future__ import annotations

import asyncio

from rag_playbook import Settings, create_pattern
from rag_playbook.core.embedder import create_embedder
from rag_playbook.core.llm import create_llm
from rag_playbook.core.models import Document, EmbeddedChunk
from rag_playbook.core.vector_store import InMemoryVectorStore


async def main() -> None:
    settings = Settings()

    # The default InMemoryVectorStore works out of the box.
    # To use ChromaDB, pgvector, or Qdrant, call create_vector_store():
    #
    #   from rag_playbook.core.vector_store import create_vector_store
    #   store = create_vector_store("chromadb")
    #
    # Or implement BaseVectorStore for any custom backend:
    #
    #   class MyStore(BaseVectorStore):
    #       async def add(self, chunks): ...
    #       async def search(self, query_embedding, top_k=10): ...
    #       async def hybrid_search(self, query_text, query_embedding, ...): ...
    #       async def delete(self, chunk_ids): ...
    #       async def count(self): ...
    #       async def reset(self): ...

    store = InMemoryVectorStore()
    llm = create_llm(settings)
    embedder = create_embedder(settings)

    # Ingest sample documents
    docs = [
        Document(id="d1", content="Python is a high-level programming language."),
        Document(id="d2", content="Vector databases store embeddings for fast search."),
        Document(id="d3", content="RAG combines retrieval with generation."),
    ]

    for doc in docs:
        embeddings = await embedder.embed([doc.content])
        chunk = EmbeddedChunk(
            id=doc.id,
            document_id=doc.id,
            content=doc.content,
            embedding=embeddings[0],
        )
        await store.add([chunk])

    print(f"Store contains {await store.count()} chunks\n")

    # Use the store with any pattern
    pattern = create_pattern("hybrid_search", llm=llm, embedder=embedder, store=store)
    result = await pattern.query("What is RAG?")

    print(f"Answer: {result.answer}")
    print(f"Pattern: {result.metadata.pattern}")
    print(f"Sources: {result.metadata.final_chunk_count} chunks")


if __name__ == "__main__":
    asyncio.run(main())
