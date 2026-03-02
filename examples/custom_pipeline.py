"""Build a custom RAG pipeline from individual components.

Demonstrates how to use the chunker, embedder, and vector store
independently -- useful when you need fine-grained control over each
pipeline stage.

Set your API key before running:
    export RAG_OPENAI_API_KEY="sk-..."

Usage:
    python examples/custom_pipeline.py
"""

import asyncio

from rag_playbook import Document, Settings
from rag_playbook.core.chunker import create_chunker
from rag_playbook.core.embedder import create_embedder
from rag_playbook.core.models import EmbeddedChunk
from rag_playbook.core.vector_store import create_vector_store

LONG_DOC = Document(
    id="architecture",
    content=(
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with text generation. The pipeline has several "
        "stages.\n\n"
        "First, documents are split into chunks. Chunking strategies include "
        "fixed-size splitting, recursive splitting by paragraph and sentence "
        "boundaries, and structural splitting on headings.\n\n"
        "Second, each chunk is embedded into a dense vector using a model "
        "like text-embedding-3-small. These vectors capture semantic meaning "
        "so that similar content clusters together.\n\n"
        "Third, vectors are stored in a vector database such as ChromaDB, "
        "Qdrant, or pgvector. At query time the user question is embedded "
        "and the nearest neighbors are retrieved.\n\n"
        "Finally, the retrieved chunks are passed as context to an LLM, "
        "which generates a grounded answer. Advanced patterns add reranking, "
        "query decomposition, or self-correction on top of this baseline."
    ),
)


async def main() -> None:
    settings = Settings()

    # -- Step 1: Chunk -------------------------------------------------------
    chunker = create_chunker(strategy="recursive", chunk_size=128, chunk_overlap=20)
    chunks = chunker.chunk(LONG_DOC)
    print(f"Chunked document into {len(chunks)} pieces (recursive, 128 tokens):\n")
    for c in chunks:
        preview = c.content[:80].replace("\n", " ")
        print(f"  [{c.metadata.chunk_index}] {preview}...")

    # -- Step 2: Embed -------------------------------------------------------
    embedder = create_embedder(settings)
    texts = [c.content for c in chunks]
    vectors = await embedder.embed(texts)
    print(f"\nEmbedded {len(vectors)} chunks (dim={len(vectors[0])}).")

    # -- Step 3: Store -------------------------------------------------------
    store = create_vector_store("memory")
    embedded_chunks = [
        EmbeddedChunk(
            id=c.id,
            document_id=c.document_id,
            content=c.content,
            metadata=c.metadata,
            embedding=vec,
        )
        for c, vec in zip(chunks, vectors)
    ]
    await store.add(embedded_chunks)
    print(f"Stored {await store.count()} chunks in the vector store.")

    # -- Step 4: Search ------------------------------------------------------
    query = "How are vectors used in RAG?"
    query_vec = (await embedder.embed([query]))[0]
    results = await store.search(query_vec, top_k=3)

    print(f"\nSearch: \"{query}\"")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        snippet = r.content[:100].replace("\n", " ")
        print(f"  {i}. [score={r.score:.4f}] {snippet}...")


if __name__ == "__main__":
    asyncio.run(main())
