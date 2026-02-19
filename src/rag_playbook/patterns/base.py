"""Base RAG pattern using the Template Method design pattern.

The ``query()`` method is the fixed skeleton — it orchestrates timing,
cost tracking, logging, and result assembly. Subclasses override the
individual steps (preprocess, retrieve, postprocess, generate, validate)
to implement different RAG strategies.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import structlog

from rag_playbook.core.config import Settings
from rag_playbook.core.embedder import BaseEmbedder
from rag_playbook.core.llm import BaseLLM, LLMResponse, Message
from rag_playbook.core.models import (
    QueryMetadata,
    RAGResult,
    RetrievedChunk,
    StepTiming,
)
from rag_playbook.core.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from rag_playbook.core.vector_store import BaseVectorStore


class BaseRAGPattern(ABC):
    """Template Method base for all RAG patterns.

    Override individual steps to create a new pattern. Do NOT override
    ``query()`` — it handles the orchestration contract.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        config: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.config = config or Settings()
        self.logger = structlog.get_logger(pattern=self.pattern_name)

    @property
    @abstractmethod
    def pattern_name(self) -> str:
        """Unique identifier for this pattern (e.g. 'reranking')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description for CLI output."""

    # -- Template Method (fixed skeleton) ------------------------------------

    async def query(self, question: str, top_k: int | None = None) -> RAGResult:
        """Run the full RAG pipeline. This method is final — do not override."""
        steps: list[StepTiming] = []
        top_k = top_k or self.config.default_top_k
        total_start = time.perf_counter()

        # Step 1: Preprocess
        t0 = time.perf_counter()
        processed_queries = await self.preprocess_query(question)
        steps.append(
            StepTiming(
                step="preprocess",
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail=f"{len(processed_queries)} queries" if len(processed_queries) > 1 else None,
            )
        )

        # Step 2: Retrieve
        t0 = time.perf_counter()
        all_chunks: list[RetrievedChunk] = []
        for q in processed_queries:
            chunks = await self.retrieve(q, top_k)
            all_chunks.extend(chunks)
        steps.append(
            StepTiming(
                step="retrieve",
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail=f"{len(all_chunks)} chunks",
            )
        )

        # Step 3: Postprocess
        t0 = time.perf_counter()
        final_chunks = await self.postprocess_chunks(question, all_chunks)
        steps.append(
            StepTiming(
                step="postprocess",
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail=f"{len(all_chunks)}\u2192{len(final_chunks)} chunks",
            )
        )

        # Step 4: Generate
        t0 = time.perf_counter()
        llm_response = await self.generate(question, final_chunks)
        steps.append(
            StepTiming(
                step="generate",
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail=f"{llm_response.input_tokens}+{llm_response.output_tokens} tokens",
            )
        )

        # Step 5: Validate
        t0 = time.perf_counter()
        validated = await self.validate(question, llm_response.content, final_chunks)
        steps.append(
            StepTiming(
                step="validate",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        total_ms = (time.perf_counter() - total_start) * 1000

        return RAGResult(
            answer=validated,
            sources=final_chunks,
            metadata=QueryMetadata(
                pattern=self.pattern_name,
                model=self.llm.model_name,
                embedding_model=self.embedder.model_name,
                tokens_used=llm_response.input_tokens + llm_response.output_tokens,
                latency_ms=total_ms,
                cost_usd=llm_response.cost_usd,
                retrieval_count=len(all_chunks),
                final_chunk_count=len(final_chunks),
                steps=steps,
            ),
        )

    # -- Overridable steps ---------------------------------------------------

    async def preprocess_query(self, question: str) -> list[str]:
        """Preprocess the query. Override for decomposition / HyDE."""
        return [question]

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve chunks from the vector store."""
        embeddings = await self.embedder.embed([query])
        return await self.store.search(embeddings[0], top_k=top_k)

    async def postprocess_chunks(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Postprocess retrieved chunks. Override for reranking / parent expansion."""
        return chunks

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> LLMResponse:
        """Generate an answer from the question and context chunks."""
        context = "\n\n---\n\n".join(c.content for c in chunks)
        messages = [
            Message(role="system", content=RAG_SYSTEM_PROMPT),
            Message(
                role="user",
                content=RAG_USER_PROMPT.format(context=context, question=question),
            ),
        ]
        return await self.llm.generate(messages)

    async def validate(self, question: str, answer: str, chunks: list[RetrievedChunk]) -> str:
        """Validate the answer. Override for self-correcting patterns."""
        return answer
