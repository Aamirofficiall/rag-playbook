"""Shared test fixtures for rag-playbook.

Provides mock LLM, embedder, and vector store that return deterministic
results for unit testing without external API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pytest

from rag_playbook.core.config import Settings
from rag_playbook.core.embedder import BaseEmbedder
from rag_playbook.core.llm import BaseLLM, Message
from rag_playbook.core.models import ChunkMetadata, EmbeddedChunk
from rag_playbook.core.vector_store import InMemoryVectorStore


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent .env file and real env vars from polluting unit tests."""
    for key in list(os.environ):
        if key.startswith("RAG_"):
            monkeypatch.delenv(key, raising=False)
    # Change to a nonexistent dir so pydantic-settings can't find .env
    monkeypatch.chdir("/tmp")


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM(BaseLLM):
    """Deterministic LLM that routes responses based on prompt content.

    - "decompose" in prompt → JSON list of sub-questions
    - "faithful" in prompt → faithfulness check JSON
    - "hypothetical" / "answer would look like" → hypothetical answer
    - "relevance" / "rate" → numeric score
    - "tool" / "search" in system → agentic tool response
    - default → answer that references context
    """

    def __init__(self) -> None:
        super().__init__(Settings())
        self.call_count = 0
        self.last_messages: list[Message] = []

    @property
    def model_name(self) -> str:
        return "mock-llm"

    async def _call(self, messages: list[Message], **kwargs: Any) -> tuple[str, int, int]:
        self.call_count += 1
        self.last_messages = messages

        combined = " ".join(m.content for m in messages).lower()

        if "decompose" in combined or "break it down" in combined:
            return json.dumps(["What is the refund policy?", "How long do refunds take?"]), 50, 30

        if "faithful" in combined:
            payload = {"is_faithful": True, "unsupported_claims": [], "confidence": 0.9}
            return json.dumps(payload), 80, 40

        if "hypothetical" in combined or "answer would look like" in combined:
            return "Refunds are typically processed within 14 business days of the request.", 40, 25

        if "rate" in combined or "relevance" in combined or "score" in combined:
            return "0.85", 20, 5

        if any("search" in m.content.lower() and m.role == "system" for m in messages):
            if "search results" in combined:
                answer_payload = {
                    "tool": "answer",
                    "args": {"text": "Based on search results, refunds take 14 days."},
                }
                return json.dumps(answer_payload), 60, 30
            search_payload = {"tool": "search", "args": {"query": "refund policy"}}
            return json.dumps(search_payload), 40, 20

        return "Based on the context, refunds are processed within 14 business days.", 60, 30


# ---------------------------------------------------------------------------
# Mock Embedder
# ---------------------------------------------------------------------------


class MockEmbedder(BaseEmbedder):
    """Hash-based deterministic embedder.

    Same text → same vector every time. Dimension = 4 for fast tests.
    """

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-embedder"

    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return [int(digest[i : i + 2], 16) / 255.0 for i in range(0, self._dim * 2, 2)]


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    EmbeddedChunk(
        id="chunk-1",
        document_id="doc-1",
        content="Refunds are processed within 14 business days of the request.",
        metadata=ChunkMetadata(chunk_index=0),
        embedding=[0.9, 0.1, 0.0, 0.0],
    ),
    EmbeddedChunk(
        id="chunk-2",
        document_id="doc-1",
        content="Shipping takes 5-7 business days for domestic orders.",
        metadata=ChunkMetadata(chunk_index=1),
        embedding=[0.0, 0.9, 0.1, 0.0],
    ),
    EmbeddedChunk(
        id="chunk-3",
        document_id="doc-1",
        content="Customer support is available 24/7 via chat and email.",
        metadata=ChunkMetadata(chunk_index=2),
        embedding=[0.0, 0.0, 0.9, 0.1],
    ),
    EmbeddedChunk(
        id="chunk-4",
        document_id="doc-2",
        content="Premium customers receive expedited refund processing within 3 days.",
        metadata=ChunkMetadata(chunk_index=0),
        embedding=[0.8, 0.2, 0.0, 0.0],
    ),
    EmbeddedChunk(
        id="chunk-5",
        document_id="doc-2",
        content="International shipping rates vary by destination country.",
        metadata=ChunkMetadata(chunk_index=1),
        embedding=[0.1, 0.8, 0.0, 0.1],
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
async def mock_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.add(SAMPLE_CHUNKS)
    return store


@pytest.fixture
def settings() -> Settings:
    return Settings()
