"""Tests for core data models.

Verifies serialization, immutability, validation, and pipeline stage
separation — the guarantees that keep the RAG pipeline correct.
"""

import json

import pytest

from rag_playbook.core.models import (
    Chunk,
    ChunkMetadata,
    ChunkStrategy,
    ComparisonResult,
    Document,
    DocumentMetadata,
    EmbeddedChunk,
    QueryMetadata,
    RAGResult,
    RetrievalMethod,
    RetrievedChunk,
    StepTiming,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document() -> Document:
    return Document(
        id="doc-1",
        content="Refunds are processed within 14 business days.",
        metadata=DocumentMetadata(
            title="Refund Policy",
            source="policies/refund.md",
            format="markdown",
            page_count=1,
        ),
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Refunds are processed within 14 business days.",
        metadata=ChunkMetadata(chunk_index=0, overlap_tokens=0),
    )


@pytest.fixture
def sample_embedding() -> list[float]:
    return [0.1] * 1536


@pytest.fixture
def sample_embedded_chunk(sample_embedding: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        id="chunk-1",
        document_id="doc-1",
        content="Refunds are processed within 14 business days.",
        embedding=sample_embedding,
    )


@pytest.fixture
def sample_retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        id="chunk-1",
        document_id="doc-1",
        content="Refunds are processed within 14 business days.",
        score=0.92,
        retrieval_method=RetrievalMethod.VECTOR,
    )


@pytest.fixture
def sample_query_metadata() -> QueryMetadata:
    return QueryMetadata(
        pattern="naive",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        tokens_used=611,
        latency_ms=980.0,
        cost_usd=0.003,
        retrieval_count=5,
        final_chunk_count=5,
        steps=[
            StepTiming(step="retrieve", latency_ms=280.0, detail="5 chunks"),
            StepTiming(step="generate", latency_ms=700.0),
        ],
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSerialization:
    def test_document_roundtrip(self, sample_document: Document) -> None:
        data = sample_document.model_dump()
        restored = Document.model_validate(data)
        assert restored == sample_document

    def test_document_json_roundtrip(self, sample_document: Document) -> None:
        json_str = sample_document.model_dump_json()
        restored = Document.model_validate_json(json_str)
        assert restored == sample_document

    def test_embedded_chunk_roundtrip(self, sample_embedded_chunk: EmbeddedChunk) -> None:
        data = sample_embedded_chunk.model_dump()
        restored = EmbeddedChunk.model_validate(data)
        assert restored == sample_embedded_chunk

    def test_rag_result_to_json(
        self,
        sample_retrieved_chunk: RetrievedChunk,
        sample_query_metadata: QueryMetadata,
    ) -> None:
        result = RAGResult(
            answer="Refunds take 14 business days.",
            sources=[sample_retrieved_chunk],
            metadata=sample_query_metadata,
        )
        parsed = json.loads(result.model_dump_json())
        assert parsed["answer"] == "Refunds take 14 business days."
        assert len(parsed["sources"]) == 1
        assert parsed["metadata"]["pattern"] == "naive"

    def test_comparison_result_roundtrip(
        self,
        sample_retrieved_chunk: RetrievedChunk,
        sample_query_metadata: QueryMetadata,
    ) -> None:
        rag_result = RAGResult(
            answer="14 days",
            sources=[sample_retrieved_chunk],
            metadata=sample_query_metadata,
        )
        comparison = ComparisonResult(
            query="What is the refund policy?",
            results={"naive": rag_result},
            recommendation="naive",
            recommendation_reasoning="Simple query, baseline sufficient.",
        )
        restored = ComparisonResult.model_validate_json(comparison.model_dump_json())
        assert restored == comparison


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImmutability:
    def test_document_is_frozen(self, sample_document: Document) -> None:
        with pytest.raises(Exception):  # noqa: B017
            sample_document.content = "changed"  # type: ignore[misc]

    def test_chunk_is_frozen(self, sample_chunk: Chunk) -> None:
        with pytest.raises(Exception):  # noqa: B017
            sample_chunk.content = "changed"  # type: ignore[misc]

    def test_embedded_chunk_is_frozen(self, sample_embedded_chunk: EmbeddedChunk) -> None:
        with pytest.raises(Exception):  # noqa: B017
            sample_embedded_chunk.embedding = [0.0]  # type: ignore[misc]

    def test_retrieved_chunk_is_frozen(self, sample_retrieved_chunk: RetrievedChunk) -> None:
        with pytest.raises(Exception):  # noqa: B017
            sample_retrieved_chunk.score = 0.5  # type: ignore[misc]

    def test_step_timing_is_frozen(self) -> None:
        timing = StepTiming(step="retrieve", latency_ms=100.0)
        with pytest.raises(Exception):  # noqa: B017
            timing.latency_ms = 200.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidation:
    def test_embedded_chunk_rejects_empty_embedding(self) -> None:
        with pytest.raises(ValueError, match="embedding must not be empty"):
            EmbeddedChunk(
                id="c1",
                document_id="d1",
                content="text",
                embedding=[],
            )

    def test_retrieved_chunk_score_must_be_between_0_and_1(self) -> None:
        with pytest.raises(ValueError):
            RetrievedChunk(
                id="c1",
                document_id="d1",
                content="text",
                score=1.5,
                retrieval_method=RetrievalMethod.VECTOR,
            )

        with pytest.raises(ValueError):
            RetrievedChunk(
                id="c1",
                document_id="d1",
                content="text",
                score=-0.1,
                retrieval_method=RetrievalMethod.VECTOR,
            )

    def test_step_timing_rejects_negative_latency(self) -> None:
        with pytest.raises(ValueError):
            StepTiming(step="retrieve", latency_ms=-1.0)

    def test_query_metadata_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValueError):
            QueryMetadata(
                pattern="naive",
                model="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                tokens_used=-1,
                latency_ms=100.0,
                cost_usd=0.001,
                retrieval_count=5,
                final_chunk_count=5,
            )

    def test_query_metadata_rejects_negative_cost(self) -> None:
        with pytest.raises(ValueError):
            QueryMetadata(
                pattern="naive",
                model="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                tokens_used=100,
                latency_ms=100.0,
                cost_usd=-0.001,
                retrieval_count=5,
                final_chunk_count=5,
            )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaults:
    def test_document_default_metadata(self) -> None:
        doc = Document(id="d1", content="hello")
        assert doc.metadata.title == ""
        assert doc.metadata.page_count == 0

    def test_chunk_default_metadata(self) -> None:
        chunk = Chunk(id="c1", document_id="d1", content="hello")
        assert chunk.metadata.chunk_index == 0
        assert chunk.metadata.overlap_tokens == 0

    def test_query_metadata_default_steps(self) -> None:
        meta = QueryMetadata(
            pattern="naive",
            model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            tokens_used=100,
            latency_ms=500.0,
            cost_usd=0.001,
            retrieval_count=5,
            final_chunk_count=5,
        )
        assert meta.steps == []


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnums:
    def test_retrieval_method_values(self) -> None:
        assert RetrievalMethod.VECTOR == "vector"
        assert RetrievalMethod.HYBRID == "hybrid"
        assert RetrievalMethod.RERANKED == "reranked"

    def test_chunk_strategy_values(self) -> None:
        assert ChunkStrategy.FIXED == "fixed"
        assert ChunkStrategy.SEMANTIC == "semantic"
        assert ChunkStrategy.STRUCTURAL == "structural"

    def test_retrieval_method_is_string(self) -> None:
        assert isinstance(RetrievalMethod.VECTOR, str)
