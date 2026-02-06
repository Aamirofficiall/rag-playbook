"""Tests for the exception hierarchy."""

import pytest

from rag_playbook.core.exceptions import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    PatternError,
    ProviderError,
    RAGPlaybookError,
    VectorStoreError,
)


@pytest.mark.unit
class TestExceptionHierarchy:
    """Verify the inheritance tree allows precise and broad catching."""

    def test_all_exceptions_inherit_from_base(self) -> None:
        exceptions = [
            ConfigurationError,
            ProviderError,
            GenerationError,
            EmbeddingError,
            VectorStoreError,
            ChunkingError,
            EvaluationError,
            PatternError,
        ]
        for exc_cls in exceptions:
            assert issubclass(exc_cls, RAGPlaybookError)

    def test_generation_error_is_provider_error(self) -> None:
        assert issubclass(GenerationError, ProviderError)

    def test_embedding_error_is_provider_error(self) -> None:
        assert issubclass(EmbeddingError, ProviderError)

    def test_catching_base_catches_all_subtypes(self) -> None:
        with pytest.raises(RAGPlaybookError):
            raise GenerationError("LLM timed out")

    def test_catching_provider_catches_generation_and_embedding(self) -> None:
        with pytest.raises(ProviderError):
            raise GenerationError("rate limited")

        with pytest.raises(ProviderError):
            raise EmbeddingError("dimension mismatch")

    def test_exception_preserves_message(self) -> None:
        msg = "OPENAI_API_KEY not set"
        exc = ConfigurationError(msg)
        assert str(exc) == msg

    def test_exception_is_not_bare_exception(self) -> None:
        """Ensure our hierarchy inherits Exception, not BaseException."""
        assert issubclass(RAGPlaybookError, Exception)
        assert not issubclass(RAGPlaybookError, KeyboardInterrupt)
