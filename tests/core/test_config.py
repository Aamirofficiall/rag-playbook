"""Tests for application configuration."""

import pytest

from rag_playbook.core.config import Settings


@pytest.mark.unit
class TestSettingsDefaults:
    """Verify that Settings works with zero configuration."""

    def test_creates_with_defaults(self) -> None:
        settings = Settings()
        assert settings.default_llm_provider == "openai"
        assert settings.default_llm_model == "gpt-4o-mini"

    def test_default_embedding_config(self) -> None:
        settings = Settings()
        assert settings.embedding_provider == "openai"
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.embedding_dimension == 1536

    def test_default_vector_store(self) -> None:
        settings = Settings()
        assert settings.vector_store_provider == "chromadb"

    def test_default_retrieval_config(self) -> None:
        settings = Settings()
        assert settings.default_top_k == 5
        assert settings.hybrid_search_alpha == 0.5

    def test_default_chunking_config(self) -> None:
        settings = Settings()
        assert settings.default_chunk_size == 512
        assert settings.default_chunk_overlap == 50

    def test_default_eval_judge(self) -> None:
        settings = Settings()
        assert settings.eval_judge_model == "gpt-4o-mini"

    def test_default_timeout_and_retries(self) -> None:
        settings = Settings()
        assert settings.llm_timeout_seconds == 30
        assert settings.llm_max_retries == 3


@pytest.mark.unit
class TestSettingsOverrides:
    """Verify that env vars and constructor kwargs override defaults."""

    def test_override_via_constructor(self) -> None:
        settings = Settings(
            default_llm_model="claude-sonnet-4-6",
            default_top_k=10,
        )
        assert settings.default_llm_model == "claude-sonnet-4-6"
        assert settings.default_top_k == 10

    def test_override_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_DEFAULT_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("RAG_DEFAULT_TOP_K", "20")
        settings = Settings()
        assert settings.default_llm_model == "gpt-4o"
        assert settings.default_top_k == 20

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_OPENAI_API_KEY", "sk-test-123")
        settings = Settings()
        assert settings.openai_api_key == "sk-test-123"


@pytest.mark.unit
class TestSettingsValidation:
    """Verify that field constraints are enforced."""

    def test_rejects_negative_timeout(self) -> None:
        with pytest.raises(ValueError):
            Settings(llm_timeout_seconds=0)

    def test_rejects_negative_embedding_dimension(self) -> None:
        with pytest.raises(ValueError):
            Settings(embedding_dimension=0)

    def test_rejects_invalid_alpha(self) -> None:
        with pytest.raises(ValueError):
            Settings(hybrid_search_alpha=1.5)

        with pytest.raises(ValueError):
            Settings(hybrid_search_alpha=-0.1)

    def test_rejects_small_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            Settings(default_chunk_size=32)

    def test_accepts_valid_chunk_size(self) -> None:
        settings = Settings(default_chunk_size=64)
        assert settings.default_chunk_size == 64
