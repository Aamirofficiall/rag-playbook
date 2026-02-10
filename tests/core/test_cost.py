"""Tests for cost calculation."""

import pytest

from rag_playbook.core.cost import (
    embedding_cost,
    llm_cost,
    supported_embedding_models,
    supported_llm_models,
)


@pytest.mark.unit
class TestLLMCost:
    def test_gpt4o_mini_cost(self) -> None:
        cost = llm_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_gpt4o_cost(self) -> None:
        cost = llm_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_zero(self) -> None:
        assert llm_cost("unknown-model", 1000, 500) == 0.0

    def test_zero_tokens_returns_zero(self) -> None:
        assert llm_cost("gpt-4o-mini", 0, 0) == 0.0

    def test_output_more_expensive_than_input(self) -> None:
        input_only = llm_cost("gpt-4o", input_tokens=1000, output_tokens=0)
        output_only = llm_cost("gpt-4o", input_tokens=0, output_tokens=1000)
        assert output_only > input_only


@pytest.mark.unit
class TestEmbeddingCost:
    def test_text_embedding_3_small_cost(self) -> None:
        cost = embedding_cost("text-embedding-3-small", token_count=10_000)
        expected = 10_000 * 0.02 / 1_000_000
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_zero(self) -> None:
        assert embedding_cost("unknown-model", 1000) == 0.0

    def test_zero_tokens_returns_zero(self) -> None:
        assert embedding_cost("text-embedding-3-small", 0) == 0.0


@pytest.mark.unit
class TestSupportedModels:
    def test_supported_llm_models_includes_openai(self) -> None:
        models = supported_llm_models()
        assert "gpt-4o-mini" in models
        assert "gpt-4o" in models

    def test_supported_llm_models_includes_anthropic(self) -> None:
        models = supported_llm_models()
        assert "claude-sonnet-4-6" in models

    def test_supported_embedding_models(self) -> None:
        models = supported_embedding_models()
        assert "text-embedding-3-small" in models
        assert "text-embedding-3-large" in models
