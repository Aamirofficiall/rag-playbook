"""Tests for tiktoken helpers."""

import pytest

from rag_playbook.utils.tokenizer import count_tokens, get_encoder, truncate_to_tokens


@pytest.mark.unit
class TestCountTokens:
    def test_counts_tokens(self) -> None:
        count = count_tokens("Hello, world!")
        assert count > 0

    def test_empty_string_is_zero(self) -> None:
        assert count_tokens("") == 0

    def test_longer_text_has_more_tokens(self) -> None:
        short = count_tokens("hello")
        long = count_tokens("hello world how are you doing today")
        assert long > short


@pytest.mark.unit
class TestTruncateToTokens:
    def test_short_text_unchanged(self) -> None:
        text = "Hello"
        assert truncate_to_tokens(text, max_tokens=100) == text

    def test_long_text_truncated(self) -> None:
        text = " ".join(f"word{i}" for i in range(500))
        result = truncate_to_tokens(text, max_tokens=10)
        assert count_tokens(result) <= 10

    def test_empty_string(self) -> None:
        assert truncate_to_tokens("", max_tokens=10) == ""


@pytest.mark.unit
class TestGetEncoder:
    def test_returns_encoder(self) -> None:
        enc = get_encoder()
        tokens = enc.encode("test")
        assert len(tokens) > 0

    def test_caches_encoder(self) -> None:
        enc1 = get_encoder("cl100k_base")
        enc2 = get_encoder("cl100k_base")
        assert enc1 is enc2
