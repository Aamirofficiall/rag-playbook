"""Tiktoken helpers for accurate token counting."""

from __future__ import annotations

import tiktoken

_ENCODER_CACHE: dict[str, tiktoken.Encoding] = {}


def get_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get a cached tiktoken encoder."""
    if encoding_name not in _ENCODER_CACHE:
        _ENCODER_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODER_CACHE[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a string."""
    return len(get_encoder(encoding_name).encode(text))


def truncate_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Truncate text to fit within a token budget."""
    encoder = get_encoder(encoding_name)
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])
