"""Pattern 01: Naive RAG — the baseline.

Embed the query, retrieve top-K chunks by vector similarity, generate
an answer. No postprocessing, no validation. Every other pattern is
measured against this.
"""

from __future__ import annotations

from rag_playbook.patterns.base import BaseRAGPattern


class NaiveRAG(BaseRAGPattern):
    """Baseline RAG: embed → retrieve → generate."""

    _pattern_name = "naive"

    @property
    def pattern_name(self) -> str:
        return self._pattern_name

    @property
    def description(self) -> str:
        return "Simple embed → retrieve → generate (baseline)"
