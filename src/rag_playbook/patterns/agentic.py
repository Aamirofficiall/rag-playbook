"""Pattern 08: Agentic RAG.

The LLM decides when, what, and how to search. It operates in a
tool-calling loop: the model can issue search calls, inspect results,
and decide whether to search again or provide a final answer.
Maximum 5 search iterations to bound cost.
"""

from __future__ import annotations

import json

from rag_playbook.core.llm import Message
from rag_playbook.core.models import (
    QueryMetadata,
    RAGResult,
    RetrievedChunk,
    StepTiming,
)
from rag_playbook.core.prompts import AGENTIC_SYSTEM_PROMPT
from rag_playbook.patterns.base import BaseRAGPattern

_MAX_ITERATIONS = 5


class AgenticRAG(BaseRAGPattern):
    """LLM decides when/what/how to search — tool-calling loop."""

    _pattern_name = "agentic"

    @property
    def pattern_name(self) -> str:
        return self._pattern_name

    @property
    def description(self) -> str:
        return "LLM decides when/what/how to search"

    async def query(self, question: str, top_k: int | None = None) -> RAGResult:
        """Override the full query flow with a tool-calling loop."""
        import time

        top_k = top_k or self.config.default_top_k
        total_start = time.perf_counter()
        steps: list[StepTiming] = []
        all_chunks: list[RetrievedChunk] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        messages = [
            Message(role="system", content=AGENTIC_SYSTEM_PROMPT),
            Message(role="user", content=question),
        ]

        for iteration in range(1, _MAX_ITERATIONS + 1):
            t0 = time.perf_counter()
            response = await self.llm.generate(messages)
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            total_cost += response.cost_usd

            tool_call = self._parse_tool_call(response.content)

            if tool_call is None or tool_call["tool"] == "answer":
                answer_text = (
                    tool_call["args"]["text"]
                    if tool_call and "text" in tool_call.get("args", {})
                    else response.content
                )
                steps.append(
                    StepTiming(
                        step=f"iteration_{iteration}",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        detail="final answer",
                    )
                )
                break

            if tool_call["tool"] == "search":
                search_query = tool_call["args"].get("query", question)
                embeddings = await self.embedder.embed([search_query])
                chunks = await self.store.search(embeddings[0], top_k=top_k)
                all_chunks.extend(chunks)

                chunk_text = "\n\n".join(c.content for c in chunks)
                messages.append(Message(role="assistant", content=response.content))
                messages.append(
                    Message(
                        role="user",
                        content=f"Search results:\n{chunk_text}",
                    )
                )

                steps.append(
                    StepTiming(
                        step=f"iteration_{iteration}",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        detail=f'search("{search_query[:50]}") -> {len(chunks)} chunks',
                    )
                )
        else:
            # Exhausted iterations — generate final answer from gathered context
            final_response = await self.generate(question, all_chunks)
            answer_text = final_response.content
            total_input_tokens += final_response.input_tokens
            total_output_tokens += final_response.output_tokens
            total_cost += final_response.cost_usd

        total_ms = (time.perf_counter() - total_start) * 1000

        # Deduplicate chunks by ID
        seen_ids: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []
        for chunk in all_chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)

        return RAGResult(
            answer=answer_text,
            sources=unique_chunks,
            metadata=QueryMetadata(
                pattern=self.pattern_name,
                model=self.llm.model_name,
                embedding_model=self.embedder.model_name,
                tokens_used=total_input_tokens + total_output_tokens,
                latency_ms=total_ms,
                cost_usd=total_cost,
                retrieval_count=len(all_chunks),
                final_chunk_count=len(unique_chunks),
                steps=steps,
            ),
        )

    @staticmethod
    def _parse_tool_call(text: str) -> dict | None:
        """Extract a tool call JSON from the LLM response."""
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and "tool" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON embedded in text by scanning for balanced braces
        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(text[start : i + 1])
                        if isinstance(candidate, dict) and "tool" in candidate:
                            return candidate
                    except json.JSONDecodeError:
                        pass
                    break
            start = text.find("{", start + 1)

        return None
