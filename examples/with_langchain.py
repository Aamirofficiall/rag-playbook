"""Using rag-playbook alongside LangChain.

Shows how to evaluate a LangChain pipeline using rag-playbook's evaluator,
or use rag-playbook patterns as drop-in alternatives.

Usage:
    python examples/with_langchain.py
"""

from __future__ import annotations

import asyncio

from rag_playbook import Settings
from rag_playbook.core.evaluator import Evaluator
from rag_playbook.core.llm import create_llm


async def main() -> None:
    settings = Settings()

    # --- Scenario 1: Use rag-playbook's evaluator on LangChain output ---
    #
    # If you already have a LangChain RAG pipeline, you can evaluate its
    # quality using rag-playbook's LLM-as-judge metrics.

    llm = create_llm(settings)
    evaluator = Evaluator(llm=llm)

    # Simulate output from your LangChain pipeline
    query = "What is the company refund policy?"
    langchain_answer = "Refunds are processed within 14 days of the request."
    retrieved_contexts = [
        "Our refund policy allows returns within 30 days. "
        "Refunds are processed within 14 business days of receiving the return.",
        "Contact support@example.com for refund status inquiries.",
    ]

    # Evaluate with free metrics (no LLM cost)
    scores = evaluator.evaluate_free(
        query=query,
        answer=langchain_answer,
        contexts=retrieved_contexts,
    )
    print("Free metrics (no API cost):")
    print(f"  Chunk utilization: {scores.chunk_utilization:.2f}")
    print(f"  Latency:           {scores.latency_ms:.0f} ms")
    print(f"  Cost:              ${scores.cost_usd:.6f}")
    print()

    # Evaluate with LLM-as-judge (costs ~$0.003)
    judge_scores = await evaluator.evaluate_with_judges(
        query=query,
        answer=langchain_answer,
        contexts=retrieved_contexts,
    )
    print("LLM-as-judge metrics:")
    print(f"  Retrieval relevance:  {judge_scores.retrieval_relevance:.2f}")
    print(f"  Answer faithfulness:  {judge_scores.answer_faithfulness:.2f}")
    print(f"  Answer relevance:     {judge_scores.answer_relevance:.2f}")
    print()

    # --- Scenario 2: Compare LangChain vs rag-playbook ---
    #
    # Run the same query through both systems and compare metrics.
    # Use rag-playbook's compare command for systematic comparison:
    #
    #   rag-playbook compare --data ./docs/ --query "What is the refund policy?"
    #
    # Then compare the scores against your LangChain pipeline's output.

    print("To compare with rag-playbook patterns, run:")
    print('  rag-playbook compare --data ./docs/ --query "What is the refund policy?"')


if __name__ == "__main__":
    asyncio.run(main())
