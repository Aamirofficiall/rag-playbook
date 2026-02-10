"""Centralized prompt templates for all LLM interactions.

Every prompt the system sends lives here — not scattered across pattern files.
This makes it trivial to audit, test, and swap prompts for different models.

Each template uses ``{placeholder}`` syntax for ``str.format()`` substitution.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context. If the context doesn't contain the answer, say \"I don't have "
    'enough information to answer this question." Do not make up information.'
)

RAG_USER_PROMPT = """Context:
{context}

Question: {question}

Answer:"""

# ---------------------------------------------------------------------------
# Query Decomposition (Pattern 05)
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = (
    "Given the following question, break it down into 2-4 simpler "
    "sub-questions that together fully answer the original. Each "
    "sub-question must be self-contained. If the question is already "
    "simple, return it as-is.\n"
    "Respond with a JSON array of strings. Nothing else.\n\n"
    "Question: {question}"
)

# ---------------------------------------------------------------------------
# HyDE (Pattern 06)
# ---------------------------------------------------------------------------

HYDE_PROMPT = (
    "Write a short paragraph that would answer the following question. "
    "It doesn't need to be accurate — just write what a good answer "
    "would look like.\n\n"
    "Question: {question}"
)

# ---------------------------------------------------------------------------
# Faithfulness Check (Pattern 07)
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """Check if every claim in this answer is supported by the provided context.

Context:
{context}

Answer:
{answer}

Respond with JSON:
{{"is_faithful": true/false, "unsupported_claims": ["..."], "confidence": 0.0-1.0}}"""

# ---------------------------------------------------------------------------
# Agentic Tool System (Pattern 08)
# ---------------------------------------------------------------------------

AGENTIC_SYSTEM_PROMPT = (
    "You have access to a search tool. Use it to find information needed "
    "to answer the user's question.\n\n"
    "Available tools:\n"
    "- search(query: str) → returns relevant text chunks\n"
    "- answer(text: str) → provide your final answer\n\n"
    "Call tools by responding with JSON: "
    '{{"tool": "search", "args": {{"query": "..."}}}}\n'
    "When you have enough information: "
    '{{"tool": "answer", "args": {{"text": "..."}}}}\n\n'
    "Maximum 5 search calls. Be efficient."
)

# ---------------------------------------------------------------------------
# Recommendation (CLI recommend command)
# ---------------------------------------------------------------------------

RECOMMEND_PROMPT = """Analyze this query and suggest the best RAG pattern.

Query: {question}

Consider:
- Is it a simple factual lookup? → naive or hybrid_search
- Does it contain specific codes/IDs/names? → hybrid_search
- Does it need high precision from many candidates? → reranking
- Is the answer spread across long documents? → parent_child
- Is it a complex multi-part question? → query_decomposition
- Is it very short or ambiguous? → hyde
- Does accuracy matter more than speed? → self_correcting
- Is intent unclear and needs exploration? → agentic

Respond with JSON:
{{"pattern": "...", "reasoning": "...", "confidence": 0.0-1.0}}"""

# ---------------------------------------------------------------------------
# Evaluation (LLM-as-judge)
# ---------------------------------------------------------------------------

RELEVANCE_JUDGE_PROMPT = """Rate how relevant this text chunk is to the given query.

Query: {question}
Chunk: {chunk}

Respond with a single integer from 1 to 5:
1 = completely irrelevant
2 = slightly relevant
3 = moderately relevant
4 = highly relevant
5 = perfectly relevant

Score:"""

ANSWER_FAITHFULNESS_JUDGE_PROMPT = """\
Evaluate whether every claim in the answer is supported by the context.

Context:
{context}

Answer:
{answer}

Respond with a single float from 0.0 to 1.0 representing the faithfulness score.
1.0 = every claim is supported
0.0 = no claims are supported

Score:"""

ANSWER_RELEVANCE_JUDGE_PROMPT = """Rate how well this answer addresses the question.

Question: {question}
Answer: {answer}

Respond with a single integer from 1 to 5:
1 = does not address the question at all
2 = barely addresses the question
3 = partially addresses the question
4 = mostly addresses the question
5 = fully and directly addresses the question

Score:"""
