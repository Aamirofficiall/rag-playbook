"""Tests for prompt templates."""

import pytest

from rag_playbook.core import prompts


@pytest.mark.unit
class TestPromptTemplates:
    """Verify all prompts are well-formed and substitutable."""

    def test_rag_user_prompt_has_placeholders(self) -> None:
        result = prompts.RAG_USER_PROMPT.format(context="Some context", question="What is X?")
        assert "Some context" in result
        assert "What is X?" in result

    def test_decompose_prompt_has_question_placeholder(self) -> None:
        result = prompts.DECOMPOSE_PROMPT.format(question="Complex query?")
        assert "Complex query?" in result

    def test_hyde_prompt_has_question_placeholder(self) -> None:
        result = prompts.HYDE_PROMPT.format(question="Short query?")
        assert "Short query?" in result

    def test_faithfulness_prompt_has_placeholders(self) -> None:
        result = prompts.FAITHFULNESS_PROMPT.format(context="ctx", answer="ans")
        assert "ctx" in result
        assert "ans" in result

    def test_recommend_prompt_has_question_placeholder(self) -> None:
        result = prompts.RECOMMEND_PROMPT.format(question="My query?")
        assert "My query?" in result

    def test_relevance_judge_prompt_has_placeholders(self) -> None:
        result = prompts.RELEVANCE_JUDGE_PROMPT.format(question="q", chunk="c")
        assert "q" in result
        assert "c" in result

    def test_answer_faithfulness_judge_has_placeholders(self) -> None:
        result = prompts.ANSWER_FAITHFULNESS_JUDGE_PROMPT.format(context="ctx", answer="ans")
        assert "ctx" in result
        assert "ans" in result

    def test_answer_relevance_judge_has_placeholders(self) -> None:
        result = prompts.ANSWER_RELEVANCE_JUDGE_PROMPT.format(question="q", answer="a")
        assert "q" in result
        assert "a" in result

    def test_rag_system_prompt_is_not_empty(self) -> None:
        assert len(prompts.RAG_SYSTEM_PROMPT) > 50

    def test_agentic_system_prompt_mentions_tools(self) -> None:
        assert "search" in prompts.AGENTIC_SYSTEM_PROMPT
        assert "answer" in prompts.AGENTIC_SYSTEM_PROMPT
