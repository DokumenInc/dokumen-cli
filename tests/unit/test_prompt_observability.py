"""Tests for prompt observability logging in test execution.

Issue #591: Adds structured log events to track which prompts are applied
to executor and judge agents, enabling CI log inspection and debugging.

TDD: Tests written first per CLAUDE.md rules.
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from dokumen.agent_object import AgentType, ExecutorOutput, JudgeResult
from dokumen.test_object import TestObject, TestResult


@pytest.fixture
def mock_provider():
    """Create a mock provider that returns successful executor output."""
    provider = MagicMock()
    return provider


@pytest.fixture
def mock_executor(mock_provider):
    """Create a mock executor agent."""
    executor = MagicMock()
    executor.id = "test-executor"
    executor.agent_type = AgentType.EXECUTOR
    executor.system_prompt = "You are a documentation validator."
    executor.user_prompt = "Read docs/api.md and verify the endpoints."
    executor.tools = []
    executor.provider = mock_provider
    executor.provider.model = "claude-sonnet-4-5-20250929"
    executor.run = AsyncMock(return_value=ExecutorOutput(
        tool_calls=[],
        final_response="The API has 3 endpoints.",
        success=True,
        error=None,
        input_tokens=100,
        output_tokens=50,
    ))
    return executor


@pytest.fixture
def mock_judge(mock_provider):
    """Create a mock judge agent."""
    judge = MagicMock()
    judge.id = "accuracy-judge"
    judge.agent_type = AgentType.JUDGE
    judge.system_prompt = "Evaluate if the executor correctly identified endpoints."
    judge.tools = []
    judge.provider = mock_provider
    judge.provider.model = "claude-haiku-4-5-20251001"
    judge._get_assertion_text = MagicMock(return_value="accuracy assertion")
    judge.run = AsyncMock(return_value=JudgeResult(
        judge_id="accuracy-judge",
        passed=True,
        failure_reason=None,
        assertion_text="accuracy assertion",
        confidence=0.95,
        input_tokens=80,
        output_tokens=30,
    ))
    return judge


@pytest.fixture
def test_object(mock_executor, mock_judge):
    """Create a TestObject for testing prompt observability.

    Uses files=[] to skip explore phase (explore is tested elsewhere).
    """
    test = TestObject(
        id="test-prompt-obs",
        reason="Test prompt observability logging",
        executor=mock_executor,
        judges=[mock_judge],
        files=[],
        timeout=30,
        retries=0,
    )
    return test


class TestExecutorPromptLogging:
    """Tests that executor prompts are logged before execution."""

    @pytest.mark.asyncio
    async def test_logs_executor_prompt_applied_event(self, test_object):
        """agent.prompt_applied event is logged for executor before execution."""
        with patch("dokumen.test_object.logger") as mock_logger:
            await test_object.run()

            # Find the agent.prompt_applied log call for executor
            prompt_calls = [
                c for c in mock_logger.info.call_args_list
                if c.args and "agent.prompt_applied" in str(c.args[0])
                and any(
                    kv.get("role") == "executor" or kv.get("role") == "executor"
                    for kv in [c.kwargs] if isinstance(kv, dict)
                )
            ]
            assert len(prompt_calls) >= 1, "Should log agent.prompt_applied for executor"

    @pytest.mark.asyncio
    async def test_executor_prompt_log_includes_scaffold_name(self, test_object):
        """agent.prompt_applied for executor includes the test scaffold name."""
        with patch("dokumen.test_object.logger") as mock_logger:
            await test_object.run()

            prompt_calls = [
                c for c in mock_logger.info.call_args_list
                if c.args and "agent.prompt_applied" in str(c.args[0])
            ]
            # At least one call should have the scaffold/test_id
            found = False
            for c in prompt_calls:
                kwargs = c.kwargs
                if kwargs.get("test_id") == "test-prompt-obs" and kwargs.get("role") == "executor":
                    found = True
                    break
            assert found, "Executor prompt log should include test_id"

    @pytest.mark.asyncio
    async def test_executor_prompt_log_includes_prompt_hash(self, test_object):
        """agent.prompt_applied for executor includes hash of system and user prompt."""
        with patch("dokumen.test_object.logger") as mock_logger:
            await test_object.run()

            prompt_calls = [
                c for c in mock_logger.info.call_args_list
                if c.args and "agent.prompt_applied" in str(c.args[0])
                and c.kwargs.get("role") == "executor"
            ]
            assert len(prompt_calls) >= 1
            kwargs = prompt_calls[0].kwargs
            assert "system_prompt_hash" in kwargs, "Should include system_prompt_hash"
            assert "user_prompt_hash" in kwargs, "Should include user_prompt_hash"


class TestJudgePromptLogging:
    """Tests that judge prompts are logged before execution."""

    @pytest.mark.asyncio
    async def test_logs_judge_prompt_applied_event(self, test_object):
        """agent.prompt_applied event is logged for each judge."""
        with patch("dokumen.test_object.logger") as mock_logger:
            await test_object.run()

            judge_prompt_calls = [
                c for c in mock_logger.info.call_args_list
                if c.args and "agent.prompt_applied" in str(c.args[0])
                and c.kwargs.get("role") == "judge"
            ]
            assert len(judge_prompt_calls) >= 1, "Should log agent.prompt_applied for judge"

    @pytest.mark.asyncio
    async def test_judge_prompt_log_includes_judge_name(self, test_object):
        """agent.prompt_applied for judge includes the judge name."""
        with patch("dokumen.test_object.logger") as mock_logger:
            await test_object.run()

            judge_prompt_calls = [
                c for c in mock_logger.info.call_args_list
                if c.args and "agent.prompt_applied" in str(c.args[0])
                and c.kwargs.get("role") == "judge"
            ]
            assert len(judge_prompt_calls) >= 1
            kwargs = judge_prompt_calls[0].kwargs
            assert kwargs.get("judge_name") == "accuracy-judge"


class TestMultipleJudgesPromptLogging:
    """Tests prompt logging with multiple judges."""

    @pytest.mark.asyncio
    async def test_logs_prompt_for_each_judge(self, mock_executor):
        """Each judge gets its own agent.prompt_applied log entry."""
        judge1 = MagicMock()
        judge1.id = "accuracy"
        judge1.agent_type = AgentType.JUDGE
        judge1.system_prompt = "Check accuracy."
        judge1.tools = []
        judge1.provider = MagicMock()
        judge1.provider.model = "claude-haiku-4-5-20251001"
        judge1._get_assertion_text = MagicMock(return_value="accuracy")
        judge1.run = AsyncMock(return_value=JudgeResult(
            judge_id="accuracy", passed=True, assertion_text="accuracy",
            input_tokens=50, output_tokens=20,
        ))

        judge2 = MagicMock()
        judge2.id = "completeness"
        judge2.agent_type = AgentType.JUDGE
        judge2.system_prompt = "Check completeness."
        judge2.tools = []
        judge2.provider = MagicMock()
        judge2.provider.model = "claude-haiku-4-5-20251001"
        judge2._get_assertion_text = MagicMock(return_value="completeness")
        judge2.run = AsyncMock(return_value=JudgeResult(
            judge_id="completeness", passed=True, assertion_text="completeness",
            input_tokens=60, output_tokens=25,
        ))

        test = TestObject(
            id="multi-judge-test",
            reason="Test multi-judge prompt logging",
            executor=mock_executor,
            judges=[judge1, judge2],
            files=[],
            timeout=30,
            retries=0,
        )

        with patch("dokumen.test_object.logger") as mock_logger:
            await test.run()

            judge_prompt_calls = [
                c for c in mock_logger.info.call_args_list
                if c.args and "agent.prompt_applied" in str(c.args[0])
                and c.kwargs.get("role") == "judge"
            ]
            judge_names = [c.kwargs.get("judge_name") for c in judge_prompt_calls]
            assert "accuracy" in judge_names, "Should log prompt for accuracy judge"
            assert "completeness" in judge_names, "Should log prompt for completeness judge"


class TestPromptHashConsistency:
    """Tests that prompt hashes are computed correctly."""

    def test_prompt_hash_matches_sha256_prefix(self):
        """The prompt hash logged should match SHA256 prefix of the prompt text."""
        from dokumen.test_object import _prompt_hash

        prompt = "You are a documentation validator."
        expected = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        assert _prompt_hash(prompt) == expected

    def test_prompt_hash_empty_string(self):
        """Empty prompt produces a valid hash."""
        from dokumen.test_object import _prompt_hash

        result = _prompt_hash("")
        assert len(result) == 12
        assert result == hashlib.sha256(b"").hexdigest()[:12]

    def test_prompt_hash_none_returns_none(self):
        """None prompt returns 'none' string."""
        from dokumen.test_object import _prompt_hash

        result = _prompt_hash(None)
        assert result == "none"
