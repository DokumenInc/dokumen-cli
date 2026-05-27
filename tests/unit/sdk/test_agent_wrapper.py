"""Tests for dokumen.sdk.agent_wrapper — SdkExecutorWrapper and SdkJudgeWrapper."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from dokumen.sdk.types import ExecutorResult, JudgeVerdict
from dokumen.agent_object import AgentType, ExecutorOutput, JudgeResult
from dokumen.sdk.agent_wrapper import SdkExecutorWrapper, SdkJudgeWrapper
from dokumen.sdk.executor import ExecutorAgent
from dokumen.sdk.judge import JudgeAgent
from dokumen.sdk.query_runner import MockQueryRunner
from dokumen.sdk.testing import (
    make_assistant,
    make_executor_simple,
    make_executor_with_tools,
    make_init,
    make_judge_fail,
    make_judge_pass,
    make_result,
    make_tool_result,
)


def _make_executor_output(**overrides) -> ExecutorResult:
    """Build an ExecutorResult with sane defaults."""
    defaults = dict(
        tool_calls=[],
        final_response="The docs are correct.",
        success=True,
        error=None,
        system_prompt="sys",
        user_prompt="usr",
        original_user_prompt="usr",
        input_tokens=100,
        output_tokens=50,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        conversation_log=[],
    )
    defaults.update(overrides)
    return ExecutorResult(**defaults)


# ── SdkExecutorWrapper tests ─────────────────────────────────────────


class TestSdkExecutorWrapperInit:
    def test_wrapper_exposes_correct_attributes(self):
        """Wrapper exposes id, agent_type, system_prompt, user_prompt, tools."""
        runner = MockQueryRunner(make_executor_simple("ok"))
        executor = ExecutorAgent(
            id="test-exec",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkExecutorWrapper(executor, system_prompt="sys", user_prompt="usr")

        assert wrapper.id == "test-exec"
        assert wrapper.agent_type == AgentType.EXECUTOR
        assert wrapper.system_prompt == "sys"
        assert wrapper.user_prompt == "usr"
        assert wrapper.tools == []


class TestSdkExecutorWrapperRun:
    async def test_run_returns_executor_result(self):
        """Wrapper.run() returns ExecutorResult."""
        runner = MockQueryRunner(make_executor_simple("The answer is 42."))
        executor = ExecutorAgent(
            id="test-exec",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkExecutorWrapper(executor, system_prompt="sys", user_prompt="usr")

        result = await wrapper.run()

        assert isinstance(result, ExecutorResult)
        # Also verify backward compat alias
        assert isinstance(result, ExecutorOutput)
        assert result.success is True
        assert result.final_response == "The answer is 42."
        assert result.system_prompt == "sys"
        assert result.user_prompt == "usr"

    async def test_run_passes_original_user_prompt(self):
        """original_user_prompt is passed through to ExecutorResult."""
        runner = MockQueryRunner(make_executor_simple("done"))
        executor = ExecutorAgent(
            id="test-exec",
            system_prompt="sys",
            user_prompt="expanded prompt",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkExecutorWrapper(
            executor, system_prompt="sys", user_prompt="expanded prompt"
        )

        result = await wrapper.run(original_user_prompt="original prompt")

        assert result.original_user_prompt == "original prompt"

    async def test_run_defaults_original_to_user_prompt(self):
        """original_user_prompt defaults to user_prompt when empty."""
        runner = MockQueryRunner(make_executor_simple("done"))
        executor = ExecutorAgent(
            id="test-exec",
            system_prompt="sys",
            user_prompt="the prompt",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkExecutorWrapper(
            executor, system_prompt="sys", user_prompt="the prompt"
        )

        result = await wrapper.run(original_user_prompt="")

        assert result.original_user_prompt == "the prompt"

    async def test_run_overrides_timeout(self):
        """Timeout kwarg overrides executor timeout."""
        runner = MockQueryRunner(make_executor_simple("done"))
        executor = ExecutorAgent(
            id="test-exec",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
            timeout=60.0,
        )
        wrapper = SdkExecutorWrapper(executor, system_prompt="sys", user_prompt="usr")

        assert executor.timeout == 60.0
        await wrapper.run(timeout=30.0)
        assert executor.timeout == 30.0

    async def test_run_preserves_timeout_when_none(self):
        """Timeout is not changed when kwarg is None."""
        runner = MockQueryRunner(make_executor_simple("done"))
        executor = ExecutorAgent(
            id="test-exec",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
            timeout=60.0,
        )
        wrapper = SdkExecutorWrapper(executor, system_prompt="sys", user_prompt="usr")

        await wrapper.run(timeout=None)
        assert executor.timeout == 60.0

    async def test_run_with_tool_calls(self):
        """Wrapper correctly returns tool call data from SDK."""
        messages = make_executor_with_tools(
            tool_sequence=[
                ("read_file", {"path": "docs/api.md"}, "# API"),
            ],
            final_text="Found the API docs.",
        )
        runner = MockQueryRunner(messages)
        executor = ExecutorAgent(
            id="tool-exec",
            system_prompt="sys",
            user_prompt="Read the API docs.",
            sdk_tools=["Read"],
            query_runner=runner,
        )
        wrapper = SdkExecutorWrapper(executor, system_prompt="sys", user_prompt="Read the API docs.")

        result = await wrapper.run()

        assert isinstance(result, ExecutorResult)
        assert result.success is True
        assert result.final_response == "Found the API docs."

    async def test_run_error_result(self):
        """Wrapper returns error ExecutorResult when SDK reports error."""
        messages = [
            make_init(),
            make_assistant("Something went wrong"),
            make_result("Something went wrong", is_error=True),
        ]
        runner = MockQueryRunner(messages)
        executor = ExecutorAgent(
            id="err-exec",
            system_prompt="sys",
            user_prompt="do something",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkExecutorWrapper(executor, system_prompt="sys", user_prompt="do something")

        result = await wrapper.run()

        assert isinstance(result, ExecutorResult)
        assert result.success is False
        assert result.error is not None

    async def test_run_ignores_extra_kwargs(self):
        """Wrapper.run() accepts and ignores on_tool_call, on_conversation_message."""
        runner = MockQueryRunner(make_executor_simple("ok"))
        executor = ExecutorAgent(
            id="test-exec",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkExecutorWrapper(executor, system_prompt="sys", user_prompt="usr")

        # Should not raise even with extra kwargs
        result = await wrapper.run(
            on_tool_call=lambda *a: None,
            on_conversation_message=lambda *a: None,
        )
        assert isinstance(result, ExecutorResult)


# ── SdkJudgeWrapper tests ────────────────────────────────────────────


class TestSdkJudgeWrapperInit:
    def test_wrapper_exposes_correct_attributes(self):
        """Wrapper exposes id, agent_type, system_prompt, tools, include_executor_output."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="accuracy",
            system_prompt="Evaluate.",
            user_prompt="Is it correct?",
            include_executor_output=True,
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(
            judge, assertion_text="Must be accurate", system_prompt="Evaluate."
        )

        assert wrapper.id == "accuracy"
        assert wrapper.agent_type == AgentType.JUDGE
        assert wrapper.system_prompt == "Evaluate."
        assert wrapper.tools == []
        assert wrapper.include_executor_output is True


class TestSdkJudgeWrapperRun:
    async def test_run_returns_judge_verdict_pass(self):
        """Wrapper.run() returns JudgeVerdict with PASS."""
        runner = MockQueryRunner(make_judge_pass(confidence=0.95, reason="All correct"))
        judge = JudgeAgent(
            id="accuracy",
            system_prompt="Evaluate.",
            user_prompt="Is it correct?",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(
            judge, assertion_text="accuracy", system_prompt="Evaluate."
        )

        executor_output = _make_executor_output()
        result = await wrapper.run(executor_output=executor_output)

        assert isinstance(result, JudgeVerdict)
        # Also verify backward compat alias
        assert isinstance(result, JudgeResult)
        assert result.passed is True
        assert result.confidence == 0.95
        assert result.failure_reason is None
        assert result.judge_id == "accuracy"
        assert result.assertion_text == "accuracy"

    async def test_run_returns_judge_verdict_fail(self):
        """Wrapper.run() returns JudgeVerdict with FAIL."""
        runner = MockQueryRunner(make_judge_fail(reason="Missing info", confidence=0.7))
        judge = JudgeAgent(
            id="completeness",
            system_prompt="Evaluate completeness.",
            user_prompt="Is it complete?",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(
            judge, assertion_text="completeness", system_prompt="Evaluate completeness."
        )

        result = await wrapper.run(executor_output=_make_executor_output())

        assert isinstance(result, JudgeVerdict)
        assert result.passed is False
        assert result.failure_reason == "Missing info"
        assert result.confidence == 0.7

    async def test_run_passes_executor_result_directly(self):
        """Wrapper passes ExecutorResult directly to judge (no conversion needed)."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="accuracy",
            system_prompt="Evaluate.",
            user_prompt="Check.",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Evaluate.")

        executor_output = _make_executor_output(
            final_response="The answer is 42.",
            success=True,
            input_tokens=200,
            output_tokens=75,
        )

        result = await wrapper.run(executor_output=executor_output)

        # The judge should receive the executor's final response in its context
        sent_prompt = runner.calls[0].prompt
        assert "The answer is 42." in sent_prompt
        assert result.passed is True

    async def test_run_with_none_executor_output(self):
        """Wrapper handles None executor_output gracefully."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="null-judge",
            system_prompt="Evaluate.",
            user_prompt="Check.",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Evaluate.")

        result = await wrapper.run(executor_output=None)

        assert isinstance(result, JudgeVerdict)
        assert result.passed is True

    async def test_run_overrides_timeout(self):
        """Timeout kwarg overrides judge timeout."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="timeout-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            sdk_tools=[],
            query_runner=runner,
            timeout=120.0,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Judge.")

        assert judge.timeout == 120.0
        await wrapper.run(executor_output=_make_executor_output(), timeout=30.0)
        assert judge.timeout == 30.0

    async def test_run_preserves_timeout_when_none(self):
        """Timeout is not changed when kwarg is None."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="timeout-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            sdk_tools=[],
            query_runner=runner,
            timeout=120.0,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Judge.")

        await wrapper.run(executor_output=_make_executor_output(), timeout=None)
        assert judge.timeout == 120.0

    async def test_run_ignores_extra_kwargs(self):
        """Wrapper.run() accepts and ignores on_tool_call, on_conversation_message."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="extra-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Judge.")

        result = await wrapper.run(
            executor_output=_make_executor_output(),
            on_tool_call=lambda *a: None,
            on_conversation_message=lambda *a: None,
        )
        assert isinstance(result, JudgeVerdict)

    async def test_run_passes_assertion_text(self):
        """assertion_text is passed through to JudgeVerdict."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="assert-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(
            judge, assertion_text="Must mention OAuth", system_prompt="Judge."
        )

        result = await wrapper.run(executor_output=_make_executor_output())

        assert result.assertion_text == "Must mention OAuth"


class TestSdkJudgeWrapperIncludeOutput:
    async def test_include_executor_output_true(self):
        """include_executor_output=True includes executor output in judge context."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="include-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            include_executor_output=True,
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Judge.")

        executor_output = _make_executor_output(final_response="Executor said hello.")
        await wrapper.run(executor_output=executor_output)

        sent_prompt = runner.calls[0].prompt
        assert "Executor said hello." in sent_prompt

    async def test_tool_calls_included_in_judge_context(self):
        """tool_calls from ExecutorResult appear in judge context."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="tc-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            include_executor_output=True,
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Judge.")

        executor_output = _make_executor_output(
            tool_calls=[
                {"tool_name": "read_file", "tool_input": {"path": "docs/api.md"}, "tool_result": "# API docs"},
                {"tool_name": "glob", "tool_input": {"pattern": "*.md"}, "tool_result": "a.md\nb.md"},
            ],
        )
        await wrapper.run(executor_output=executor_output)

        sent_prompt = runner.calls[0].prompt
        assert "read_file" in sent_prompt
        assert "glob" in sent_prompt

    async def test_include_executor_output_false(self):
        """include_executor_output=False omits executor output from judge context."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="no-include-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            include_executor_output=False,
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Judge.")

        executor_output = _make_executor_output(final_response="Should not appear.")
        await wrapper.run(executor_output=executor_output)

        sent_prompt = runner.calls[0].prompt
        assert "## Executor Output" not in sent_prompt


class TestSdkJudgeWrapperPromptForwarding:
    async def test_judge_wrapper_forwards_executor_prompts(self):
        """Executor system/user prompts appear in judge context."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="prompt-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            include_executor_output=True,
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Judge.")

        executor_output = _make_executor_output()
        await wrapper.run(
            executor_output=executor_output,
            executor_system_prompt="You are a doc validator.",
            executor_user_prompt="Read the API docs.",
        )

        sent_prompt = runner.calls[0].prompt
        assert "## Executor Task" in sent_prompt
        assert "You are a doc validator." in sent_prompt
        assert "Read the API docs." in sent_prompt

    async def test_judge_wrapper_defaults_executor_prompts_to_empty(self):
        """No Executor Task section when prompts not provided."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="no-prompt-judge",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            include_executor_output=True,
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(judge, system_prompt="Judge.")

        executor_output = _make_executor_output()
        await wrapper.run(executor_output=executor_output)

        sent_prompt = runner.calls[0].prompt
        assert "## Executor Task" not in sent_prompt


class TestSdkJudgeWrapperGetAssertionText:
    def test_get_assertion_text(self):
        """_get_assertion_text returns the assertion text."""
        runner = MockQueryRunner(make_judge_pass())
        judge = JudgeAgent(
            id="test",
            system_prompt="Judge.",
            user_prompt="Evaluate.",
            sdk_tools=[],
            query_runner=runner,
        )
        wrapper = SdkJudgeWrapper(
            judge, assertion_text="My assertion", system_prompt="Judge."
        )
        assert wrapper._get_assertion_text() == "My assertion"
