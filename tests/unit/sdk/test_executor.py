"""Tests for dokumen.sdk.executor — ExecutorAgent."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from dokumen.sdk.executor import ExecutorAgent
from dokumen.sdk.query_runner import MockQueryRunner
from dokumen.sdk.testing import (
    make_assistant,
    make_executor_simple,
    make_executor_with_tools,
    make_init,
    make_result,
    make_tool_result,
)
from dokumen.sdk.types import SdkExecutorResult


class TestExecutorSimpleSuccess:
    async def test_executor_simple_success(self):
        """Simple execution with text response returns success."""
        messages = make_executor_simple("The documentation is correct.")
        runner = MockQueryRunner(messages)

        agent = ExecutorAgent(
            id="test-exec",
            system_prompt="You are a doc validator.",
            user_prompt="Check the docs.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await agent.run()

        assert isinstance(result, SdkExecutorResult)
        assert result.success is True
        assert result.final_response == "The documentation is correct."
        assert result.error is None
        assert len(runner.calls) == 1
        assert runner.calls[0].prompt == "Check the docs."


class TestExecutorWithToolCalls:
    async def test_executor_with_tool_calls(self):
        """Execution that uses tools reports tool calls correctly."""
        messages = make_executor_with_tools(
            tool_sequence=[
                ("read_file", {"path": "docs/api.md"}, "# API Reference\nGET /users"),
                ("glob", {"pattern": "docs/*.md"}, "docs/api.md\ndocs/guide.md"),
            ],
            final_text="Found 2 documentation files.",
        )
        runner = MockQueryRunner(messages)

        agent = ExecutorAgent(
            id="tool-exec",
            system_prompt="You validate docs.",
            user_prompt="Read all markdown files.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await agent.run()

        assert result.success is True
        assert result.final_response == "Found 2 documentation files."
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["tool_name"] == "read_file"
        assert result.tool_calls[1]["tool_name"] == "glob"


class TestExecutorErrorResult:
    async def test_executor_error_result(self):
        """SDK returns is_error=True -> success=False."""
        messages = [
            make_init(),
            make_assistant("Something went wrong"),
            make_result("Something went wrong", is_error=True),
        ]
        runner = MockQueryRunner(messages)

        agent = ExecutorAgent(
            id="err-exec",
            system_prompt="You validate docs.",
            user_prompt="Read the docs.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await agent.run()

        assert result.success is False
        assert result.error == "Something went wrong"


class TestExecutorTimeout:
    async def test_executor_timeout(self):
        """Agent times out -> returns error result with timeout message."""

        class SlowRunner:
            def __init__(self):
                self.calls = []

            async def run(self, prompt, options=None):
                self.calls.append(prompt)
                await asyncio.sleep(10)
                # Should never reach here
                yield make_init()  # pragma: no cover

        runner = SlowRunner()
        agent = ExecutorAgent(
            id="timeout-exec",
            system_prompt="Slow agent.",
            user_prompt="Do something slow.",
            sdk_tools=[],
            query_runner=runner,
            timeout=0.01,
        )
        result = await agent.run()

        assert result.success is False
        assert "timed out" in result.error
        assert result.duration_ms == 10  # int(0.01 * 1000)
        assert result.tool_calls == []
        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestExecutorNoResultMessage:
    async def test_executor_no_result_message(self):
        """Stream has no ResultMessage -> success=False."""
        # Only init + assistant, no ResultMessage
        messages = [make_init(), make_assistant("partial response")]
        runner = MockQueryRunner(messages)

        agent = ExecutorAgent(
            id="no-result-exec",
            system_prompt="Incomplete agent.",
            user_prompt="Try something.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await agent.run()

        # qr.result is None -> is_error=True -> success=False
        assert result.success is False
        assert result.final_response == ""
        assert result.duration_ms == 0


class TestExecutorExtractsUsage:
    async def test_executor_extracts_usage(self):
        """Usage tokens extracted from ResultMessage."""
        messages = [
            make_init(),
            make_assistant("done"),
            make_result("done", usage={"input_tokens": 500, "output_tokens": 150}),
        ]
        runner = MockQueryRunner(messages)

        agent = ExecutorAgent(
            id="usage-exec",
            system_prompt="Agent.",
            user_prompt="Work.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await agent.run()

        assert result.input_tokens == 500
        assert result.output_tokens == 150


class TestExecutorConversationLog:
    async def test_executor_conversation_log(self):
        """Conversation log built from all messages collected by _collect.

        _collect captures both AssistantMessage and UserMessage (tool results),
        so the conversation log contains assistant, tool, and user entries.
        """
        messages = [
            make_init(),
            make_assistant(
                "Let me read the file.",
                tool_calls=[{"id": "tc_1", "name": "read_file", "input": {"path": "a.md"}}],
            ),
            make_tool_result("tc_1", "file contents"),
            make_assistant("The file says hello."),
            make_result("The file says hello."),
        ]
        runner = MockQueryRunner(messages)

        agent = ExecutorAgent(
            id="log-exec",
            system_prompt="Agent.",
            user_prompt="Read the file.",
            sdk_tools=[],
            query_runner=runner,
        )
        result = await agent.run()

        assert result.success is True
        # 2 assistant + 1 tool result
        assert len(result.conversation_log) == 3
        # First log entry should be assistant with tool call
        assert result.conversation_log[0]["role"] == "assistant"
        assert "tool_calls" in result.conversation_log[0]
        assert result.conversation_log[0]["tool_calls"][0]["name"] == "read_file"
        # Second entry is the tool result
        assert result.conversation_log[1]["role"] == "tool"
        assert result.conversation_log[1]["tool_results"][0]["tool_use_id"] == "tc_1"
        # Third entry is the final assistant response
        assert result.conversation_log[2]["role"] == "assistant"
        assert result.conversation_log[2]["content"] == "The file says hello."
