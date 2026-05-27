"""Tests for dokumen.sdk.messages — message stream processing utilities."""

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from dokumen.sdk.messages import (
    build_conversation_log,
    build_judge_context,
    extract_tool_calls,
    extract_usage,
)
from dokumen.sdk.testing import (
    make_assistant,
    make_result,
    make_tool_result,
)
from dokumen.sdk.types import SdkExecutorResult


# ── extract_tool_calls ──────────────────────────────────────────────


class TestExtractToolCalls:
    def test_extract_tool_calls_empty(self):
        """Empty list returns empty."""
        assert extract_tool_calls([]) == []

    def test_extract_tool_calls_no_tools(self):
        """Text-only messages return empty."""
        msgs = [make_assistant("Hello world")]
        assert extract_tool_calls(msgs) == []

    def test_extract_tool_calls_single(self):
        """One tool call with result."""
        msgs = [
            make_assistant(
                "",
                tool_calls=[{"id": "tc_1", "name": "read_file", "input": {"path": "a.md"}}],
            ),
            make_tool_result("tc_1", "file contents"),
        ]
        result = extract_tool_calls(msgs)

        assert len(result) == 1
        assert result[0]["tool_name"] == "read_file"
        assert result[0]["tool_input"] == {"path": "a.md"}
        assert result[0]["tool_result"] == "file contents"
        assert result[0]["tool_use_id"] == "tc_1"

    def test_extract_tool_calls_multiple(self):
        """Multiple sequential tool calls."""
        msgs = [
            make_assistant(
                "",
                tool_calls=[{"id": "tc_1", "name": "read_file", "input": {"path": "a.md"}}],
            ),
            make_tool_result("tc_1", "contents of a"),
            make_assistant(
                "",
                tool_calls=[{"id": "tc_2", "name": "glob", "input": {"pattern": "*.md"}}],
            ),
            make_tool_result("tc_2", "a.md\nb.md"),
        ]
        result = extract_tool_calls(msgs)

        assert len(result) == 2
        assert result[0]["tool_name"] == "read_file"
        assert result[1]["tool_name"] == "glob"

    def test_extract_tool_calls_pending(self):
        """Tool call without result still returned."""
        msgs = [
            make_assistant(
                "",
                tool_calls=[{"id": "tc_1", "name": "read_file", "input": {"path": "a.md"}}],
            ),
            # No tool result follows
        ]
        result = extract_tool_calls(msgs)

        assert len(result) == 1
        assert result[0]["tool_name"] == "read_file"
        assert result[0]["tool_result"] is None


# ── build_conversation_log ──────────────────────────────────────────


class TestBuildConversationLog:
    def test_build_conversation_log_text_only(self):
        """Text message produces role:assistant entry."""
        msgs = [make_assistant("Hello world")]
        log = build_conversation_log(msgs)

        assert len(log) == 1
        assert log[0]["role"] == "assistant"
        assert log[0]["content"] == "Hello world"

    def test_build_conversation_log_with_tools(self):
        """Tool calls appear in log."""
        msgs = [
            make_assistant(
                "Let me read that file.",
                tool_calls=[{"id": "tc_1", "name": "read_file", "input": {"path": "a.md"}}],
            ),
        ]
        log = build_conversation_log(msgs)

        assert len(log) == 1
        assert log[0]["role"] == "assistant"
        assert log[0]["content"] == "Let me read that file."
        assert len(log[0]["tool_calls"]) == 1
        assert log[0]["tool_calls"][0]["name"] == "read_file"
        assert log[0]["tool_calls"][0]["id"] == "tc_1"

    def test_build_conversation_log_tool_results(self):
        """Tool results appear as role:tool."""
        msgs = [
            make_assistant(
                "",
                tool_calls=[{"id": "tc_1", "name": "read_file", "input": {"path": "a.md"}}],
            ),
            make_tool_result("tc_1", "file contents here"),
        ]
        log = build_conversation_log(msgs)

        # First entry: assistant with tool call (no text content)
        assert log[0]["role"] == "assistant"
        assert "content" not in log[0]
        assert log[0]["tool_calls"][0]["name"] == "read_file"

        # Second entry: tool result
        assert log[1]["role"] == "tool"
        assert log[1]["tool_results"][0]["tool_use_id"] == "tc_1"
        assert log[1]["tool_results"][0]["content"] == "file contents here"

    def test_build_conversation_log_user_text(self):
        """String user message produces role:user entry."""
        msgs = [UserMessage(content="What does the doc say?")]
        log = build_conversation_log(msgs)

        assert len(log) == 1
        assert log[0]["role"] == "user"
        assert log[0]["content"] == "What does the doc say?"


# ── extract_usage ────────────────────────────────────────────────────


class TestExtractUsage:
    def test_extract_usage_valid(self):
        """Extracts tokens correctly from a ResultMessage."""
        result = make_result(
            "done", usage={"input_tokens": 200, "output_tokens": 80}
        )
        usage = extract_usage(result)

        assert usage["input_tokens"] == 200
        assert usage["output_tokens"] == 80

    def test_extract_usage_none(self):
        """None returns zeros."""
        usage = extract_usage(None)

        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_extract_usage_no_usage(self):
        """ResultMessage with no usage returns zeros."""
        result = ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="test",
            stop_reason="end_turn",
            usage=None,
        )
        usage = extract_usage(result)

        assert usage == {"input_tokens": 0, "output_tokens": 0}


# ── build_judge_context ──────────────────────────────────────────────


class TestBuildJudgeContext:
    def _make_executor_result(self, final_response: str = "The answer is 42.") -> SdkExecutorResult:
        return SdkExecutorResult(
            success=True,
            final_response=final_response,
            tool_calls=[],
            input_tokens=100,
            output_tokens=50,
            conversation_log=[],
            duration_ms=1000,
        )

    def test_build_judge_context_with_output(self):
        """Includes executor output when include_output=True."""
        executor = self._make_executor_result("The answer is 42.")
        context = build_judge_context(executor, "Is the answer correct?", include_output=True)

        assert "## Executor Output" in context
        assert "The answer is 42." in context
        assert "## Evaluation Criteria" in context
        assert "Is the answer correct?" in context

    def test_build_judge_context_without_output(self):
        """Only criteria when include_output=False."""
        executor = self._make_executor_result("The answer is 42.")
        context = build_judge_context(executor, "Is the answer correct?", include_output=False)

        assert "## Executor Output" not in context
        assert "The answer is 42." not in context
        assert "## Evaluation Criteria" in context
        assert "Is the answer correct?" in context

    def test_build_judge_context_truncates_large_output(self):
        """Large executor output is truncated to max_response_chars."""
        large_response = "A" * 50_000
        executor = self._make_executor_result(large_response)
        context = build_judge_context(
            executor, "Evaluate.", include_output=True, max_response_chars=10_000
        )

        assert "## Executor Output" in context
        # Full response should NOT be present
        assert large_response not in context
        # Truncated content should be present
        assert "A" * 10_000 in context
        assert "[response truncated" in context.lower()

    def test_build_judge_context_no_truncation_when_under_limit(self):
        """Short responses are not truncated even when max_response_chars is set."""
        short_response = "The answer is 42."
        executor = self._make_executor_result(short_response)
        context = build_judge_context(
            executor, "Evaluate.", include_output=True, max_response_chars=10_000
        )

        assert short_response in context
        assert "truncated" not in context.lower()

    def test_build_judge_context_no_truncation_when_zero(self):
        """max_response_chars=0 means unlimited (no truncation)."""
        large_response = "B" * 100_000
        executor = self._make_executor_result(large_response)
        context = build_judge_context(
            executor, "Evaluate.", include_output=True, max_response_chars=0
        )

        assert large_response in context
        assert "truncated" not in context.lower()

    def test_build_judge_context_default_no_truncation(self):
        """Default behavior (no max_response_chars) does not truncate."""
        large_response = "C" * 100_000
        executor = self._make_executor_result(large_response)
        context = build_judge_context(executor, "Evaluate.", include_output=True)

        assert large_response in context

    def test_build_judge_context_includes_tool_calls(self):
        """Tool calls from executor appear in judge context."""
        executor = SdkExecutorResult(
            success=True,
            final_response="Found the docs.",
            tool_calls=[
                {"tool_name": "read_file", "tool_input": {"path": "docs/api.md"}, "tool_result": "# API Reference"},
                {"tool_name": "glob", "tool_input": {"pattern": "*.md"}, "tool_result": "a.md\nb.md"},
            ],
        )
        context = build_judge_context(executor, "Evaluate.", include_output=True)

        assert "## Executor Tool Calls" in context
        assert "read_file" in context
        assert "glob" in context
        assert "# API Reference" in context

    def test_build_judge_context_no_tool_calls_section_when_empty(self):
        """No tool calls section when executor has no tool calls."""
        executor = self._make_executor_result("The answer.")
        context = build_judge_context(executor, "Evaluate.", include_output=True)

        assert "## Executor Tool Calls" not in context

    def test_build_judge_context_tool_calls_disabled(self):
        """include_tool_calls=False suppresses tool calls section."""
        executor = SdkExecutorResult(
            success=True,
            final_response="Found it.",
            tool_calls=[
                {"tool_name": "read_file", "tool_input": {"path": "a.md"}, "tool_result": "contents"},
            ],
        )
        context = build_judge_context(
            executor, "Evaluate.", include_output=True, include_tool_calls=False
        )

        assert "## Executor Tool Calls" not in context

    def test_build_judge_context_truncates_large_tool_results(self):
        """Tool results are truncated at 500 chars."""
        large_result = "X" * 1000
        executor = SdkExecutorResult(
            success=True,
            final_response="Done.",
            tool_calls=[
                {"tool_name": "read_file", "tool_input": {"path": "big.md"}, "tool_result": large_result},
            ],
        )
        context = build_judge_context(executor, "Evaluate.", include_output=True)

        assert "## Executor Tool Calls" in context
        # Full 1000-char result should NOT appear
        assert large_result not in context

    def test_build_judge_context_includes_executor_prompts(self):
        """Executor system/user prompts appear in judge context."""
        executor = self._make_executor_result("The answer.")
        context = build_judge_context(
            executor, "Evaluate.", include_output=True,
            executor_system_prompt="You are a doc validator.",
            executor_user_prompt="Read the API docs and verify endpoints.",
        )

        assert "## Executor Task" in context
        assert "You are a doc validator." in context
        assert "Read the API docs and verify endpoints." in context

    def test_build_judge_context_no_task_section_when_empty(self):
        """No Executor Task section when prompts are empty."""
        executor = self._make_executor_result("The answer.")
        context = build_judge_context(
            executor, "Evaluate.", include_output=True,
            executor_system_prompt="",
            executor_user_prompt="",
        )

        assert "## Executor Task" not in context

    def test_build_judge_context_truncates_long_prompts(self):
        """Long executor prompts are truncated."""
        long_sys = "S" * 1000
        long_usr = "U" * 2000
        executor = self._make_executor_result("The answer.")
        context = build_judge_context(
            executor, "Evaluate.", include_output=True,
            executor_system_prompt=long_sys,
            executor_user_prompt=long_usr,
        )

        assert "## Executor Task" in context
        # System prompt truncated at 500
        assert "S" * 500 in context
        assert "S" * 501 not in context
        # User prompt truncated at 1000
        assert "U" * 1000 in context
        assert "U" * 1001 not in context

    def test_build_judge_context_skips_empty_criteria(self):
        """No Evaluation Criteria section when judge_prompt is empty."""
        executor = self._make_executor_result("The answer.")
        context = build_judge_context(executor, "", include_output=True)

        assert "## Evaluation Criteria" not in context

    def test_build_judge_context_section_order(self):
        """Sections appear in correct order: Task, Output, Tool Calls, Criteria."""
        executor = SdkExecutorResult(
            success=True,
            final_response="Found it.",
            tool_calls=[
                {"tool_name": "read_file", "tool_input": {"path": "a.md"}, "tool_result": "contents"},
            ],
        )
        context = build_judge_context(
            executor, "Is it correct?", include_output=True,
            executor_system_prompt="Validate docs.",
            executor_user_prompt="Read the file.",
        )

        task_pos = context.index("## Executor Task")
        output_pos = context.index("## Executor Output")
        tools_pos = context.index("## Executor Tool Calls")
        criteria_pos = context.index("## Evaluation Criteria")

        assert task_pos < output_pos < tools_pos < criteria_pos
