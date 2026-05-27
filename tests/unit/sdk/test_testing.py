"""Tests for dokumen.sdk.testing — message factory functions."""

import json

import pytest

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

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


class TestMakeInit:
    def test_default_session_id(self):
        """make_init() uses 'test-session' as default."""
        msg = make_init()
        assert isinstance(msg, SystemMessage)
        assert msg.subtype == "init"
        assert msg.data["session_id"] == "test-session"

    def test_custom_session_id(self):
        """make_init() accepts custom session_id."""
        msg = make_init(session_id="custom-123")
        assert msg.data["session_id"] == "custom-123"


class TestMakeAssistant:
    def test_text_only(self):
        """make_assistant() with text only creates TextBlock."""
        msg = make_assistant("hello world")
        assert isinstance(msg, AssistantMessage)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextBlock)
        assert msg.content[0].text == "hello world"

    def test_with_tool_calls(self):
        """make_assistant() with tool_calls creates ToolUseBlocks."""
        msg = make_assistant(
            "calling tool",
            tool_calls=[
                {"id": "tc_1", "name": "read_file", "input": {"path": "a.md"}},
            ],
        )
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextBlock)
        assert isinstance(msg.content[1], ToolUseBlock)
        assert msg.content[1].name == "read_file"
        assert msg.content[1].input == {"path": "a.md"}

    def test_empty_text_no_text_block(self):
        """make_assistant() with empty text skips TextBlock."""
        msg = make_assistant(
            "",
            tool_calls=[{"id": "tc_1", "name": "glob", "input": {}}],
        )
        # Empty text means no TextBlock, only ToolUseBlock
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolUseBlock)

    def test_multiple_tool_calls(self):
        """make_assistant() with multiple tool calls creates multiple ToolUseBlocks."""
        msg = make_assistant(
            "",
            tool_calls=[
                {"id": "tc_1", "name": "read_file", "input": {}},
                {"id": "tc_2", "name": "glob", "input": {"pattern": "*.md"}},
            ],
        )
        assert len(msg.content) == 2
        assert msg.content[0].name == "read_file"
        assert msg.content[1].name == "glob"

    def test_model_is_set(self):
        """make_assistant() sets model to claude-sonnet-4-6."""
        msg = make_assistant("test")
        assert msg.model == "claude-sonnet-4-6"


class TestMakeToolResult:
    def test_basic_tool_result(self):
        """make_tool_result() creates UserMessage with ToolResultBlock."""
        msg = make_tool_result("tc_1", "file contents")
        assert isinstance(msg, UserMessage)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolResultBlock)
        assert msg.content[0].tool_use_id == "tc_1"
        assert msg.content[0].content == "file contents"

    def test_error_tool_result(self):
        """make_tool_result() with is_error=True."""
        msg = make_tool_result("tc_1", "permission denied", is_error=True)
        assert msg.content[0].is_error is True


class TestMakeResult:
    def test_success_result(self):
        """make_result() creates success ResultMessage."""
        msg = make_result("final answer")
        assert isinstance(msg, ResultMessage)
        assert msg.result == "final answer"
        assert msg.is_error is False
        assert msg.subtype == "success"
        assert msg.session_id == "test-session"
        assert msg.duration_ms == 1000

    def test_error_result(self):
        """make_result() with is_error=True."""
        msg = make_result("error text", is_error=True)
        assert msg.is_error is True
        assert msg.subtype == "error"
        assert msg.stop_reason == "error"

    def test_custom_usage(self):
        """make_result() with custom usage dict."""
        msg = make_result("ok", usage={"input_tokens": 300, "output_tokens": 100})
        assert msg.usage["input_tokens"] == 300
        assert msg.usage["output_tokens"] == 100

    def test_custom_session_id(self):
        """make_result() with custom session_id."""
        msg = make_result("ok", session_id="custom-sess")
        assert msg.session_id == "custom-sess"

    def test_custom_duration(self):
        """make_result() with custom duration_ms."""
        msg = make_result("ok", duration_ms=5000)
        assert msg.duration_ms == 5000
        assert msg.duration_api_ms == 4000  # 0.8 * 5000


class TestMakeJudgePass:
    def test_default_pass(self):
        """make_judge_pass() creates a 3-message PASS sequence."""
        msgs = make_judge_pass()
        assert len(msgs) == 3
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], AssistantMessage)
        assert isinstance(msgs[2], ResultMessage)

        # Parse the verdict from the result
        verdict = json.loads(msgs[2].result)
        assert verdict["verdict"] == "PASS"
        assert verdict["confidence"] == 0.95
        assert verdict["reason"] == "Correct"

    def test_custom_pass(self):
        """make_judge_pass() with custom confidence and reason."""
        msgs = make_judge_pass(confidence=0.8, reason="Mostly right")
        verdict = json.loads(msgs[2].result)
        assert verdict["confidence"] == 0.8
        assert verdict["reason"] == "Mostly right"


class TestMakeJudgeFail:
    def test_default_fail(self):
        """make_judge_fail() creates a 3-message FAIL sequence."""
        msgs = make_judge_fail()
        assert len(msgs) == 3
        verdict = json.loads(msgs[2].result)
        assert verdict["verdict"] == "FAIL"
        assert verdict["reason"] == "Incorrect"
        assert verdict["confidence"] == 0.8

    def test_custom_fail(self):
        """make_judge_fail() with custom reason and confidence."""
        msgs = make_judge_fail(reason="Missing details", confidence=0.6)
        verdict = json.loads(msgs[2].result)
        assert verdict["reason"] == "Missing details"
        assert verdict["confidence"] == 0.6


class TestMakeExecutorSimple:
    def test_simple_executor(self):
        """make_executor_simple() creates 3-message sequence."""
        msgs = make_executor_simple("The answer is 42.")
        assert len(msgs) == 3
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], AssistantMessage)
        assert isinstance(msgs[2], ResultMessage)
        assert msgs[2].result == "The answer is 42."


class TestMakeExecutorWithTools:
    def test_single_tool(self):
        """make_executor_with_tools() with single tool call."""
        msgs = make_executor_with_tools(
            tool_sequence=[("read_file", {"path": "a.md"}, "contents")],
            final_text="Done.",
        )
        # init + tool_call_assistant + tool_result + final_assistant + result
        assert len(msgs) == 5
        assert isinstance(msgs[0], SystemMessage)  # init
        assert isinstance(msgs[1], AssistantMessage)  # tool call
        assert isinstance(msgs[2], UserMessage)  # tool result
        assert isinstance(msgs[3], AssistantMessage)  # final
        assert isinstance(msgs[4], ResultMessage)  # result

    def test_multiple_tools(self):
        """make_executor_with_tools() with multiple tool calls."""
        msgs = make_executor_with_tools(
            tool_sequence=[
                ("read_file", {"path": "a.md"}, "content a"),
                ("glob", {"pattern": "*.md"}, "a.md\nb.md"),
            ],
            final_text="Found files.",
        )
        # init + (tool_call + tool_result) * 2 + final + result = 7
        assert len(msgs) == 7

    def test_empty_tools(self):
        """make_executor_with_tools() with no tools is like simple."""
        msgs = make_executor_with_tools(tool_sequence=[], final_text="Done.")
        # init + final_assistant + result = 3
        assert len(msgs) == 3
