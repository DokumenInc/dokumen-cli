"""Tests for debug module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

from dokumen.debug import (
    set_debug,
    is_debug,
    debug,
    DebugSession,
    _serialize_messages,
    _serialize_response,
    _truncate_tool_calls_for_debug,
    _truncate_message_content,
    MAX_DEBUG_TOOL_RESULT_CHARS,
    MAX_DEBUG_MESSAGE_CONTENT_CHARS,
    start_debug_session,
    get_debug_session,
    end_debug_session,
)


@pytest.fixture(autouse=True)
def reset_debug_state():
    """Reset global debug state before each test."""
    import dokumen.debug as debug_module
    debug_module._debug_enabled = False
    debug_module._debug_session = None
    yield
    debug_module._debug_enabled = False
    debug_module._debug_session = None


class TestGlobalDebugState:
    """Tests for global debug state functions."""

    def test_set_debug_enables(self):
        """set_debug(True) should enable debug mode."""
        set_debug(True)
        assert is_debug() is True

    def test_set_debug_disables(self):
        """set_debug(False) should disable debug mode."""
        set_debug(True)
        set_debug(False)
        assert is_debug() is False

    def test_is_debug_default_false(self):
        """is_debug() should return False by default."""
        assert is_debug() is False

    def test_debug_prints_when_enabled(self, capsys):
        """debug() should print when enabled."""
        set_debug(True)
        debug("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_debug_silent_when_disabled(self, capsys):
        """debug() should not print when disabled."""
        set_debug(False)
        debug("test message")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestSerializeMessages:
    """Tests for _serialize_messages helper."""

    def test_serialize_dict_passthrough(self):
        """Dict messages should pass through unchanged."""
        messages = [{"role": "user", "content": "hello"}]
        result = _serialize_messages(messages)
        assert result == [{"role": "user", "content": "hello"}]

    def test_serialize_object_with_to_dict(self):
        """Objects with to_dict() should be serialized."""
        mock_msg = MagicMock()
        mock_msg.to_dict.return_value = {"role": "assistant", "content": "hi"}

        result = _serialize_messages([mock_msg])

        assert result == [{"role": "assistant", "content": "hi"}]
        mock_msg.to_dict.assert_called_once()

    def test_serialize_object_with_dict_attr(self):
        """Objects with __dict__ should be serialized."""
        class SimpleMessage:
            def __init__(self):
                self.role = "user"
                self.content = "test"

        msg = SimpleMessage()
        result = _serialize_messages([msg])

        assert result[0]["role"] == "user"
        assert result[0]["content"] == "test"

    def test_serialize_unsupported_to_string(self):
        """Unsupported types should be converted to string."""
        result = _serialize_messages(["plain string"])
        assert result == [{"content": "plain string"}]

    def test_serialize_empty_list(self):
        """Empty list should return empty list."""
        result = _serialize_messages([])
        assert result == []

    def test_serialize_mixed_types(self):
        """Should handle mixed message types."""
        mock_msg = MagicMock()
        mock_msg.to_dict.return_value = {"type": "mock"}

        messages = [
            {"role": "user", "content": "dict"},
            mock_msg,
            "string message"
        ]
        result = _serialize_messages(messages)

        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "dict"}
        assert result[1] == {"type": "mock"}
        assert result[2] == {"content": "string message"}


class TestSerializeResponse:
    """Tests for _serialize_response helper."""

    def test_serialize_none(self):
        """None should return None."""
        assert _serialize_response(None) is None

    def test_serialize_dict_passthrough(self):
        """Dict should pass through unchanged."""
        response = {"content": "hello", "tool_calls": []}
        result = _serialize_response(response)
        assert result == {"content": "hello", "tool_calls": []}

    def test_serialize_object_with_to_dict(self):
        """Objects with to_dict() should be serialized."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {"content": "response"}

        result = _serialize_response(mock_response)

        assert result == {"content": "response"}

    def test_serialize_object_with_dict_attr(self):
        """Objects with __dict__ should be serialized."""
        class SimpleResponse:
            def __init__(self):
                self.content = "test"
                self.status = "ok"

        response = SimpleResponse()
        result = _serialize_response(response)

        assert result["content"] == "test"
        assert result["status"] == "ok"

    def test_serialize_unsupported_to_string(self):
        """Unsupported types should be converted to string."""
        result = _serialize_response(12345)
        assert result == "12345"


class TestDebugSession:
    """Tests for DebugSession class."""

    def test_create_session(self):
        """Should create session with command."""
        session = DebugSession(command="run")
        assert session.command == "run"
        assert session.output_dir == ".dokumen-cache/debug-traces"
        assert session.tests == []
        assert session.analyzers == []

    def test_create_session_with_meta(self):
        """Should create session with metadata."""
        session = DebugSession(command="run", meta={"config": "test.yaml"})
        assert session.meta == {"config": "test.yaml"}

    def test_get_output_path(self):
        """Should generate unique output path."""
        session = DebugSession(command="run")
        path = session.get_output_path()

        assert path.parent == Path(".dokumen-cache/debug-traces")
        assert "run_" in path.name
        assert path.suffix == ".json"

    def test_start_test(self):
        """Should start tracking a test."""
        session = DebugSession(command="run")
        session.start_test("my-test")

        assert session._current_test is not None
        assert session._current_test["test_id"] == "my-test"
        assert "started_at" in session._current_test

    def test_start_executor(self):
        """Should start tracking executor for current test."""
        session = DebugSession(command="run")
        session.start_test("my-test")
        session.start_executor()

        assert session._current_executor is not None
        assert session._current_executor == session._current_test["executor"]

    def test_add_executor_iteration(self):
        """Should record executor iteration."""
        session = DebugSession(command="run")
        session.start_test("my-test")
        session.start_executor()

        session.add_executor_iteration(
            iteration=1,
            messages=[{"role": "user", "content": "test"}],
            response={"content": "response"},
            tool_calls=[{"name": "read_file"}]
        )

        iterations = session._current_test["executor"]["iterations"]
        assert len(iterations) == 1
        assert iterations[0]["iteration"] == 1
        assert iterations[0]["tool_calls"] == [{"name": "read_file"}]

    def test_finish_executor(self):
        """Should record executor output."""
        session = DebugSession(command="run")
        session.start_test("my-test")
        session.start_executor()

        session.finish_executor({"final_response": "done", "success": True})

        assert session._current_test["executor"]["output"]["final_response"] == "done"
        assert session._current_executor is None

    def test_start_judge(self):
        """Should start tracking a judge."""
        session = DebugSession(command="run")
        session.start_test("my-test")
        session.start_judge("accuracy")

        assert session._current_judge is not None
        assert session._current_judge["judge_id"] == "accuracy"

    def test_add_judge_iteration(self):
        """Should record judge iteration."""
        session = DebugSession(command="run")
        session.start_test("my-test")
        session.start_judge("accuracy")

        session.add_judge_iteration(
            iteration=1,
            messages=[{"role": "system", "content": "eval"}],
            response={"verdict": "PASS"},
            tool_calls=[]
        )

        iterations = session._current_judge["iterations"]
        assert len(iterations) == 1
        assert iterations[0]["iteration"] == 1

    def test_finish_judge(self):
        """Should record judge result and add to test."""
        session = DebugSession(command="run")
        session.start_test("my-test")
        session.start_judge("accuracy")

        session.finish_judge({"passed": True, "confidence": 0.95})

        assert len(session._current_test["judges"]) == 1
        assert session._current_test["judges"][0]["judge_id"] == "accuracy"
        assert session._current_test["judges"][0]["result"]["passed"] is True
        assert session._current_judge is None

    def test_finish_test(self):
        """Should complete test and add to tests list."""
        session = DebugSession(command="run")
        session.start_test("my-test")

        session.finish_test({"passed": True, "duration": 1.5})

        assert len(session.tests) == 1
        assert session.tests[0]["test_id"] == "my-test"
        assert session.tests[0]["result"]["passed"] is True
        assert "completed_at" in session.tests[0]
        assert session._current_test is None

    def test_flush_test_when_debug_disabled(self, tmp_path):
        """flush_test should do nothing when debug disabled."""
        session = DebugSession(command="run", output_dir=str(tmp_path))
        session.start_test("my-test")
        session.finish_test({"passed": True})

        # Debug is disabled by default
        session.flush_test("my-test")

        # No file should be created
        assert not list(tmp_path.glob("*.json"))

    def test_flush_test_writes_json(self, tmp_path):
        """flush_test should write incremental JSON when debug enabled."""
        set_debug(True)
        session = DebugSession(command="run", output_dir=str(tmp_path))
        session.start_test("my-test")
        session.finish_test({"passed": True})

        session.flush_test("my-test")

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

        with open(files[0]) as f:
            data = json.load(f)

        assert data["meta"]["command"] == "run"
        assert len(data["tests"]) == 1
        assert data["tests"][0]["test_id"] == "my-test"

    def test_flush_test_appends_to_existing(self, tmp_path):
        """flush_test should append to existing file."""
        set_debug(True)
        session = DebugSession(command="run", output_dir=str(tmp_path))

        # First test
        session.start_test("test-1")
        session.finish_test({"passed": True})
        session.flush_test("test-1")

        # Second test
        session.start_test("test-2")
        session.finish_test({"passed": False})
        session.flush_test("test-2")

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

        with open(files[0]) as f:
            data = json.load(f)

        assert len(data["tests"]) == 2
        assert data["tests"][0]["test_id"] == "test-1"
        assert data["tests"][1]["test_id"] == "test-2"

    def test_write_creates_file(self, tmp_path):
        """write() should create JSON file."""
        session = DebugSession(command="run", output_dir=str(tmp_path))
        session.start_test("my-test")
        session.finish_test({"passed": True})

        path = session.write()

        assert path.exists()
        with open(path) as f:
            data = json.load(f)

        assert data["meta"]["command"] == "run"
        assert len(data["tests"]) == 1

    def test_write_excludes_none_values(self, tmp_path):
        """write() should exclude None values from output."""
        session = DebugSession(command="run", output_dir=str(tmp_path))
        # No tests, no analyzers, no scaffold_generation

        path = session.write()

        with open(path) as f:
            data = json.load(f)

        assert "meta" in data
        assert "tests" not in data  # Empty list converted to None, excluded
        assert "analyzers" not in data
        assert "scaffold_generation" not in data


class TestAnalyzerTracking:
    """Tests for analyzer tracking methods."""

    def test_start_analyzer(self):
        """Should start tracking an analyzer."""
        session = DebugSession(command="analyze")
        session.start_analyzer("cli-problems")

        assert session._current_analyzer is not None
        assert session._current_analyzer["analyzer_name"] == "cli-problems"

    def test_add_analyzer_iteration(self):
        """Should record analyzer iteration."""
        session = DebugSession(command="analyze")
        session.start_analyzer("cli-problems")

        session.add_analyzer_iteration(
            iteration=1,
            messages=[{"role": "user", "content": "analyze"}],
            response={"content": "found issues"},
            tool_calls=[]
        )

        iterations = session._current_analyzer["iterations"]
        assert len(iterations) == 1

    def test_add_analyzer_problem(self):
        """Should record analyzer problem."""
        session = DebugSession(command="analyze")
        session.start_analyzer("cli-problems")

        session.add_analyzer_problem({"type": "bug", "description": "test fails"})

        assert len(session._current_analyzer["problems"]) == 1

    def test_finish_analyzer(self):
        """Should complete analyzer and add to list."""
        session = DebugSession(command="analyze")
        session.start_analyzer("cli-problems")

        session.finish_analyzer({"problems_found": 5})

        assert len(session.analyzers) == 1
        assert session.analyzers[0]["analyzer_name"] == "cli-problems"
        assert session._current_analyzer is None


class TestScaffoldGenerationTracking:
    """Tests for scaffold generation tracking methods."""

    def test_start_scaffold_generation(self):
        """Should start tracking scaffold generation."""
        session = DebugSession(command="add")
        session.start_scaffold_generation("docs/api.md", name="api-test")

        assert session.scaffold_generation is not None
        assert session.scaffold_generation["doc_path"] == "docs/api.md"
        assert session.scaffold_generation["name"] == "api-test"

    def test_add_scaffold_iteration(self):
        """Should record scaffold iteration."""
        session = DebugSession(command="add")
        session.start_scaffold_generation("docs/api.md")

        session.add_scaffold_iteration(
            iteration=1,
            messages=[{"role": "user", "content": "generate"}],
            response={"content": "yaml scaffold"},
            tool_calls=[]
        )

        iterations = session.scaffold_generation["iterations"]
        assert len(iterations) == 1

    def test_finish_scaffold_generation(self):
        """Should complete scaffold generation."""
        session = DebugSession(command="add")
        session.start_scaffold_generation("docs/api.md")

        session.finish_scaffold_generation({"scaffold_path": "tests/api.test.yaml"})

        assert session.scaffold_generation["result"]["scaffold_path"] == "tests/api.test.yaml"
        assert "completed_at" in session.scaffold_generation


class TestTruncateToolCallsForDebug:
    """Tests for _truncate_tool_calls_for_debug helper."""

    def test_short_results_unchanged(self):
        """Tool calls with short results should pass through unchanged."""
        tool_calls = [{"name": "read_file", "result": "short content"}]
        result = _truncate_tool_calls_for_debug(tool_calls)
        assert result[0]["result"] == "short content"

    def test_long_result_truncated(self):
        """Tool calls with long results should be truncated."""
        long_result = "x" * (MAX_DEBUG_TOOL_RESULT_CHARS + 1000)
        tool_calls = [{"name": "read_file", "result": long_result}]
        result = _truncate_tool_calls_for_debug(tool_calls)
        assert len(result[0]["result"]) < len(long_result)
        assert result[0]["result"].startswith("x" * 100)
        assert "truncated" in result[0]["result"]

    def test_truncation_includes_original_length(self):
        """Truncation message should include the original length."""
        original_len = MAX_DEBUG_TOOL_RESULT_CHARS + 5000
        long_result = "a" * original_len
        tool_calls = [{"name": "read_file", "result": long_result}]
        result = _truncate_tool_calls_for_debug(tool_calls)
        assert str(original_len) in result[0]["result"]

    def test_non_string_results_unchanged(self):
        """Non-string results should pass through unchanged."""
        tool_calls = [{"name": "list_dir", "result": ["file1.md", "file2.md"]}]
        result = _truncate_tool_calls_for_debug(tool_calls)
        assert result[0]["result"] == ["file1.md", "file2.md"]

    def test_missing_result_key_unchanged(self):
        """Tool calls without result key should pass through."""
        tool_calls = [{"name": "read_file", "args": {"path": "test.md"}}]
        result = _truncate_tool_calls_for_debug(tool_calls)
        assert result[0] == {"name": "read_file", "args": {"path": "test.md"}}

    def test_does_not_mutate_original(self):
        """Truncation should not modify the original tool_calls list."""
        long_result = "z" * (MAX_DEBUG_TOOL_RESULT_CHARS + 500)
        original = [{"name": "read_file", "result": long_result}]
        _truncate_tool_calls_for_debug(original)
        assert len(original[0]["result"]) == MAX_DEBUG_TOOL_RESULT_CHARS + 500

    def test_empty_list(self):
        """Empty tool call list should return empty list."""
        assert _truncate_tool_calls_for_debug([]) == []

    def test_multiple_tool_calls_mixed(self):
        """Should handle mix of short and long results."""
        tool_calls = [
            {"name": "read_file", "result": "short"},
            {"name": "search", "result": "y" * (MAX_DEBUG_TOOL_RESULT_CHARS + 100)},
            {"name": "glob", "result": "also short"},
        ]
        result = _truncate_tool_calls_for_debug(tool_calls)
        assert result[0]["result"] == "short"
        assert "truncated" in result[1]["result"]
        assert result[2]["result"] == "also short"

    def test_non_dict_tool_calls_passthrough(self):
        """Non-dict items in tool_calls should pass through."""
        tool_calls = ["string_item", 42]
        result = _truncate_tool_calls_for_debug(tool_calls)
        assert result == ["string_item", 42]


class TestTruncateMessageContent:
    """Tests for _truncate_message_content helper."""

    def test_short_content_unchanged(self):
        """Messages with short content should pass through unchanged."""
        messages = [{"role": "user", "content": "hello"}]
        result = _truncate_message_content(messages)
        assert result[0]["content"] == "hello"

    def test_long_content_truncated(self):
        """Messages with long content should be truncated."""
        long_content = "x" * (MAX_DEBUG_MESSAGE_CONTENT_CHARS + 1000)
        messages = [{"role": "user", "content": long_content}]
        result = _truncate_message_content(messages)
        assert len(result[0]["content"]) < len(long_content)
        assert "truncated" in result[0]["content"]

    def test_non_string_content_unchanged(self):
        """Messages with non-string content (e.g. list blocks) pass through."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = _truncate_message_content(messages)
        assert result[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_does_not_mutate_original(self):
        """Truncation should not modify the original messages list."""
        long_content = "z" * (MAX_DEBUG_MESSAGE_CONTENT_CHARS + 500)
        original = [{"role": "user", "content": long_content}]
        _truncate_message_content(original)
        assert len(original[0]["content"]) == MAX_DEBUG_MESSAGE_CONTENT_CHARS + 500

    def test_non_dict_messages_passthrough(self):
        """Non-dict messages should pass through."""
        messages = ["plain string"]
        result = _truncate_message_content(messages)
        assert result == ["plain string"]

    def test_missing_content_key_passthrough(self):
        """Messages without content key should pass through."""
        messages = [{"role": "system"}]
        result = _truncate_message_content(messages)
        assert result == [{"role": "system"}]


class TestDebugTraceIntegration:
    """Tests that debug trace truncation is applied in iteration recording."""

    def test_executor_iteration_truncates_tool_calls(self):
        """add_executor_iteration should truncate long tool results."""
        session = DebugSession(command="run")
        session.start_test("my-test")
        session.start_executor()

        long_result = "x" * (MAX_DEBUG_TOOL_RESULT_CHARS + 5000)
        session.add_executor_iteration(
            iteration=1,
            messages=[{"role": "user", "content": "test"}],
            response={"content": "response"},
            tool_calls=[{"name": "read_file", "result": long_result}]
        )

        stored = session._current_test["executor"]["iterations"][0]
        assert "truncated" in stored["tool_calls"][0]["result"]
        assert len(stored["tool_calls"][0]["result"]) < len(long_result)

    def test_executor_iteration_truncates_messages(self):
        """add_executor_iteration should truncate long message content."""
        session = DebugSession(command="run")
        session.start_test("my-test")
        session.start_executor()

        long_content = "y" * (MAX_DEBUG_MESSAGE_CONTENT_CHARS + 5000)
        session.add_executor_iteration(
            iteration=1,
            messages=[{"role": "user", "content": long_content}],
            response={"content": "response"},
            tool_calls=[]
        )

        stored = session._current_test["executor"]["iterations"][0]
        # _serialize_messages converts to dicts, then content is truncated
        first_msg = stored["messages_sent"][0]
        assert "truncated" in first_msg["content"]

    def test_judge_iteration_truncates_tool_calls(self):
        """add_judge_iteration should truncate long tool results."""
        session = DebugSession(command="run")
        session.start_test("my-test")
        session.start_judge("accuracy")

        long_result = "x" * (MAX_DEBUG_TOOL_RESULT_CHARS + 5000)
        session.add_judge_iteration(
            iteration=1,
            messages=[{"role": "system", "content": "eval"}],
            response={"verdict": "PASS"},
            tool_calls=[{"name": "read_file", "result": long_result}]
        )

        stored = session._current_judge["iterations"][0]
        assert "truncated" in stored["tool_calls"][0]["result"]

    def test_analyzer_iteration_truncates_tool_calls(self):
        """add_analyzer_iteration should truncate long tool results."""
        session = DebugSession(command="analyze")
        session.start_analyzer("cli-problems")

        long_result = "x" * (MAX_DEBUG_TOOL_RESULT_CHARS + 5000)
        session.add_analyzer_iteration(
            iteration=1,
            messages=[{"role": "user", "content": "analyze"}],
            response={"content": "found issues"},
            tool_calls=[{"name": "search", "result": long_result}]
        )

        stored = session._current_analyzer["iterations"][0]
        assert "truncated" in stored["tool_calls"][0]["result"]

    def test_scaffold_iteration_truncates_tool_calls(self):
        """add_scaffold_iteration should truncate long tool results."""
        session = DebugSession(command="add")
        session.start_scaffold_generation("docs/api.md")

        long_result = "x" * (MAX_DEBUG_TOOL_RESULT_CHARS + 5000)
        session.add_scaffold_iteration(
            iteration=1,
            messages=[{"role": "user", "content": "generate"}],
            response={"content": "yaml scaffold"},
            tool_calls=[{"name": "read_file", "result": long_result}]
        )

        stored = session.scaffold_generation["iterations"][0]
        assert "truncated" in stored["tool_calls"][0]["result"]

    def test_debug_trace_size_reasonable_with_large_content(self, tmp_path):
        """A debug trace with large tool results should still be manageable size."""
        set_debug(True)
        session = DebugSession(command="run", output_dir=str(tmp_path))
        session.start_test("large-test")
        session.start_executor()

        # Simulate 10 iterations with large tool results (like reading 140 pages)
        for i in range(10):
            large_result = "page content " * 10000  # ~130KB per result
            session.add_executor_iteration(
                iteration=i + 1,
                messages=[{"role": "user", "content": "x" * 50000}],
                response={"content": "response"},
                tool_calls=[{"name": "read_file", "result": large_result}]
            )

        session.finish_executor({"final_response": "done"})
        session.finish_test({"passed": True})
        session.flush_test("large-test")

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        file_size = files[0].stat().st_size
        # With truncation, 10 iterations should be well under 5MB
        assert file_size < 5 * 1024 * 1024, f"Debug trace too large: {file_size} bytes"


class TestSessionManagement:
    """Tests for session management functions."""

    def test_start_debug_session(self):
        """Should start and return new session."""
        session = start_debug_session("run", meta={"test": "value"})

        assert session is not None
        assert session.command == "run"
        assert session.meta == {"test": "value"}
        assert get_debug_session() is session

    def test_get_debug_session_returns_current(self):
        """Should return current session."""
        session = start_debug_session("run")
        assert get_debug_session() is session

    def test_get_debug_session_none_when_no_session(self):
        """Should return None when no session."""
        assert get_debug_session() is None

    def test_end_debug_session(self, tmp_path):
        """Should end session and write file."""
        session = start_debug_session("run")
        session.output_dir = str(tmp_path)

        path = end_debug_session()

        assert path is not None
        assert path.exists()
        assert get_debug_session() is None

    def test_end_debug_session_none_when_no_session(self):
        """Should return None when no session to end."""
        path = end_debug_session()
        assert path is None
