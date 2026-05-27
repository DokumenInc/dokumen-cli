"""Tests for test_object module."""

import base64
import os
import tempfile

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime


class TestFailureAnalysis:
    """Tests for FailureAnalysis dataclass."""

    def test_creation(self):
        """FailureAnalysis can be created."""
        from dokumen.test_object import FailureAnalysis
        from dokumen.file_object import IncorrectLine

        fa = FailureAnalysis(
            file_path="docs/api.md",
            referenced_lines=[1, 2, 3],
            incorrect_lines=[
                IncorrectLine(line_number=2, reason="Outdated", test_id="t1")
            ],
            analysis="Line 2 is outdated"
        )

        assert fa.file_path == "docs/api.md"
        assert fa.referenced_lines == [1, 2, 3]
        assert len(fa.incorrect_lines) == 1
        assert fa.analysis == "Line 2 is outdated"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        from dokumen.test_object import FailureAnalysis
        from dokumen.file_object import IncorrectLine

        fa = FailureAnalysis(
            file_path="docs/api.md",
            referenced_lines=[5, 6],
            incorrect_lines=[
                IncorrectLine(line_number=5, reason="Wrong", test_id="t1", confidence=0.9)
            ],
            analysis="Analysis text"
        )

        d = fa.to_dict()

        assert d["file_path"] == "docs/api.md"
        assert d["referenced_lines"] == [5, 6]
        assert len(d["incorrect_lines"]) == 1
        assert d["incorrect_lines"][0]["line_number"] == 5
        assert d["incorrect_lines"][0]["confidence"] == 0.9
        assert d["analysis"] == "Analysis text"


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_creation(self):
        """TestResult can be created."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.5,
            timestamp=datetime.now()
        )

        assert result.test_id == "test-1"
        assert result.passed is True
        assert result.executor_passed is True
        assert result.duration == 1.5

    def test_default_fields(self):
        """TestResult has correct default fields."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=False,
            executor_passed=False,
            judge_results=[],
            executor_output=None,
            duration=0.0,
            timestamp=datetime.now()
        )

        assert result.failure_reasons == []
        assert result.line_coverage == {}
        assert result.failure_analysis == {}

    def test_to_dict(self):
        """to_dict serializes correctly."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=2.5,
            timestamp=datetime(2025, 1, 1, 12, 0, 0)
        )

        d = result.to_dict()

        assert d["test_id"] == "test-1"
        assert d["passed"] is True
        assert d["executor_passed"] is True
        assert d["duration"] == 2.5
        assert "2025-01-01" in d["timestamp"]
        assert d["executor_output"] is None
        assert d["judge_results"] == []

    def test_to_dict_with_executor_output(self):
        """to_dict includes executor_output."""
        from dokumen.test_object import TestResult

        mock_executor_output = MagicMock()
        mock_executor_output.to_dict.return_value = {"content": "output"}

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=mock_executor_output,
            duration=1.0,
            timestamp=datetime.now()
        )

        d = result.to_dict()

        assert d["executor_output"] == {"content": "output"}

    def test_to_dict_with_line_coverage(self):
        """to_dict includes line_coverage."""
        from dokumen.test_object import TestResult
        from dokumen.file_object import LineCoverage

        line_cov = LineCoverage(
            file_path="docs/api.md",
            total_lines=100,
            covered_lines={1, 2, 3}
        )

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
            line_coverage={"docs/api.md": line_cov}
        )

        d = result.to_dict()

        assert "docs/api.md" in d["line_coverage"]
        assert d["line_coverage"]["docs/api.md"]["total_lines"] == 100

    def test_explore_status_defaults_to_none(self):
        """explore_status defaults to None when explore not run."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
        )

        assert result.explore_status is None

    def test_explore_status_pass(self):
        """explore_status can be set to 'pass'."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
            explore_status="pass",
        )

        assert result.explore_status == "pass"

    def test_explore_status_fail(self):
        """explore_status can be set to 'fail'."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=False,
            executor_passed=False,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
            explore_status="fail",
        )

        assert result.explore_status == "fail"

    def test_explore_status_in_to_dict(self):
        """to_dict includes explore_status."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            explore_status="pass",
        )

        d = result.to_dict()
        assert d["explore_status"] == "pass"

    def test_explore_status_none_in_to_dict(self):
        """to_dict includes explore_status as None when not set."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
        )

        d = result.to_dict()
        assert d["explore_status"] is None


    def test_judge_models_defaults_to_none(self):
        """judge_models defaults to None when not set."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
        )

        assert result.judge_models is None

    def test_judge_models_populated(self):
        """judge_models can be set with per-judge model map."""
        from dokumen.test_object import TestResult

        models = {"accuracy": "claude-opus-4-6", "format": "claude-haiku-4-5-20251001"}
        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
            judge_models=models,
        )

        assert result.judge_models == models

    def test_judge_models_in_to_dict(self):
        """to_dict includes judge_models key."""
        from dokumen.test_object import TestResult

        models = {"accuracy": "claude-opus-4-6", "format": "claude-haiku-4-5-20251001"}
        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            judge_models=models,
        )

        d = result.to_dict()
        assert d["judge_models"] == models

    def test_judge_models_none_in_to_dict(self):
        """to_dict includes judge_models as None when not set."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
        )

        d = result.to_dict()
        assert d["judge_models"] is None


class TestTestConfig:
    """Tests for TestConfig dataclass."""

    def test_creation(self):
        """TestConfig can be created."""
        from dokumen.test_object import TestConfig

        mock_executor = MagicMock()
        mock_judge = MagicMock()

        config = TestConfig(
            id="test-1",
            reason="Test reason",
            executor=mock_executor,
            judges=[mock_judge]
        )

        assert config.id == "test-1"
        assert config.reason == "Test reason"
        assert config.executor is mock_executor
        assert len(config.judges) == 1

    def test_defaults(self):
        """TestConfig has correct defaults."""
        from dokumen.test_object import TestConfig

        config = TestConfig(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[]
        )

        assert config.timeout == 60.0
        assert config.retries == 0


class TestTestObject:
    """Tests for TestObject class."""

    def test_init(self):
        """TestObject initializes correctly."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_judge = MagicMock()

        test = TestObject(
            id="my-test",
            reason="Test reason",
            executor=mock_executor,
            judges=[mock_judge],
            timeout=120.0,
            retries=2
        )

        assert test.id == "my-test"
        assert test.reason == "Test reason"
        assert test.executor is mock_executor
        assert len(test.judges) == 1
        assert test.timeout == 120.0
        assert test.retries == 2

    def test_repr(self):
        """TestObject has string representation."""
        from dokumen.test_object import TestObject

        test = TestObject(
            id="my-test",
            reason="Test",
            executor=MagicMock(),
            judges=[MagicMock(), MagicMock()]
        )

        repr_str = repr(test)
        assert "my-test" in repr_str
        assert "judges=2" in repr_str

    def test_get_hash(self):
        """get_hash returns consistent hash."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec-1"
        mock_executor.system_prompt = "System prompt"
        mock_executor.user_prompt = "User prompt"

        mock_judge = MagicMock()
        mock_judge.id = "judge-1"
        mock_judge.system_prompt = "Judge prompt"

        test = TestObject(
            id="my-test",
            reason="Test reason",
            executor=mock_executor,
            judges=[mock_judge],
            timeout=60.0
        )

        hash1 = test.get_hash()
        hash2 = test.get_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

    def test_get_hash_changes_with_config(self):
        """get_hash changes when config changes."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec-1"
        mock_executor.system_prompt = "System prompt"
        mock_executor.user_prompt = "User prompt"

        mock_judge = MagicMock()
        mock_judge.id = "judge-1"
        mock_judge.system_prompt = "Judge prompt"

        test = TestObject(
            id="my-test",
            reason="Test reason",
            executor=mock_executor,
            judges=[mock_judge]
        )

        hash1 = test.get_hash()

        # Change system prompt
        mock_executor.system_prompt = "Different prompt"
        hash2 = test.get_hash()

        assert hash1 != hash2

    def test_is_stale_no_cached_hash(self):
        """is_stale returns True when no cached hash."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[]
        )

        assert test.is_stale() is True

    def test_is_stale_with_matching_hash(self):
        """is_stale returns False when hash matches."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[]
        )

        test.set_cached_hash(test.get_hash())
        assert test.is_stale() is False

    def test_is_stale_with_different_hash(self):
        """is_stale returns True when hash differs."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[]
        )

        test.set_cached_hash("old-hash-value")
        assert test.is_stale() is True

    def test_get_hash_changes_with_setup_steps(self):
        """get_hash must change when setup_steps differ."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"

        test_no_setup = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[],
        )

        step = MagicMock()
        step.name = "install"
        step.command = "npm install"
        step.working_dir = None
        step.timeout = 60
        step.background = False
        step.ready_url = None
        step.ready_timeout = 30

        test_with_setup = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[],
            setup_steps=[step],
        )

        assert test_no_setup.get_hash() != test_with_setup.get_hash()

    def test_get_hash_changes_with_different_setup_command(self):
        """get_hash must change when setup command changes."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"

        step_a = MagicMock()
        step_a.name = "install"
        step_a.command = "npm install"
        step_a.working_dir = None
        step_a.timeout = 60
        step_a.background = False
        step_a.ready_url = None
        step_a.ready_timeout = 30

        step_b = MagicMock()
        step_b.name = "install"
        step_b.command = "yarn install"
        step_b.working_dir = None
        step_b.timeout = 60
        step_b.background = False
        step_b.ready_url = None
        step_b.ready_timeout = 30

        test_a = TestObject(
            id="test", reason="r", executor=mock_executor,
            judges=[], setup_steps=[step_a],
        )
        test_b = TestObject(
            id="test", reason="r", executor=mock_executor,
            judges=[], setup_steps=[step_b],
        )

        assert test_a.get_hash() != test_b.get_hash()

    def test_get_hash_changes_with_different_ready_timeout(self):
        """get_hash must change when setup ready_timeout changes."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"

        step_a = MagicMock()
        step_a.name = "server"
        step_a.command = "npm run dev"
        step_a.working_dir = None
        step_a.timeout = 60
        step_a.background = True
        step_a.ready_url = "http://localhost:3000"
        step_a.ready_timeout = 5

        step_b = MagicMock()
        step_b.name = "server"
        step_b.command = "npm run dev"
        step_b.working_dir = None
        step_b.timeout = 60
        step_b.background = True
        step_b.ready_url = "http://localhost:3000"
        step_b.ready_timeout = 30

        test_a = TestObject(
            id="test", reason="r", executor=mock_executor,
            judges=[], setup_steps=[step_a],
        )
        test_b = TestObject(
            id="test", reason="r", executor=mock_executor,
            judges=[], setup_steps=[step_b],
        )

        assert test_a.get_hash() != test_b.get_hash()

    def test_set_cached_hash(self):
        """set_cached_hash sets the cached hash."""
        from dokumen.test_object import TestObject

        test = TestObject(
            id="test",
            reason="r",
            executor=MagicMock(),
            judges=[]
        )

        test.set_cached_hash("my-hash")
        assert test._cached_hash == "my-hash"

    @pytest.mark.asyncio
    async def test_run_executor_success(self):
        """run executes executor and returns result."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"
        mock_executor.tools = []

        mock_output = ExecutorOutput(
            tool_calls=[],
            final_response="Result content",
            success=True
        )
        mock_executor.run = AsyncMock(return_value=mock_output)

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[]
        )

        result = await test.run()

        assert result.executor_passed is True
        assert result.passed is True  # No judges to fail
        assert result.executor_output is mock_output

    @pytest.mark.asyncio
    async def test_run_executor_failure(self):
        """run handles executor failure."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"
        mock_executor.tools = []

        mock_output = ExecutorOutput(
            tool_calls=[],
            final_response="",
            success=False,
            error="Executor failed"
        )
        mock_executor.run = AsyncMock(return_value=mock_output)

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[]
        )

        result = await test.run()

        assert result.executor_passed is False
        assert result.passed is False
        assert len(result.failure_reasons) > 0

    @pytest.mark.asyncio
    async def test_run_with_judges(self):
        """run executes all judges."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput, JudgeResult

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"
        mock_executor.tools = []

        mock_output = ExecutorOutput(tool_calls=[], final_response="Result", success=True)
        mock_executor.run = AsyncMock(return_value=mock_output)

        mock_judge1 = MagicMock()
        mock_judge1.id = "judge1"
        mock_judge1.tools = None
        mock_judge1.run = AsyncMock(return_value=JudgeResult(
            judge_id="judge1", passed=True
        ))

        mock_judge2 = MagicMock()
        mock_judge2.id = "judge2"
        mock_judge2.tools = None
        mock_judge2.run = AsyncMock(return_value=JudgeResult(
            judge_id="judge2", passed=True
        ))

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[mock_judge1, mock_judge2]
        )

        result = await test.run()

        assert result.passed is True
        assert len(result.judge_results) == 2
        assert all(jr.passed for jr in result.judge_results)

    @pytest.mark.asyncio
    async def test_run_judge_failure(self):
        """run handles judge failure."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput, JudgeResult

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.tools = []
        mock_executor.run = AsyncMock(return_value=ExecutorOutput(
            tool_calls=[], final_response="Result", success=True
        ))

        mock_judge = MagicMock()
        mock_judge.id = "judge1"
        mock_judge.tools = None
        mock_judge.run = AsyncMock(return_value=JudgeResult(
            judge_id="judge1", passed=False, failure_reason="Assertion failed"
        ))

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[mock_judge]
        )

        result = await test.run()

        assert result.executor_passed is True
        assert result.passed is False
        assert "Assertion failed" in str(result.failure_reasons)

    @pytest.mark.asyncio
    async def test_run_with_callbacks(self):
        """run calls callbacks."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput, JudgeResult

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.tools = []
        mock_executor.run = AsyncMock(return_value=ExecutorOutput(
            tool_calls=[], final_response="Result", success=True
        ))

        mock_judge = MagicMock()
        mock_judge.id = "judge1"
        mock_judge.tools = None
        mock_judge.run = AsyncMock(return_value=JudgeResult(
            judge_id="judge1", passed=True
        ))

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[mock_judge]
        )

        executor_complete_called = []
        judge_complete_called = []

        def on_executor_complete(output):
            executor_complete_called.append(output)

        def on_judge_complete(jr):
            judge_complete_called.append(jr)

        result = await test.run(
            on_executor_complete=on_executor_complete,
            on_judge_complete=on_judge_complete
        )

        assert len(executor_complete_called) == 1
        assert len(judge_complete_called) == 1

    @pytest.mark.asyncio
    async def test_run_with_retries(self):
        """run retries on executor failure."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.tools = []

        # Fail first, succeed second
        call_count = [0]

        async def mock_run(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ExecutorOutput(tool_calls=[], final_response="", success=False, error="First fail")
            return ExecutorOutput(tool_calls=[], final_response="Result", success=True)

        mock_executor.run = mock_run

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[],
            retries=1
        )

        result = await test.run()

        assert call_count[0] == 2
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_executor_exception(self):
        """run handles executor exception."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.tools = []
        mock_executor.run = AsyncMock(side_effect=RuntimeError("Executor error"))

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[]
        )

        result = await test.run()

        assert result.passed is False
        assert "Executor error" in str(result.failure_reasons)

    @pytest.mark.asyncio
    async def test_run_judge_exception(self):
        """run handles judge exception."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.tools = []
        mock_executor.run = AsyncMock(return_value=ExecutorOutput(
            tool_calls=[], final_response="Result", success=True
        ))

        mock_judge = MagicMock()
        mock_judge.id = "judge1"
        mock_judge.tools = None
        mock_judge.run = AsyncMock(side_effect=RuntimeError("Judge error"))

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[mock_judge]
        )

        result = await test.run()

        assert result.passed is False
        assert len(result.judge_results) == 1
        assert result.judge_results[0].passed is False
        assert "Judge error" in str(result.failure_reasons)


class TestResolveToolsWithSandbox:
    """Tests for _resolve_tools_with_sandbox method."""

    def test_base_dir_uses_project_root(self):
        """_resolve_tools_with_sandbox uses project root as base_dir."""
        from dokumen.test_object import TestObject
        from dokumen.tools_object import ToolDefinition

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.tools = [
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                handler=AsyncMock()
            )
        ]
        mock_executor.provider = MagicMock()

        # Create TestObject with source_path in a subdirectory
        test = TestObject(
            id="test",
            reason="Test path resolution",
            executor=mock_executor,
            judges=[],
            source_path="/project/tests/subdir/my-test.yaml"
        )

        mock_sandbox = MagicMock()

        # Patch both resolve_tools and find_project_root
        with patch('dokumen.loader.resolve_tools') as mock_resolve, \
             patch('dokumen.loader.find_project_root') as mock_find_root:
            mock_resolve.return_value = []
            mock_find_root.return_value = "/project"

            test._resolve_tools_with_sandbox(mock_sandbox)

            # Verify find_project_root was called with source_path
            mock_find_root.assert_called_once_with("/project/tests/subdir/my-test.yaml")

            # Verify resolve_tools was called with the project root
            mock_resolve.assert_called_once()
            call_kwargs = mock_resolve.call_args.kwargs
            assert call_kwargs.get('base_dir') == "/project", \
                f"Expected base_dir='/project', got {call_kwargs.get('base_dir')}"

    def test_base_dir_defaults_to_cwd_when_no_source_path(self):
        """_resolve_tools_with_sandbox defaults to '.' when source_path is None."""
        from dokumen.test_object import TestObject
        from dokumen.tools_object import ToolDefinition

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.tools = [
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                handler=AsyncMock()
            )
        ]
        mock_executor.provider = MagicMock()

        # Create TestObject without source_path
        test = TestObject(
            id="test",
            reason="Test path resolution",
            executor=mock_executor,
            judges=[],
            source_path=None
        )

        mock_sandbox = MagicMock()

        with patch('dokumen.loader.resolve_tools') as mock_resolve:
            mock_resolve.return_value = []

            test._resolve_tools_with_sandbox(mock_sandbox)

            call_kwargs = mock_resolve.call_args.kwargs
            assert call_kwargs.get('base_dir') == ".", \
                f"Expected base_dir='.', got {call_kwargs.get('base_dir')}"


class TestTestObjectExplore:
    """Tests for explore phase in TestObject."""

    def test_init_with_explore_config(self):
        """TestObject can be initialized with explore config."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig

        mock_executor = MagicMock()
        explore_config = ExploreConfig(enabled=True, model="claude-haiku-4-5-20251001")

        test = TestObject(
            id="my-test",
            reason="Test",
            executor=mock_executor,
            judges=[],
            explore_config=explore_config
        )

        assert test.explore_config is not None
        assert test.explore_config.enabled is True
        assert test.explore_config.model == "claude-haiku-4-5-20251001"

    def test_init_without_explore_config(self):
        """TestObject works without explore config (defaults to None)."""
        from dokumen.test_object import TestObject

        test = TestObject(
            id="my-test",
            reason="Test",
            executor=MagicMock(),
            judges=[]
        )

        assert test.explore_config is None

    @pytest.mark.asyncio
    async def test_run_with_explore_enabled(self):
        """run() executes explore phase when enabled."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from dokumen.explore_agent import ExploreResult, FileDiscovery
        from dokumen.stages.explore import ExploreStage
        from mock_provider import MockProvider

        # Create mock executor with provider
        mock_executor = MagicMock()
        mock_executor.run = AsyncMock(return_value=MagicMock(
            success=True,
            final_response="Done",
            tool_calls=[],
            error=None,
            system_prompt="",
            user_prompt=""
        ))
        mock_executor.system_prompt = "System"
        mock_executor.user_prompt = "User"
        mock_executor.tools = []
        mock_executor.provider = MockProvider()

        # Create explore config
        explore_config = ExploreConfig(enabled=True)

        test = TestObject(
            id="test",
            reason="Test explore",
            executor=mock_executor,
            judges=[],
            explore_config=explore_config
        )

        # Mock the explore agent at the stage level
        mock_explore_result = ExploreResult(
            files=[FileDiscovery(path="docs/api.md", summary="API docs", relevance=0.9)],
            duration=1.0,
            tool_calls_count=2,
            success=True
        )

        mock_run_explore = AsyncMock(return_value=mock_explore_result)
        with patch.object(ExploreStage, '_run_explore', mock_run_explore):
            result = await test.run()

        # Explore should have run
        mock_run_explore.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio
    async def test_run_with_explore_disabled(self):
        """run() skips explore phase when disabled."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from mock_provider import MockProvider

        mock_executor = MagicMock()
        mock_executor.run = AsyncMock(return_value=MagicMock(
            success=True,
            final_response="Done",
            tool_calls=[],
            error=None,
            system_prompt="",
            user_prompt=""
        ))
        mock_executor.system_prompt = "System"
        mock_executor.user_prompt = "User"
        mock_executor.tools = []
        mock_executor.provider = MockProvider()

        # Explore disabled
        explore_config = ExploreConfig(enabled=False)

        test = TestObject(
            id="test",
            reason="Test no explore",
            executor=mock_executor,
            judges=[],
            explore_config=explore_config
        )

        with patch.object(test, '_run_explore', AsyncMock()) as mock_explore:
            result = await test.run()

        # Explore should NOT have run
        mock_explore.assert_not_called()

    @pytest.mark.asyncio
    async def test_explore_callback_called(self):
        """run() calls on_explore_event callback during exploration."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from dokumen.explore_agent import ExploreResult, FileDiscovery
        from dokumen.stages.explore import ExploreStage
        from mock_provider import MockProvider

        mock_executor = MagicMock()
        mock_executor.run = AsyncMock(return_value=MagicMock(
            success=True,
            final_response="Done",
            tool_calls=[],
            error=None,
            system_prompt="",
            user_prompt=""
        ))
        mock_executor.system_prompt = "System"
        mock_executor.user_prompt = "User"
        mock_executor.tools = []
        mock_executor.provider = MockProvider()

        explore_config = ExploreConfig(enabled=True)

        test = TestObject(
            id="test",
            reason="Test callback",
            executor=mock_executor,
            judges=[],
            explore_config=explore_config
        )

        callback_events = []
        def on_explore(event_type, data):
            callback_events.append((event_type, data))

        mock_result = ExploreResult(
            files=[],
            duration=1.0,
            tool_calls_count=0,
            success=True
        )

        mock_run_explore = AsyncMock(return_value=mock_result)
        with patch.object(ExploreStage, '_run_explore', mock_run_explore):
            await test.run(on_explore_event=on_explore)

        # Callback should have been passed to _run_explore (now via ExploreStage)
        mock_run_explore.assert_called_once()

    @pytest.mark.asyncio
    async def test_explore_context_injected(self):
        """Explore results are injected into executor user_prompt."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from dokumen.explore_agent import ExploreResult, FileDiscovery
        from dokumen.stages.explore import ExploreStage
        from mock_provider import MockProvider

        mock_executor = MagicMock()
        original_user_prompt = "Find the refund policy"
        mock_executor.user_prompt = original_user_prompt
        mock_executor.system_prompt = "System"
        mock_executor.run = AsyncMock(return_value=MagicMock(
            success=True,
            final_response="Done",
            tool_calls=[],
            error=None,
            system_prompt="",
            user_prompt=""
        ))
        mock_executor.tools = []
        mock_executor.provider = MockProvider()

        explore_config = ExploreConfig(enabled=True)

        test = TestObject(
            id="test",
            reason="Test context injection",
            executor=mock_executor,
            judges=[],
            explore_config=explore_config
        )

        # Mock explore result with discovered files
        mock_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/refund.md", summary="Refund policy", relevance=0.9)
            ],
            duration=1.0,
            tool_calls_count=2,
            success=True
        )

        with patch.object(ExploreStage, '_run_explore', AsyncMock(return_value=mock_result)):
            with patch.object(ExploreStage, '_inject_explore_context') as mock_inject:
                await test.run()

        # Context should be injected (now via ExploreStage)
        mock_inject.assert_called_once()


class TestExploreFileVerification:
    """Tests for explore phase file verification."""

    def test_verify_explore_found_files_all_found(self):
        """Returns empty list when all required files are found in summary."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[MagicMock()],
            files=["docs/refund-policy.md", "docs/auth.md"]
        )

        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/refund-policy.md", summary="Refund policy", relevance=0.9),
                FileDiscovery(path="docs/auth.md", summary="Auth docs", relevance=0.8),
            ],
            duration=1.0,
            tool_calls_count=2,
            success=True,
            summary="Found docs/refund-policy.md and docs/auth.md"
        )

        missing = test._verify_explore_found_files(explore_result)
        assert missing == []

    def test_verify_explore_found_files_missing(self):
        """Returns list of missing file paths."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[MagicMock()],
            files=["docs/refund-policy.md", "docs/missing.md"]
        )

        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/refund-policy.md", summary="Refund policy", relevance=0.9),
            ],
            duration=1.0,
            tool_calls_count=2,
            success=True,
            summary="Found docs/refund-policy.md only"
        )

        missing = test._verify_explore_found_files(explore_result)
        assert missing == ["docs/missing.md"]

    def test_verify_explore_checks_files_list(self):
        """Verification checks files list when summary doesn't contain path."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[MagicMock()],
            files=["docs/policy.md"]
        )

        # Summary doesn't contain the path, but files list does
        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/policy.md", summary="Policy docs", relevance=0.9),
            ],
            duration=1.0,
            tool_calls_count=1,
            success=True,
            summary="Found relevant documentation"  # No explicit path
        )

        missing = test._verify_explore_found_files(explore_result)
        assert missing == []

    def test_verify_normalize_leading_dot_slash(self):
        """Verification normalizes leading ./ in found paths."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[MagicMock()],
            files=["docs/file.md"]
        )

        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="./docs/file.md", summary="File", relevance=0.9),
            ],
            duration=1.0,
            tool_calls_count=1,
            success=True,
            summary="Found ./docs/file.md"
        )

        missing = test._verify_explore_found_files(explore_result)
        assert missing == []

    def test_verify_normalize_found_dot_slash(self):
        """Verification normalizes leading ./ in required paths."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[MagicMock()],
            files=["./docs/file.md"]
        )

        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/file.md", summary="File", relevance=0.9),
            ],
            duration=1.0,
            tool_calls_count=1,
            success=True,
            summary="Found docs/file.md"
        )

        missing = test._verify_explore_found_files(explore_result)
        assert missing == []

    def test_verify_normalize_double_slash(self):
        """Verification normalizes double slashes in paths."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[MagicMock()],
            files=["docs/sub/file.md"]
        )

        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/sub//file.md", summary="File", relevance=0.9),
            ],
            duration=1.0,
            tool_calls_count=1,
            success=True,
            summary="Found docs/sub//file.md"
        )

        missing = test._verify_explore_found_files(explore_result)
        assert missing == []

    def test_verify_list_directory_full_path_match(self):
        """Verification matches full paths from list_directory results."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[MagicMock()],
            files=["docs/CanBSCR_Standards-1/page-16.md"]
        )

        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/CanBSCR_Standards-1/page-16.md", summary="Page 16", relevance=0.9),
            ],
            duration=1.0,
            tool_calls_count=1,
            success=True,
            summary="Found docs/CanBSCR_Standards-1/page-16.md"
        )

        missing = test._verify_explore_found_files(explore_result)
        assert missing == []

    def test_fail_with_missing_files_error_verbose(self):
        """Error message includes found/missing status for each file."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[MagicMock()],
            files=["docs/found.md", "docs/missing.md"]
        )

        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/found.md", summary="Found file", relevance=0.9),
            ],
            duration=1.0,
            tool_calls_count=1,
            success=True,
            summary="Only found docs/found.md"
        )

        missing = ["docs/missing.md"]
        result = test._fail_with_missing_files_error(missing, explore_result)

        assert result.passed is False
        assert "EXPLORE PHASE FAILED" in result.failure_reasons[0]
        assert "[FOUND] docs/found.md" in result.failure_reasons[0]
        assert "[MISSING] docs/missing.md" in result.failure_reasons[0]

    def test_inject_explore_context_uses_summary(self):
        """_inject_explore_context uses summary field when available."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult

        mock_executor = MagicMock()
        mock_executor.user_prompt = "Original prompt"

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=mock_executor,
            judges=[MagicMock()]
        )

        explore_result = ExploreResult(
            files=[],
            duration=1.0,
            tool_calls_count=1,
            success=True,
            summary="Found docs/policy.md with refund policy details."
        )

        test._inject_explore_context(explore_result)

        assert "Pre-discovered Documentation" in mock_executor.user_prompt
        assert "Found docs/policy.md" in mock_executor.user_prompt
        assert "Original prompt" in mock_executor.user_prompt

    @pytest.mark.asyncio
    async def test_explore_always_runs_for_tests(self):
        """Explore phase always runs for tests with files regardless of explore_config.enabled."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from dokumen.explore_agent import ExploreResult, FileDiscovery
        from dokumen.stages.explore import ExploreStage

        mock_executor = MagicMock()
        mock_executor.run = AsyncMock(return_value=MagicMock(success=True, output="Result", error=None))
        mock_executor.id = "exec-1"
        mock_executor.system_prompt = "System"
        mock_executor.user_prompt = "User prompt"
        mock_executor.tools = []
        mock_executor.provider = MagicMock()

        mock_judge = MagicMock()
        mock_judge.run = AsyncMock(return_value=MagicMock(
            verdict="PASS",
            confidence=0.9,
            reason="Good",
            judge_id="judge-1"
        ))
        mock_judge.id = "judge-1"
        mock_judge.tools = None

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=mock_executor,
            judges=[mock_judge],
            files=["docs/test.md"],
            explore_config=ExploreConfig(enabled=False)  # Disabled, but should still run because files exist
        )

        mock_explore_result = ExploreResult(
            files=[FileDiscovery(path="docs/test.md", summary="Test docs", relevance=0.9)],
            duration=1.0,
            tool_calls_count=1,
            success=True,
            summary="Found docs/test.md"
        )

        with patch.object(ExploreStage, '_run_explore', AsyncMock(return_value=mock_explore_result)) as mock_run:
            await test.run()

        # Explore should have been called even though enabled=False (files trigger it)
        mock_run.assert_called_once()


class TestTestResultExecutorTools:
    """Tests for executor_tools field on TestResult."""

    def test_executor_tools_default_empty(self):
        """executor_tools defaults to empty list."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now()
        )

        assert result.executor_tools == []

    def test_executor_tools_set_on_creation(self):
        """executor_tools can be set on creation."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
            executor_tools=["read_file", "glob", "run_shell_command"]
        )

        assert result.executor_tools == ["read_file", "glob", "run_shell_command"]

    def test_executor_tools_in_to_dict(self):
        """to_dict includes executor_tools."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now(),
            executor_tools=["read_file", "glob"]
        )

        d = result.to_dict()

        assert "executor_tools" in d
        assert d["executor_tools"] == ["read_file", "glob"]

    def test_executor_tools_empty_in_to_dict(self):
        """to_dict includes empty executor_tools when not set."""
        from dokumen.test_object import TestResult

        result = TestResult(
            test_id="test-1",
            passed=True,
            executor_passed=True,
            judge_results=[],
            executor_output=None,
            duration=1.0,
            timestamp=datetime.now()
        )

        d = result.to_dict()

        assert "executor_tools" in d
        assert d["executor_tools"] == []

    @pytest.mark.asyncio
    async def test_run_populates_executor_tools(self):
        """TestObject.run() populates executor_tools from executor tools."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput
        from dokumen.tools_object import ToolDefinition

        mock_executor = MagicMock()
        mock_executor.id = "exec"
        mock_executor.system_prompt = "sp"
        mock_executor.user_prompt = "up"
        mock_executor.tools = [
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters={"type": "object", "properties": {}},
                handler=AsyncMock()
            ),
            ToolDefinition(
                name="glob",
                description="Glob files",
                parameters={"type": "object", "properties": {}},
                handler=AsyncMock()
            ),
        ]
        mock_executor.provider = None

        mock_output = ExecutorOutput(
            tool_calls=[], final_response="Done", success=True
        )
        mock_executor.run = AsyncMock(return_value=mock_output)

        test = TestObject(
            id="test",
            reason="r",
            executor=mock_executor,
            judges=[]
        )

        result = await test.run()

        assert result.executor_tools == ["read_file", "glob"]


class TestDeterministicFileVerification:
    """Tests for deterministic file verification fallback.

    When the AI explore phase misses required files, the system should check
    the filesystem directly before declaring them missing.
    """

    def test_check_files_on_disk_finds_existing_file(self, tmp_path):
        """Files that exist on disk are removed from the missing list."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        # Create a real file on disk
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "page-63.md").write_text("# Page 63 content")

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[],
            files=["docs/page-63.md"],
            source_path=str(tmp_path / "tests" / "test.yaml")
        )

        explore_result = ExploreResult(
            files=[],
            duration=1.0,
            tool_calls_count=5,
            success=True,
            summary="Found some files"
        )

        with patch.object(test, '_get_base_dir', return_value=str(tmp_path)):
            still_missing = test._check_files_on_disk(
                ["docs/page-63.md"], explore_result
            )

        assert still_missing == []

    def test_check_files_on_disk_returns_truly_missing(self, tmp_path):
        """Files that don't exist on disk remain in the missing list."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[],
            files=["docs/nonexistent.md"],
            source_path=str(tmp_path / "tests" / "test.yaml")
        )

        explore_result = ExploreResult(
            files=[],
            duration=1.0,
            tool_calls_count=5,
            success=True,
        )

        with patch.object(test, '_get_base_dir', return_value=str(tmp_path)):
            still_missing = test._check_files_on_disk(
                ["docs/nonexistent.md"], explore_result
            )

        assert still_missing == ["docs/nonexistent.md"]

    def test_check_files_on_disk_adds_found_to_explore_result(self, tmp_path):
        """Found files are added to explore_result.files for context injection."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        # Create real file
        docs_dir = tmp_path / "docs" / "canbscr"
        docs_dir.mkdir(parents=True)
        (docs_dir / "page-63.md").write_text("# Fire resistance")

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[],
            files=["docs/canbscr/page-63.md"],
        )

        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/canbscr/page-62.md", summary="Page 62", relevance=0.9),
            ],
            duration=1.0,
            tool_calls_count=8,
            success=True,
        )

        with patch.object(test, '_get_base_dir', return_value=str(tmp_path)):
            test._check_files_on_disk(
                ["docs/canbscr/page-63.md"], explore_result
            )

        # explore_result should now include page-63 too
        found_paths = [f.path for f in explore_result.files]
        assert "docs/canbscr/page-63.md" in found_paths

    def test_check_files_on_disk_mixed_found_and_missing(self, tmp_path):
        """Only truly missing files are returned; found ones are added to result."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "exists.md").write_text("content")
        # does-not-exist.md is NOT created

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[],
            files=["docs/exists.md", "docs/does-not-exist.md"],
        )

        explore_result = ExploreResult(
            files=[],
            duration=1.0,
            tool_calls_count=3,
            success=True,
        )

        with patch.object(test, '_get_base_dir', return_value=str(tmp_path)):
            still_missing = test._check_files_on_disk(
                ["docs/exists.md", "docs/does-not-exist.md"], explore_result
            )

        assert still_missing == ["docs/does-not-exist.md"]
        found_paths = [f.path for f in explore_result.files]
        assert "docs/exists.md" in found_paths

    def test_check_files_on_disk_recovered_file_has_low_relevance(self, tmp_path):
        """Recovered files have lower relevance than explore-discovered files."""
        from dokumen.test_object import TestObject
        from dokumen.explore_agent import ExploreResult, FileDiscovery

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "recovered.md").write_text("content")

        test = TestObject(
            id="test-1",
            reason="Test",
            executor=MagicMock(),
            judges=[],
            files=["docs/recovered.md"],
        )

        explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/other.md", summary="Other", relevance=0.9),
            ],
            duration=1.0,
            tool_calls_count=5,
            success=True,
        )

        with patch.object(test, '_get_base_dir', return_value=str(tmp_path)):
            test._check_files_on_disk(
                ["docs/recovered.md"], explore_result
            )

        recovered = next(f for f in explore_result.files if f.path == "docs/recovered.md")
        assert recovered.relevance < 0.9  # Lower than explore-discovered

    @pytest.mark.asyncio
    async def test_explore_context_injected_on_timeout_with_partial_files(self):
        """Explore context is injected even when explore timed out but found files."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from dokumen.explore_agent import ExploreResult, FileDiscovery
        from dokumen.stages.explore import ExploreStage

        mock_executor = MagicMock()
        mock_executor.user_prompt = "Find fire resistance info"
        mock_executor.system_prompt = "System"
        mock_executor.run = AsyncMock(return_value=MagicMock(
            success=True,
            final_response="Done",
            tool_calls=[],
            error=None,
            system_prompt="",
            user_prompt=""
        ))
        mock_executor.tools = []
        mock_executor.provider = MagicMock()

        explore_config = ExploreConfig(enabled=True)

        test = TestObject(
            id="test-timeout",
            reason="Test context injection on timeout",
            executor=mock_executor,
            judges=[],
            explore_config=explore_config
        )

        # Explore timed out (success=False) but found partial files
        mock_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/page-62.md", summary="Page 62", relevance=0.7),
                FileDiscovery(path="docs/page-63.md", summary="Page 63", relevance=0.7),
            ],
            duration=60.0,
            tool_calls_count=20,
            success=False,
            error="Exploration timeout",
            summary="Exploration timed out but found 2 files"
        )

        with patch.object(ExploreStage, '_run_explore', AsyncMock(return_value=mock_result)):
            with patch.object(ExploreStage, '_inject_explore_context') as mock_inject:
                await test.run()

        # Context SHOULD be injected even though success=False (now via ExploreStage)
        mock_inject.assert_called_once()

    @pytest.mark.asyncio
    async def test_explore_context_not_injected_when_no_files_and_no_summary(self):
        """Explore context is NOT injected when explore failed with no files at all."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from dokumen.explore_agent import ExploreResult
        from dokumen.stages.explore import ExploreStage

        mock_executor = MagicMock()
        mock_executor.user_prompt = "Find info"
        mock_executor.system_prompt = "System"
        mock_executor.run = AsyncMock(return_value=MagicMock(
            success=True,
            final_response="Done",
            tool_calls=[],
            error=None,
            system_prompt="",
            user_prompt=""
        ))
        mock_executor.tools = []
        mock_executor.provider = MagicMock()

        explore_config = ExploreConfig(enabled=True)

        test = TestObject(
            id="test-empty-fail",
            reason="Test no injection when fully failed",
            executor=mock_executor,
            judges=[],
            explore_config=explore_config
        )

        # Explore failed with zero files and no useful summary
        mock_result = ExploreResult(
            files=[],
            duration=5.0,
            tool_calls_count=0,
            success=False,
            error="No LLM provider configured",
        )

        with patch.object(ExploreStage, '_run_explore', AsyncMock(return_value=mock_result)):
            with patch.object(ExploreStage, '_inject_explore_context') as mock_inject:
                await test.run()

        # Context should NOT be injected — nothing useful to inject (via ExploreStage)
        mock_inject.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_recovers_files_missed_by_explore(self, tmp_path):
        """Full integration: run() recovers files missed by explore and continues."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from dokumen.explore_agent import ExploreResult, FileDiscovery
        from dokumen.agent_object import ExecutorOutput
        from dokumen.stages.explore import ExploreStage

        # Create the required file on disk
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "page-63.md").write_text("# Fire Resistance Section 3.7.3")

        mock_executor = MagicMock()
        mock_executor.system_prompt = "System"
        mock_executor.user_prompt = "Find fire resistance info"
        mock_executor.tools = []
        mock_executor.provider = MagicMock()
        mock_executor.run = AsyncMock(return_value=ExecutorOutput(
            tool_calls=[], final_response="Found it", success=True
        ))

        test = TestObject(
            id="test-fire",
            reason="Test fire resistance",
            executor=mock_executor,
            judges=[],
            files=["docs/page-63.md"],
            explore_config=ExploreConfig(enabled=True),
            source_path=str(tmp_path / "tests" / "test.yaml")
        )

        # Explore finds page-62 but misses page-63 (the bug scenario)
        mock_explore_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/page-62.md", summary="Page 62", relevance=0.9),
            ],
            duration=2.0,
            tool_calls_count=8,
            success=True,
            summary="Found docs/page-62.md"
        )

        with patch.object(ExploreStage, '_run_explore', AsyncMock(return_value=mock_explore_result)):
            with patch.object(ExploreStage, '_get_base_dir', return_value=str(tmp_path)):
                result = await test.run()

        # Test should NOT fail from missing files — the file was recovered
        assert result.passed is True
        assert "EXPLORE PHASE FAILED" not in str(result.failure_reasons)


class TestBrowserHeadlessDetection:
    """Tests for browser headless mode CI detection."""

    def test_headless_env_var_true_forces_headless(self, monkeypatch):
        """DOKUMEN_BROWSER_HEADLESS=true forces headless mode."""
        from dokumen.test_object import resolve_browser_headless

        monkeypatch.setenv("DOKUMEN_BROWSER_HEADLESS", "true")
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("GITLAB_CI", raising=False)

        result = resolve_browser_headless(config_headless=None)
        assert result is True

    def test_headless_env_var_1_forces_headless(self, monkeypatch):
        """DOKUMEN_BROWSER_HEADLESS=1 forces headless mode."""
        from dokumen.test_object import resolve_browser_headless

        monkeypatch.setenv("DOKUMEN_BROWSER_HEADLESS", "1")
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("GITLAB_CI", raising=False)

        result = resolve_browser_headless(config_headless=None)
        assert result is True

    def test_ci_env_var_auto_enables_headless(self, monkeypatch):
        """CI=true auto-enables headless mode."""
        from dokumen.test_object import resolve_browser_headless

        monkeypatch.delenv("DOKUMEN_BROWSER_HEADLESS", raising=False)
        monkeypatch.setenv("CI", "true")
        monkeypatch.delenv("GITLAB_CI", raising=False)

        result = resolve_browser_headless(config_headless=None)
        assert result is True

    def test_gitlab_ci_env_var_auto_enables_headless(self, monkeypatch):
        """GITLAB_CI=true auto-enables headless mode."""
        from dokumen.test_object import resolve_browser_headless

        monkeypatch.delenv("DOKUMEN_BROWSER_HEADLESS", raising=False)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("GITLAB_CI", "true")

        result = resolve_browser_headless(config_headless=None)
        assert result is True

    def test_explicit_config_overrides_env_detection(self, monkeypatch):
        """Explicit YAML config headless=False overrides CI env detection."""
        from dokumen.test_object import resolve_browser_headless

        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("DOKUMEN_BROWSER_HEADLESS", "true")

        # Explicit False in config should override
        result = resolve_browser_headless(config_headless=False)
        assert result is False

    def test_explicit_config_true_works(self, monkeypatch):
        """Explicit YAML config headless=True works."""
        from dokumen.test_object import resolve_browser_headless

        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("GITLAB_CI", raising=False)
        monkeypatch.delenv("DOKUMEN_BROWSER_HEADLESS", raising=False)

        result = resolve_browser_headless(config_headless=True)
        assert result is True

    def test_no_env_vars_defaults_to_false(self, monkeypatch):
        """Without CI env vars, defaults to False (headful)."""
        from dokumen.test_object import resolve_browser_headless

        monkeypatch.delenv("DOKUMEN_BROWSER_HEADLESS", raising=False)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("GITLAB_CI", raising=False)

        result = resolve_browser_headless(config_headless=None)
        assert result is False


class TestOriginalUserPromptPreservation:
    """Tests for preserving original_user_prompt when explore context is injected."""

    @pytest.mark.asyncio
    async def test_original_user_prompt_preserved_with_explore(self):
        """original_user_prompt is passed to executor.run() before explore injection."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from dokumen.explore_agent import ExploreResult, FileDiscovery
        from dokumen.agent_object import ExecutorOutput
        from mock_provider import MockProvider

        # Create mock executor that captures the original_user_prompt parameter
        captured_kwargs = {}
        original_prompt = "Find information about refund policy"

        async def mock_run(**kwargs):
            captured_kwargs.update(kwargs)
            return ExecutorOutput(
                tool_calls=[],
                final_response="Done",
                success=True,
                system_prompt="System",
                user_prompt="Modified prompt",  # This would be the explore-modified prompt
                original_user_prompt=kwargs.get('original_user_prompt', '')
            )

        mock_executor = MagicMock()
        mock_executor.user_prompt = original_prompt
        mock_executor.system_prompt = "System"
        mock_executor.run = mock_run
        mock_executor.tools = []
        mock_executor.provider = MockProvider()

        explore_config = ExploreConfig(enabled=True)

        test = TestObject(
            id="test",
            reason="Test original prompt preservation",
            executor=mock_executor,
            judges=[],
            explore_config=explore_config
        )

        # Mock explore result that will inject context
        mock_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/refund.md", summary="Refund policy", relevance=0.9)
            ],
            duration=1.0,
            tool_calls_count=2,
            success=True,
            summary="Found docs/refund.md with refund policy details"
        )

        with patch.object(test, '_run_explore', AsyncMock(return_value=mock_result)):
            result = await test.run()

        # Verify original_user_prompt was passed to executor.run()
        assert 'original_user_prompt' in captured_kwargs
        assert captured_kwargs['original_user_prompt'] == original_prompt

    @pytest.mark.asyncio
    async def test_original_user_prompt_in_executor_output(self):
        """ExecutorOutput.original_user_prompt contains the original prompt."""
        from dokumen.test_object import TestObject
        from dokumen.config import ExploreConfig
        from dokumen.explore_agent import ExploreResult, FileDiscovery
        from dokumen.agent_object import ExecutorOutput
        from mock_provider import MockProvider

        original_prompt = "Check the API documentation for auth endpoints"

        async def mock_run(**kwargs):
            # Simulate what happens in real executor - store original_user_prompt
            return ExecutorOutput(
                tool_calls=[],
                final_response="Done",
                success=True,
                system_prompt="System",
                user_prompt="Modified with explore context",  # This is the modified prompt
                original_user_prompt=kwargs.get('original_user_prompt', '')
            )

        mock_executor = MagicMock()
        mock_executor.user_prompt = original_prompt
        mock_executor.system_prompt = "System"
        mock_executor.run = mock_run
        mock_executor.tools = []
        mock_executor.provider = MockProvider()

        explore_config = ExploreConfig(enabled=True)

        test = TestObject(
            id="test",
            reason="Test",
            executor=mock_executor,
            judges=[],
            explore_config=explore_config
        )

        mock_result = ExploreResult(
            files=[
                FileDiscovery(path="docs/api.md", summary="API docs", relevance=0.9)
            ],
            duration=1.0,
            tool_calls_count=2,
            success=True,
            summary="Found API documentation"
        )

        with patch.object(test, '_run_explore', AsyncMock(return_value=mock_result)):
            result = await test.run()

        # Verify the result has executor_output with original_user_prompt
        assert result.executor_output is not None
        assert result.executor_output.original_user_prompt == original_prompt

    @pytest.mark.asyncio
    async def test_original_user_prompt_without_explore(self):
        """original_user_prompt equals user_prompt when no explore injection."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput

        original_prompt = "Simple prompt without explore"

        async def mock_run(**kwargs):
            return ExecutorOutput(
                tool_calls=[],
                final_response="Done",
                success=True,
                system_prompt="System",
                user_prompt=original_prompt,
                original_user_prompt=kwargs.get('original_user_prompt', '')
            )

        mock_executor = MagicMock()
        mock_executor.user_prompt = original_prompt
        mock_executor.system_prompt = "System"
        mock_executor.run = mock_run
        mock_executor.tools = []
        mock_executor.provider = None  # No provider

        test = TestObject(
            id="test",
            reason="Test",
            executor=mock_executor,
            judges=[],
            explore_config=None,  # No explore
            files=[]  # No required files
        )

        result = await test.run()

        # Without explore, original_user_prompt should equal user_prompt
        assert result.executor_output is not None
        assert result.executor_output.original_user_prompt == original_prompt


class TestBrowserPlaywrightMissing:
    """Tests for graceful handling when Playwright is not available."""

    def _make_browser_tool(self):
        """Create a fake browser tool definition."""
        from dokumen.tools_object import ToolDefinition

        async def noop_handler(params):
            pass

        return ToolDefinition(
            name="browser_navigate",
            description="Navigate to URL",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}},
            handler=noop_handler,
        )

    @pytest.mark.asyncio
    async def test_missing_playwright_returns_failed_result(self):
        """Browser test returns failed result when Playwright MCP is not installed."""
        from dokumen.test_object import TestObject, BrowserConfig

        browser_tool = self._make_browser_tool()

        mock_executor = MagicMock()
        mock_executor.user_prompt = "Navigate to example.com"
        mock_executor.system_prompt = "Browser test"
        mock_executor.tools = [browser_tool]
        mock_executor.provider = None

        test = TestObject(
            id="test-browser-missing-playwright",
            reason="Test browser graceful skip",
            executor=mock_executor,
            judges=[],
            browser_config=BrowserConfig(headless=True),
            files=[],
        )

        with patch("dokumen.mcp_client.PlaywrightMCPClient.start", new_callable=AsyncMock,
                   side_effect=FileNotFoundError("Playwright MCP CLI not found at /fake/path")):
            with patch("dokumen.mcp_client.PlaywrightMCPClient.stop", new_callable=AsyncMock):
                result = await test.run()

        assert result.passed is False
        assert any("Playwright" in r for r in result.failure_reasons)

    @pytest.mark.asyncio
    async def test_missing_playwright_does_not_crash_suite(self):
        """Browser test with missing Playwright doesn't raise exception."""
        from dokumen.test_object import TestObject, BrowserConfig

        browser_tool = self._make_browser_tool()

        mock_executor = MagicMock()
        mock_executor.user_prompt = "Navigate to example.com"
        mock_executor.system_prompt = "Browser test"
        mock_executor.tools = [browser_tool]
        mock_executor.provider = None

        test = TestObject(
            id="test-browser-no-crash",
            reason="Test no crash",
            executor=mock_executor,
            judges=[],
            browser_config=BrowserConfig(headless=True),
            files=[],
        )

        with patch("dokumen.mcp_client.PlaywrightMCPClient.start", new_callable=AsyncMock,
                   side_effect=FileNotFoundError("Playwright MCP CLI not found")):
            with patch("dokumen.mcp_client.PlaywrightMCPClient.stop", new_callable=AsyncMock):
                # Should NOT raise — returns a result instead
                result = await test.run()

        assert result is not None
        assert result.passed is False


class TestCollectOutputArtifacts:
    """Tests for collect_output_artifacts image support."""

    def test_maps_png_to_image_mime_type(self):
        """PNG files should get image/png content type."""
        from dokumen.test_object import collect_output_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            # collect_output_artifacts expects a sub-directory as output_dir
            output_dir = os.path.join(tmpdir, "test-output")
            os.makedirs(output_dir)
            # Create a tiny PNG-like file
            png_path = os.path.join(output_dir, "screenshot.png")
            with open(png_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

            artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["filename"] == "screenshot.png"
        assert artifacts[0]["content_type"] == "image/png"

    def test_maps_jpg_to_image_mime_type(self):
        """JPG files should get image/jpeg content type."""
        from dokumen.test_object import collect_output_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test-output")
            os.makedirs(output_dir)
            jpg_path = os.path.join(output_dir, "photo.jpg")
            with open(jpg_path, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

            artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["content_type"] == "image/jpeg"

    def test_inlines_small_images_as_base64(self):
        """Small image files should have their content inlined as base64."""
        from dokumen.test_object import collect_output_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test-output")
            os.makedirs(output_dir)
            png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            png_path = os.path.join(output_dir, "small.png")
            with open(png_path, "wb") as f:
                f.write(png_bytes)

            artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        # Content should be base64-encoded
        assert artifacts[0]["content"] is not None
        decoded = base64.b64decode(artifacts[0]["content"])
        assert decoded == png_bytes

    def test_does_not_inline_large_images(self):
        """Images over 100KB should not be inlined."""
        from dokumen.test_object import collect_output_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test-output")
            os.makedirs(output_dir)
            # Create a file larger than 100KB
            large_path = os.path.join(output_dir, "big.png")
            with open(large_path, "wb") as f:
                f.write(b"\x89PNG" + b"\x00" * (101 * 1024))

            artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["content"] is None

    def test_text_files_still_inlined_as_text(self):
        """Text files should still be inlined as text, not base64."""
        from dokumen.test_object import collect_output_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test-output")
            os.makedirs(output_dir)
            txt_path = os.path.join(output_dir, "notes.txt")
            with open(txt_path, "w") as f:
                f.write("Hello, world!")

            artifacts = collect_output_artifacts(output_dir)

        assert len(artifacts) == 1
        assert artifacts[0]["content"] == "Hello, world!"
        assert artifacts[0]["content_type"] == "text/plain"

    def test_empty_directory_returns_empty_list(self):
        """Empty output directory returns no artifacts."""
        from dokumen.test_object import collect_output_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test-output")
            os.makedirs(output_dir)
            artifacts = collect_output_artifacts(output_dir)

        assert artifacts == []

    def test_nonexistent_directory_returns_empty_list(self):
        """Non-existent directory returns no artifacts."""
        from dokumen.test_object import collect_output_artifacts

        artifacts = collect_output_artifacts("/nonexistent/path/output")
        assert artifacts == []


class TestJudgeAggregationLogging:
    """Tests for judge aggregation summary logging."""

    @pytest.mark.asyncio
    async def test_judge_aggregation_logged(self):
        """Judge aggregation summary logs passed/failed counts."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput, JudgeResult

        mock_executor = MagicMock()
        mock_executor.id = "exec-1"
        mock_executor.system_prompt = "Test"
        mock_executor.user_prompt = "Do something"
        mock_executor.tools = []

        mock_output = ExecutorOutput(tool_calls=[], final_response="Done", success=True)
        mock_executor.run = AsyncMock(return_value=mock_output)

        mock_judge1 = MagicMock()
        mock_judge1.id = "judge-pass"
        mock_judge1.tools = None
        mock_judge1.run = AsyncMock(return_value=JudgeResult(
            judge_id="judge-pass", passed=True
        ))

        mock_judge2 = MagicMock()
        mock_judge2.id = "judge-fail"
        mock_judge2.tools = None
        mock_judge2.run = AsyncMock(return_value=JudgeResult(
            judge_id="judge-fail", passed=False, failure_reason="Bad"
        ))

        test = TestObject(
            id="test-agg",
            reason="Test aggregation",
            executor=mock_executor,
            judges=[mock_judge1, mock_judge2],
        )

        # Judge logging now happens in stages.judge module
        with patch('dokumen.stages.judge.logger') as mock_logger:
            result = await test.run()

        agg_calls = [c for c in mock_logger.info.call_args_list
                     if c[0][0] == "stage.judge.summary"]
        assert len(agg_calls) == 1
        kwargs = agg_calls[0][1]
        assert kwargs['test_id'] == "test-agg"
        assert kwargs['total'] == 2
        assert kwargs['passed'] == 1
        assert kwargs['failed'] == 1

    @pytest.mark.asyncio
    async def test_executor_attempt_timing_logged(self):
        """Executor attempt logs duration_ms and success status."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput, JudgeResult

        mock_executor = MagicMock()
        mock_executor.id = "exec-1"
        mock_executor.system_prompt = "Test"
        mock_executor.user_prompt = "Do something"
        mock_executor.tools = []

        mock_output = ExecutorOutput(tool_calls=[], final_response="Done", success=True)
        mock_executor.run = AsyncMock(return_value=mock_output)

        mock_judge = MagicMock()
        mock_judge.id = "judge-1"
        mock_judge.tools = None
        mock_judge.run = AsyncMock(return_value=JudgeResult(
            judge_id="judge-1", passed=True
        ))

        test = TestObject(
            id="test-timing",
            reason="Test timing",
            executor=mock_executor,
            judges=[mock_judge],
        )

        # Executor logging now happens in stages.executor module
        with patch('dokumen.stages.executor.logger') as mock_logger:
            await test.run()

        attempt_calls = [c for c in mock_logger.info.call_args_list
                         if c[0][0] == "stage.executor.attempt.complete"]
        assert len(attempt_calls) == 1
        kwargs = attempt_calls[0][1]
        assert kwargs['test_id'] == "test-timing"
        assert kwargs['attempt'] == 1
        assert 'duration_ms' in kwargs
        assert kwargs['success'] is True

    @pytest.mark.asyncio
    async def test_judge_error_includes_error_type(self):
        """Judge exception log includes error_type field."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput

        mock_executor = MagicMock()
        mock_executor.id = "exec-1"
        mock_executor.system_prompt = "Test"
        mock_executor.user_prompt = "Do something"
        mock_executor.tools = []

        mock_output = ExecutorOutput(tool_calls=[], final_response="Done", success=True)
        mock_executor.run = AsyncMock(return_value=mock_output)

        mock_judge = MagicMock()
        mock_judge.id = "judge-boom"
        mock_judge.tools = None
        mock_judge.run = AsyncMock(side_effect=RuntimeError("Judge exploded"))
        mock_judge._get_assertion_text = MagicMock(return_value="assertion")

        test = TestObject(
            id="test-err",
            reason="Test error type",
            executor=mock_executor,
            judges=[mock_judge],
        )

        # Judge logging now happens in stages.judge module
        with patch('dokumen.stages.judge.logger') as mock_logger:
            result = await test.run()

        error_calls = [c for c in mock_logger.error.call_args_list
                       if c[0][0] == "stage.judge.error"]
        assert len(error_calls) == 1
        kwargs = error_calls[0][1]
        assert kwargs['error_type'] == "RuntimeError"
        assert kwargs['test_id'] == "test-err"

    @pytest.mark.asyncio
    async def test_executor_exception_path_logs_timing(self):
        """Executor exception path logs attempt.complete with timing before error."""
        from dokumen.test_object import TestObject

        mock_executor = MagicMock()
        mock_executor.id = "exec-1"
        mock_executor.system_prompt = "Test"
        mock_executor.user_prompt = "Do something"
        mock_executor.tools = []
        mock_executor.run = AsyncMock(side_effect=RuntimeError("Boom"))

        mock_judge = MagicMock()
        mock_judge.id = "judge-1"
        mock_judge.tools = None

        test = TestObject(
            id="test-exc-timing",
            reason="Test exception timing",
            executor=mock_executor,
            judges=[mock_judge],
        )

        # Executor logging now happens in stages.executor module
        with patch('dokumen.stages.executor.logger') as mock_logger:
            result = await test.run()

        # Should have attempt.complete log even on exception
        attempt_calls = [c for c in mock_logger.info.call_args_list
                         if c[0][0] == "stage.executor.attempt.complete"]
        assert len(attempt_calls) == 1
        kwargs = attempt_calls[0][1]
        assert kwargs['success'] is False
        assert 'duration_ms' in kwargs

        # Should also have the error log with error_type and exc_info
        error_calls = [c for c in mock_logger.error.call_args_list
                       if c[0][0] == "stage.executor.error"]
        assert len(error_calls) == 1
        assert error_calls[0][1]['error_type'] == "RuntimeError"
        assert error_calls[0][1].get('exc_info') is True

    @pytest.mark.asyncio
    async def test_judge_error_includes_exc_info(self):
        """Judge exception log includes exc_info=True for traceback."""
        from dokumen.test_object import TestObject
        from dokumen.agent_object import ExecutorOutput

        mock_executor = MagicMock()
        mock_executor.id = "exec-1"
        mock_executor.system_prompt = "Test"
        mock_executor.user_prompt = "Do something"
        mock_executor.tools = []

        mock_output = ExecutorOutput(tool_calls=[], final_response="Done", success=True)
        mock_executor.run = AsyncMock(return_value=mock_output)

        mock_judge = MagicMock()
        mock_judge.id = "judge-exc"
        mock_judge.tools = None
        mock_judge.run = AsyncMock(side_effect=ValueError("Judge failed"))
        mock_judge._get_assertion_text = MagicMock(return_value="assertion")

        test = TestObject(
            id="test-exc-info",
            reason="Test exc_info",
            executor=mock_executor,
            judges=[mock_judge],
        )

        # Judge logging now happens in stages.judge module
        with patch('dokumen.stages.judge.logger') as mock_logger:
            await test.run()

        error_calls = [c for c in mock_logger.error.call_args_list
                       if c[0][0] == "stage.judge.error"]
        assert len(error_calls) == 1
        assert error_calls[0][1].get('exc_info') is True
