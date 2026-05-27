"""Tests for TestObject._build_result — assembles TestResult from PipelineContext."""

import time

import pytest
from unittest.mock import MagicMock
from datetime import datetime


def _make_executor(model="claude-sonnet-4-5-20250929"):
    """Create a mock executor with provider."""
    executor = MagicMock()
    executor.provider = MagicMock()
    executor.provider.model = model
    executor.system_prompt = "Executor system prompt"
    executor.user_prompt = "Executor user prompt"
    executor.tools = [MagicMock(name="read_file"), MagicMock(name="glob")]
    # Ensure .name returns actual string, not MagicMock
    for i, name in enumerate(["read_file", "glob"]):
        executor.tools[i].name = name
    return executor


def _make_judge(judge_id="accuracy", model="claude-haiku-4-5-20251001"):
    """Create a mock judge with provider."""
    judge = MagicMock()
    judge.id = judge_id
    judge.provider = MagicMock()
    judge.provider.model = model
    judge.system_prompt = f"Judge {judge_id} system prompt"
    judge.user_prompt = None
    judge.tools = None
    return judge


def _make_pipeline_context(**overrides):
    """Create a PipelineContext for testing _build_result."""
    from dokumen.pipeline import PipelineContext

    executor = overrides.pop("executor", _make_executor())
    judges = overrides.pop("judges", [_make_judge()])

    defaults = dict(
        test_id="test-build-result",
        reason="Test build result",
        executor=executor,
        judges=judges,
        files=["docs/api.md"],
        timeout=60.0,
        retries=0,
        output_dir="/tmp/test-output",
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


def _make_test_object(**overrides):
    """Create a TestObject for testing _build_result."""
    from dokumen.test_object import TestObject

    executor = overrides.pop("executor", _make_executor())
    judges = overrides.pop("judges", [_make_judge()])

    defaults = dict(
        id="test-build-result",
        reason="Test build result",
        executor=executor,
        judges=judges,
        timeout=60.0,
        retries=0,
        files=["docs/api.md"],
        source_path="/path/to/test.yaml",
    )
    defaults.update(overrides)
    return TestObject(**defaults)


def _make_executor_output(success=True, input_tokens=200, output_tokens=100,
                           conversation_log=None, error=None):
    """Create a mock ExecutorResult."""
    output = MagicMock()
    output.success = success
    output.input_tokens = input_tokens
    output.output_tokens = output_tokens
    output.cache_creation_tokens = 10
    output.cache_read_tokens = 5
    output.conversation_log = conversation_log or []
    output.final_response = "Executor done"
    output.error = error
    return output


def _make_judge_verdict(judge_id="accuracy", passed=True, failure_reason=None,
                         response=None, conversation_log=None, error=None,
                         input_tokens=50, output_tokens=25):
    """Create a JudgeVerdict."""
    from dokumen.sdk.types import JudgeVerdict
    return JudgeVerdict(
        judge_id=judge_id,
        passed=passed,
        failure_reason=failure_reason,
        response=response,
        conversation_log=conversation_log or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=3,
        cache_read_tokens=2,
        error=error,
    )


class TestBuildResult:
    """Tests for TestObject._build_result method."""

    def test_build_result_exists(self):
        """_build_result method exists on TestObject."""
        test_obj = _make_test_object()
        assert hasattr(test_obj, "_build_result")

    def test_build_result_returns_test_result(self):
        """_build_result returns a TestResult instance."""
        from dokumen.test_object import TestResult

        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert isinstance(result, TestResult)

    def test_build_result_passed_when_executor_and_judges_pass(self):
        """Test passes when executor succeeds and all judges pass."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output(success=True)
        ctx.judge_results = [_make_judge_verdict(passed=True)]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.passed is True
        assert result.executor_passed is True
        assert result.status == "passed"

    def test_build_result_failed_when_pipeline_failed(self):
        """Test fails when pipeline context is marked as failed."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.failed = True
        ctx.failure_reasons = ["Explore failed"]
        ctx.executor_output = None

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.passed is False
        assert "Explore failed" in result.failure_reasons

    def test_build_result_failed_when_judge_fails(self):
        """Test fails when a judge fails (even if executor passed)."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output(success=True)
        ctx.judge_results = [_make_judge_verdict(passed=False, failure_reason="Bad")]
        ctx.failure_reasons = ["Judge accuracy failed: Bad"]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.passed is False

    def test_build_result_error_status_on_judge_error(self):
        """Status is 'error' when a judge has error=True."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output(success=True)
        ctx.judge_results = [
            _make_judge_verdict(passed=False, error=True, failure_reason="timeout")
        ]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.passed is False
        assert result.status == "error"

    def test_build_result_executor_model_from_provider(self):
        """Executor model is extracted from provider."""
        executor = _make_executor(model="claude-opus-4-20250514")
        test_obj = _make_test_object(executor=executor)
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.executor_model == "claude-opus-4-20250514"

    def test_build_result_judge_models_from_providers(self):
        """Per-judge models are extracted from providers."""
        j1 = _make_judge("accuracy", model="claude-haiku-4-5-20251001")
        j2 = _make_judge("completeness", model="claude-sonnet-4-5-20250929")
        test_obj = _make_test_object(judges=[j1, j2])
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [
            _make_judge_verdict("accuracy"),
            _make_judge_verdict("completeness"),
        ]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.judge_models == {
            "accuracy": "claude-haiku-4-5-20251001",
            "completeness": "claude-sonnet-4-5-20250929",
        }
        # Backward compat: first judge model
        assert result.judge_model == "claude-haiku-4-5-20251001"

    def test_build_result_executor_token_usage(self):
        """Executor token usage is collected from executor_output."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output(
            input_tokens=500, output_tokens=200
        )
        ctx.judge_results = [_make_judge_verdict()]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.executor_input_tokens == 500
        assert result.executor_output_tokens == 200
        assert result.executor_cache_creation_tokens == 10
        assert result.executor_cache_read_tokens == 5

    def test_build_result_judge_token_aggregation(self):
        """Judge tokens are aggregated across all judges."""
        j1 = _make_judge("accuracy")
        j2 = _make_judge("completeness")
        test_obj = _make_test_object(judges=[j1, j2])
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [
            _make_judge_verdict("accuracy", input_tokens=50, output_tokens=25),
            _make_judge_verdict("completeness", input_tokens=60, output_tokens=30),
        ]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.judge_input_tokens == 110
        assert result.judge_output_tokens == 55
        assert result.judge_cache_creation_tokens == 6  # 3+3
        assert result.judge_cache_read_tokens == 4  # 2+2

    def test_build_result_explore_tokens_from_context(self):
        """Explore token usage comes from PipelineContext."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]
        ctx.explore_input_tokens = 150
        ctx.explore_output_tokens = 75
        ctx.explore_cache_creation_tokens = 8
        ctx.explore_cache_read_tokens = 4
        ctx.explore_model = "claude-haiku-4-5-20251001"
        ctx.explore_status = "pass"

        # Provide explore result mock
        explore_result = MagicMock()
        explore_result.summary = "Found files"
        explore_result.tool_history = [{"tool": "glob"}]
        ctx.explore_result = explore_result

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.explore_input_tokens == 150
        assert result.explore_output_tokens == 75
        assert result.explore_model == "claude-haiku-4-5-20251001"
        assert result.explore_status == "pass"
        assert result.explore_output == "Found files"
        assert result.explore_tool_calls == [{"tool": "glob"}]

    def test_build_result_executor_tools_list(self):
        """executor_tools field contains tool names."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.executor_tools == ["read_file", "glob"]

    def test_build_result_source_path(self):
        """source_path is propagated to TestResult."""
        test_obj = _make_test_object(source_path="/test/path.yaml")
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.source_path == "/test/path.yaml"

    def test_build_result_duration(self):
        """Duration is computed from start_time."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        start_time = time.time() - 2.5
        result = test_obj._build_result(ctx, start_time, None, {})
        assert result.duration >= 2.0  # At least 2 seconds

    def test_build_result_judge_prompts(self):
        """Judge prompts are extracted for UI display."""
        j1 = _make_judge("accuracy")
        test_obj = _make_test_object(judges=[j1])
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.judge_prompts is not None
        assert len(result.judge_prompts) == 1
        assert result.judge_prompts[0]["name"] == "accuracy"

    def test_build_result_executor_conversation_log(self):
        """Executor conversation log is stored."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        conv_log = [{"role": "assistant", "content": "Hello"}]
        ctx.executor_output = _make_executor_output(conversation_log=conv_log)
        ctx.judge_results = [_make_judge_verdict()]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.executor_conversation_log == conv_log

    def test_build_result_judge_conversation_logs(self):
        """Judge conversation logs are collected."""
        j1 = _make_judge("accuracy")
        test_obj = _make_test_object(judges=[j1])
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        judge_conv = [{"role": "assistant", "content": "Verdict"}]
        ctx.judge_results = [
            _make_judge_verdict("accuracy", conversation_log=judge_conv)
        ]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.judge_conversation_logs is not None
        assert len(result.judge_conversation_logs) == 1
        assert result.judge_conversation_logs[0]["judge_name"] == "accuracy"

    def test_build_result_output_artifacts_from_context(self):
        """Output artifacts are extracted from pipeline context."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        artifacts = [
            {"filename": "analysis.py", "path": "analysis.py",
             "size_bytes": 100, "content_type": "text/x-python",
             "content": "print('hi')", "source": "output"},
            {"filename": "video.webm", "path": "recordings/video.webm",
             "size_bytes": 5000, "content_type": "video/webm",
             "content": None, "source": "browser"},
        ]
        ctx.output_artifacts = artifacts

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.output_artifacts == artifacts
        # Legacy browser_artifacts derived
        assert result.browser_artifacts is not None
        assert len(result.browser_artifacts) == 1
        assert result.browser_artifacts[0]["type"] == "video"

    def test_build_result_no_output_artifacts(self):
        """No output artifacts when output_artifacts is empty."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.output_artifacts is None

    def test_build_result_report_artifacts_derived(self):
        """Report artifacts are derived from output artifacts with source=report."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        artifacts = [
            {"filename": "report.md", "path": "report.md",
             "size_bytes": 500, "content_type": "text/markdown",
             "content": "# Report", "source": "report"},
        ]
        ctx.output_artifacts = artifacts

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.report_artifacts is not None
        assert len(result.report_artifacts) == 1
        assert result.report_artifacts[0]["type"] == "report"
        assert result.report_artifacts[0]["content"] == "# Report"

    def test_build_result_files_propagated(self):
        """Files list is propagated from test object."""
        test_obj = _make_test_object(files=["docs/a.md", "docs/b.md"])
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
            files=["docs/a.md", "docs/b.md"],
        )
        ctx.executor_output = _make_executor_output()
        ctx.judge_results = [_make_judge_verdict()]

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.files == ["docs/a.md", "docs/b.md"]

    def test_build_result_no_executor_output(self):
        """Handles None executor output (pipeline failed before executor)."""
        test_obj = _make_test_object()
        ctx = _make_pipeline_context(
            executor=test_obj.executor,
            judges=test_obj.judges,
        )
        ctx.failed = True
        ctx.failure_reasons = ["Browser setup failed"]
        ctx.executor_output = None
        ctx.judge_results = []

        result = test_obj._build_result(ctx, time.time() - 1.0, None, {})
        assert result.passed is False
        assert result.executor_passed is False
        assert result.executor_output is None
        assert result.executor_input_tokens == 0
