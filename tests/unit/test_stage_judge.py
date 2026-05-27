"""Tests for JudgeStage."""

import os
import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_context(**overrides):
    """Create a minimal PipelineContext for testing."""
    from dokumen.pipeline import PipelineContext

    executor = MagicMock()
    executor.user_prompt = "Test prompt"
    executor.system_prompt = "System prompt"
    executor.tools = []

    defaults = dict(
        test_id="test-judge",
        reason="Test judge",
        executor=executor,
        judges=[],
        files=[],
        timeout=60.0,
        retries=0,
        output_dir="/tmp/test-output",
        executor_output=MagicMock(success=True, final_response="Done"),
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


def _make_judge(judge_id="accuracy", passed=True, failure_reason=None,
                response=None, conversation_log=None, error=None):
    """Create a mock judge agent and its expected result."""
    from dokumen.sdk.types import JudgeVerdict

    judge = MagicMock()
    judge.id = judge_id
    judge.system_prompt = f"System prompt for {judge_id}"
    judge.user_prompt = None
    judge._get_assertion_text.return_value = f"Assert {judge_id}"

    result = JudgeVerdict(
        judge_id=judge_id,
        passed=passed,
        failure_reason=failure_reason,
        response=response,
        conversation_log=conversation_log or [],
        input_tokens=50,
        output_tokens=25,
        error=error,
    )
    judge.run = AsyncMock(return_value=result)

    return judge, result


class TestJudgeStage:
    """Tests for JudgeStage."""

    def test_name(self):
        """Stage name is 'judge'."""
        from dokumen.stages.judge import JudgeStage
        assert JudgeStage().name == "judge"

    @pytest.mark.asyncio
    async def test_all_judges_pass(self):
        """All judges pass — no failure reasons added."""
        from dokumen.stages.judge import JudgeStage

        j1, _ = _make_judge("accuracy", passed=True)
        j2, _ = _make_judge("completeness", passed=True)

        ctx = _make_context(judges=[j1, j2])
        stage = JudgeStage()
        result = await stage.run(ctx)

        assert result.failed is False
        assert len(result.judge_results) == 2
        assert all(jr.passed for jr in result.judge_results)

    @pytest.mark.asyncio
    async def test_some_judges_fail(self):
        """Failed judges add failure reasons to context."""
        from dokumen.stages.judge import JudgeStage

        j1, _ = _make_judge("accuracy", passed=True)
        j2, _ = _make_judge("completeness", passed=False,
                            failure_reason="Missing sections")

        ctx = _make_context(judges=[j1, j2])
        stage = JudgeStage()
        result = await stage.run(ctx)

        # Note: judge stage does NOT set ctx.failed — that's done by
        # TestObject when assembling the final result
        assert len(result.judge_results) == 2
        assert any("completeness" in r for r in result.failure_reasons)

    @pytest.mark.asyncio
    async def test_judge_exception_handled(self):
        """Judge exception is caught and returned as failed verdict."""
        from dokumen.stages.judge import JudgeStage

        judge = MagicMock()
        judge.id = "broken"
        judge._get_assertion_text.return_value = "Assert broken"
        judge.run = AsyncMock(side_effect=RuntimeError("judge exploded"))

        ctx = _make_context(judges=[judge])
        stage = JudgeStage()
        result = await stage.run(ctx)

        assert len(result.judge_results) == 1
        assert result.judge_results[0].passed is False
        assert "judge exploded" in result.judge_results[0].failure_reason

    @pytest.mark.asyncio
    async def test_fires_on_judge_complete_callback(self):
        """on_judge_complete is called for each judge."""
        from dokumen.stages.judge import JudgeStage

        j1, _ = _make_judge("accuracy", passed=True)
        j2, _ = _make_judge("completeness", passed=True)
        callback = MagicMock()

        ctx = _make_context(judges=[j1, j2])
        ctx.on_judge_complete = callback

        stage = JudgeStage()
        await stage.run(ctx)

        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_research_report_extraction(self):
        """Research report is extracted from verdict judge."""
        from dokumen.stages.judge import JudgeStage

        response = (
            "# Research Report\n\nThis is the report content.\n\n"
            '```json\n{"verdict": "PASS", "confidence": 0.9, '
            '"reason": "Good"}\n```'
        )
        j1, _ = _make_judge("verdict", passed=True, response=response)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output", "test-judge")
            os.makedirs(output_dir, exist_ok=True)

            ctx = _make_context(judges=[j1], test_type="research")
            # Point the output dir to our temp dir so report.md is written there
            ctx.output_dir = output_dir

            # Patch os.path.join in the judge module to redirect .dokumen-cache
            # writes to our temp dir
            original_join = os.path.join

            def patched_join(*args):
                if args[:3] == (".dokumen-cache", "output", "test-judge"):
                    return original_join(output_dir, *args[3:])
                return original_join(*args)

            with patch("dokumen.stages.judge.os.path.join", side_effect=patched_join):
                stage = JudgeStage()
                result = await stage.run(ctx)

            assert result.research_report_rel_path == "report.md"

    @pytest.mark.asyncio
    async def test_no_research_report_for_non_research(self):
        """No report extraction for non-research tests."""
        from dokumen.stages.judge import JudgeStage

        j1, _ = _make_judge("accuracy", passed=True, response="Some response")
        ctx = _make_context(judges=[j1], test_type=None)

        stage = JudgeStage()
        result = await stage.run(ctx)

        assert result.research_report_rel_path is None


class TestExtractReportMarkdown:
    """Tests for _extract_report_markdown helper."""

    def test_extracts_from_json_fence(self):
        """Extracts report before ```json verdict block."""
        from dokumen.stages.judge import _extract_report_markdown

        response = (
            "# Report\n\nContent here.\n\n"
            '```json\n{"verdict": "PASS", "confidence": 0.9}\n```'
        )
        result = _extract_report_markdown(response)
        assert result == "# Report\n\nContent here."

    def test_extracts_from_inline_json(self):
        """Extracts report before inline JSON verdict."""
        from dokumen.stages.judge import _extract_report_markdown

        response = (
            "# Report\n\nContent here.\n\n"
            '{"verdict": "FAIL", "confidence": 0.5, "reason": "bad"}'
        )
        result = _extract_report_markdown(response)
        assert result == "# Report\n\nContent here."

    def test_returns_full_when_no_json(self):
        """Returns full response when no JSON found."""
        from dokumen.stages.judge import _extract_report_markdown

        response = "# Report without verdict"
        result = _extract_report_markdown(response)
        assert result == response

    def test_empty_response(self):
        """Returns empty string for None/empty response."""
        from dokumen.stages.judge import _extract_report_markdown

        assert _extract_report_markdown(None) == ""
        assert _extract_report_markdown("") == ""
