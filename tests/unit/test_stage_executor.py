"""Tests for ExecutorStage."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_context(**overrides):
    """Create a minimal PipelineContext for testing."""
    from dokumen.pipeline import PipelineContext

    executor = MagicMock()
    executor.user_prompt = "Test prompt"
    executor.system_prompt = "System prompt"
    executor.tools = []

    judge = MagicMock()
    judge.id = "accuracy"
    judge.system_prompt = "Judge system prompt"

    defaults = dict(
        test_id="test-executor",
        reason="Test executor",
        executor=executor,
        judges=[judge],
        files=[],
        timeout=60.0,
        retries=0,
        output_dir="/tmp/test-output",
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


def _make_executor_output(success=True, error=None, final_response="Done",
                           input_tokens=200, output_tokens=100,
                           conversation_log=None):
    """Create a mock ExecutorResult."""
    output = MagicMock()
    output.success = success
    output.error = error
    output.final_response = final_response
    output.input_tokens = input_tokens
    output.output_tokens = output_tokens
    output.cache_creation_tokens = 0
    output.cache_read_tokens = 0
    output.conversation_log = conversation_log or []
    return output


class TestExecutorStage:
    """Tests for ExecutorStage."""

    def test_name(self):
        """Stage name is 'executor'."""
        from dokumen.stages.executor import ExecutorStage
        assert ExecutorStage().name == "executor"

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Executor succeeds on first attempt."""
        from dokumen.stages.executor import ExecutorStage

        executor_output = _make_executor_output(success=True)
        ctx = _make_context()
        ctx.executor.run = AsyncMock(return_value=executor_output)

        stage = ExecutorStage()
        result = await stage.run(ctx)

        assert result.failed is False
        assert result.executor_output is executor_output
        ctx.executor.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_injects_output_folder(self):
        """Executor prompt gets output folder injection."""
        from dokumen.stages.executor import ExecutorStage

        executor_output = _make_executor_output(success=True)
        ctx = _make_context()
        ctx.executor.run = AsyncMock(return_value=executor_output)

        stage = ExecutorStage()
        await stage.run(ctx)

        # Check output folder was injected into executor prompt
        assert "OUTPUT FOLDER: /tmp/test-output" in ctx.executor.user_prompt
        # Check output folder was injected into judge prompts
        assert "OUTPUT FOLDER: /tmp/test-output" in ctx.judges[0].system_prompt

    @pytest.mark.asyncio
    async def test_captures_original_user_prompt(self):
        """Original user prompt is preserved before injection."""
        from dokumen.stages.executor import ExecutorStage

        executor_output = _make_executor_output(success=True)
        ctx = _make_context()
        ctx.executor.user_prompt = "Original prompt"
        ctx.executor.run = AsyncMock(return_value=executor_output)

        stage = ExecutorStage()
        await stage.run(ctx)

        assert ctx.original_user_prompt == "Original prompt"

    @pytest.mark.asyncio
    async def test_preserves_existing_original_prompt(self):
        """If original_user_prompt already set (by explore), don't overwrite."""
        from dokumen.stages.executor import ExecutorStage

        executor_output = _make_executor_output(success=True)
        ctx = _make_context()
        ctx.original_user_prompt = "Explore-modified prompt"
        ctx.executor.run = AsyncMock(return_value=executor_output)

        stage = ExecutorStage()
        await stage.run(ctx)

        assert ctx.original_user_prompt == "Explore-modified prompt"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Executor retries on failure up to max_attempts."""
        from dokumen.stages.executor import ExecutorStage

        fail_output = _make_executor_output(success=False, error="fail 1")
        success_output = _make_executor_output(success=True)

        ctx = _make_context(retries=1)
        ctx.executor.run = AsyncMock(side_effect=[fail_output, success_output])

        stage = ExecutorStage()
        result = await stage.run(ctx)

        assert result.failed is False
        assert result.executor_output is success_output
        assert ctx.executor.run.call_count == 2

    @pytest.mark.asyncio
    async def test_fails_after_all_retries_exhausted(self):
        """Executor fails after all retries exhausted."""
        from dokumen.stages.executor import ExecutorStage

        fail_output = _make_executor_output(success=False, error="persistent error")

        ctx = _make_context(retries=1)
        ctx.executor.run = AsyncMock(return_value=fail_output)

        stage = ExecutorStage()
        result = await stage.run(ctx)

        assert result.failed is True
        assert "Executor failed after 2 attempts" in result.failure_reasons[0]
        assert ctx.executor.run.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        """Executor handles exceptions gracefully."""
        from dokumen.stages.executor import ExecutorStage

        ctx = _make_context()
        ctx.executor.run = AsyncMock(side_effect=RuntimeError("boom"))

        stage = ExecutorStage()
        result = await stage.run(ctx)

        assert result.failed is True
        assert "Executor error: boom" in result.failure_reasons[0]

    @pytest.mark.asyncio
    async def test_exception_retry(self):
        """Executor retries on exception then succeeds."""
        from dokumen.stages.executor import ExecutorStage

        success_output = _make_executor_output(success=True)
        ctx = _make_context(retries=1)
        ctx.executor.run = AsyncMock(
            side_effect=[RuntimeError("transient"), success_output]
        )

        stage = ExecutorStage()
        result = await stage.run(ctx)

        assert result.failed is False
        assert result.executor_output is success_output

    @pytest.mark.asyncio
    async def test_fires_on_executor_complete_callback(self):
        """on_executor_complete callback is called on success."""
        from dokumen.stages.executor import ExecutorStage

        executor_output = _make_executor_output(success=True)
        callback = MagicMock()

        ctx = _make_context()
        ctx.executor.run = AsyncMock(return_value=executor_output)
        ctx.on_executor_complete = callback

        stage = ExecutorStage()
        await stage.run(ctx)

        callback.assert_called_once_with(executor_output)

    @pytest.mark.asyncio
    async def test_no_callback_on_failure(self):
        """on_executor_complete callback is NOT called on failure."""
        from dokumen.stages.executor import ExecutorStage

        fail_output = _make_executor_output(success=False, error="fail")
        callback = MagicMock()

        ctx = _make_context()
        ctx.executor.run = AsyncMock(return_value=fail_output)
        ctx.on_executor_complete = callback

        stage = ExecutorStage()
        await stage.run(ctx)

        callback.assert_not_called()
