"""Tests for ExploreStage."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional


def _make_context(**overrides):
    """Create a minimal PipelineContext for testing."""
    from dokumen.pipeline import PipelineContext

    defaults = dict(
        test_id="test-explore",
        reason="Test explore",
        executor=MagicMock(user_prompt="Read docs and answer", tools=[]),
        judges=[],
        files=[],
        timeout=60.0,
        retries=0,
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


def _make_explore_result(success=True, files=None, summary="Found files",
                         tool_history=None, input_tokens=100,
                         output_tokens=50, error=None,
                         tool_calls_count=3):
    """Create a mock ExploreResult."""
    result = MagicMock()
    result.success = success
    result.files = files or []
    result.summary = summary
    result.tool_history = tool_history or []
    result.input_tokens = input_tokens
    result.output_tokens = output_tokens
    result.cache_creation_tokens = 0
    result.cache_read_tokens = 0
    result.model = "claude-haiku-4-5-20251001"
    result.error = error
    result.tool_calls_count = tool_calls_count
    result.to_context_block.return_value = "## Explore Context\nFound: docs/api.md"
    return result


class TestExploreStage:
    """Tests for ExploreStage."""

    def test_name(self):
        """Stage name is 'explore'."""
        from dokumen.stages.explore import ExploreStage
        assert ExploreStage().name == "explore"

    @pytest.mark.asyncio
    async def test_skip_when_no_files_and_no_config(self):
        """Stage is a no-op when no files and no explore config."""
        from dokumen.stages.explore import ExploreStage

        ctx = _make_context()
        stage = ExploreStage()
        result = await stage.run(ctx)

        assert result.failed is False
        assert result.explore_result is None

    @pytest.mark.asyncio
    async def test_runs_when_files_specified(self):
        """Stage runs explore when files are specified."""
        from dokumen.stages.explore import ExploreStage

        explore_result = _make_explore_result(
            files=[MagicMock(path="docs/api.md")]
        )

        ctx = _make_context(files=["docs/api.md"])
        stage = ExploreStage()

        with patch.object(stage, "_run_explore", return_value=explore_result):
            result = await stage.run(ctx)

        assert result.failed is False
        assert result.explore_result is not None
        assert result.explore_status == "pass"
        assert result.explore_input_tokens == 100

    @pytest.mark.asyncio
    async def test_runs_when_explore_config_enabled(self):
        """Stage runs explore when explore_config.enabled is True."""
        from dokumen.stages.explore import ExploreStage

        explore_config = MagicMock(enabled=True, model="haiku",
                                    max_files=20, max_iterations=50,
                                    timeout=60)
        explore_result = _make_explore_result()

        ctx = _make_context(explore_config=explore_config)
        stage = ExploreStage()

        with patch.object(stage, "_run_explore", return_value=explore_result):
            result = await stage.run(ctx)

        assert result.failed is False
        assert result.explore_result is not None

    @pytest.mark.asyncio
    async def test_fails_when_explore_returns_none_with_required_files(self):
        """Stage fails when explore returns None but files are required."""
        from dokumen.stages.explore import ExploreStage

        ctx = _make_context(files=["docs/api.md"])
        stage = ExploreStage()

        with patch.object(stage, "_run_explore", return_value=None):
            result = await stage.run(ctx)

        assert result.failed is True
        assert "EXPLORE PHASE FAILED" in result.failure_reasons[0]
        assert result.explore_status == "fail"

    @pytest.mark.asyncio
    async def test_injects_context_into_executor(self):
        """Stage injects explore context into executor prompt."""
        from dokumen.stages.explore import ExploreStage

        explore_result = _make_explore_result(
            files=[MagicMock(path="docs/api.md")],
            summary="Found docs/api.md",
        )

        executor = MagicMock(user_prompt="Original prompt")
        ctx = _make_context(
            files=["docs/api.md"],
            executor=executor,
        )
        stage = ExploreStage()

        with patch.object(stage, "_run_explore", return_value=explore_result):
            result = await stage.run(ctx)

        assert result.failed is False
        # Original prompt should be preserved
        assert result.original_user_prompt == "Original prompt"
        # Executor prompt should now contain context block
        assert "Explore Context" in ctx.executor.user_prompt

    @pytest.mark.asyncio
    async def test_missing_files_deterministic_recovery(self):
        """Stage recovers missing files via filesystem check."""
        from dokumen.stages.explore import ExploreStage

        explore_result = _make_explore_result(files=[])  # no files found

        ctx = _make_context(files=["docs/api.md"])
        stage = ExploreStage()

        with patch.object(stage, "_run_explore", return_value=explore_result):
            with patch.object(stage, "_verify_explore_found_files",
                              return_value=["docs/api.md"]):
                with patch.object(stage, "_check_files_on_disk",
                                  return_value=[]):  # all recovered
                    result = await stage.run(ctx)

        assert result.failed is False

    @pytest.mark.asyncio
    async def test_missing_files_not_on_disk(self):
        """Stage fails when files are truly missing from disk."""
        from dokumen.stages.explore import ExploreStage

        explore_result = _make_explore_result(
            files=[], error=None, tool_calls_count=5
        )

        ctx = _make_context(files=["docs/nonexistent.md"])
        stage = ExploreStage()

        with patch.object(stage, "_run_explore", return_value=explore_result):
            with patch.object(stage, "_verify_explore_found_files",
                              return_value=["docs/nonexistent.md"]):
                with patch.object(stage, "_check_files_on_disk",
                                  return_value=["docs/nonexistent.md"]):
                    result = await stage.run(ctx)

        assert result.failed is True
        assert result.explore_status == "fail"

    def test_verify_explore_found_files_all_found(self):
        """All required files found in explore result."""
        from dokumen.stages.explore import ExploreStage

        explore_result = _make_explore_result(
            files=[MagicMock(path="docs/api.md")],
        )
        ctx = _make_context(files=["docs/api.md"])
        stage = ExploreStage()

        missing = stage._verify_explore_found_files(ctx, explore_result)
        assert missing == []

    def test_verify_explore_found_files_missing(self):
        """Missing files are reported."""
        from dokumen.stages.explore import ExploreStage

        explore_result = _make_explore_result(files=[], summary="")
        ctx = _make_context(files=["docs/missing.md"])
        stage = ExploreStage()

        missing = stage._verify_explore_found_files(ctx, explore_result)
        assert missing == ["docs/missing.md"]

    def test_format_missing_files_error(self):
        """Error message includes diagnostics."""
        from dokumen.stages.explore import ExploreStage

        explore_result = _make_explore_result(
            success=False, error="timeout", tool_calls_count=10
        )
        ctx = _make_context(files=["docs/a.md", "docs/b.md"])
        stage = ExploreStage()

        msg = stage._format_missing_files_error(
            ctx, ["docs/b.md"], explore_result
        )

        assert "EXPLORE PHASE FAILED" in msg
        assert "[FOUND] docs/a.md" in msg
        assert "[MISSING] docs/b.md" in msg
