"""Tests for ArtifactStage."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch


def _make_context(**overrides):
    """Create a minimal PipelineContext for testing."""
    from dokumen.pipeline import PipelineContext

    defaults = dict(
        test_id="test-artifact",
        reason="Test artifacts",
        executor=MagicMock(),
        judges=[],
        files=[],
        timeout=60.0,
        retries=0,
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


class TestArtifactStage:
    """Tests for ArtifactStage."""

    def test_name(self):
        """Stage name is 'artifact'."""
        from dokumen.stages.artifact import ArtifactStage
        assert ArtifactStage().name == "artifact"

    @pytest.mark.asyncio
    async def test_skip_when_no_output_dir(self):
        """Stage skips when output_dir is not set."""
        from dokumen.stages.artifact import ArtifactStage

        ctx = _make_context()
        ctx.output_dir = ""

        stage = ArtifactStage()
        result = await stage.run(ctx)

        assert result.failed is False
        assert result.output_artifacts == []

    @pytest.mark.asyncio
    async def test_collects_text_artifacts(self):
        """Stage collects text files from output directory."""
        from dokumen.stages.artifact import ArtifactStage

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output", "test-artifact")
            os.makedirs(output_dir, exist_ok=True)

            # Write a test file
            with open(os.path.join(output_dir, "analysis.py"), "w") as f:
                f.write("print('hello')")

            ctx = _make_context()
            ctx.output_dir = output_dir

            stage = ArtifactStage()
            result = await stage.run(ctx)

            assert len(result.output_artifacts) == 1
            assert result.output_artifacts[0]["filename"] == "analysis.py"
            assert result.output_artifacts[0]["source"] == "output"

    @pytest.mark.asyncio
    async def test_tags_browser_artifacts(self):
        """Stage tags recordings/ files as browser source."""
        from dokumen.stages.artifact import ArtifactStage

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output", "test-artifact")
            recordings_dir = os.path.join(output_dir, "recordings")
            os.makedirs(recordings_dir, exist_ok=True)

            # Write a dummy video file
            with open(os.path.join(recordings_dir, "video-1.webm"), "wb") as f:
                f.write(b"\x00" * 100)

            ctx = _make_context()
            ctx.output_dir = output_dir

            stage = ArtifactStage()
            result = await stage.run(ctx)

            browser_artifacts = [
                a for a in result.output_artifacts if a["source"] == "browser"
            ]
            assert len(browser_artifacts) == 1
            assert browser_artifacts[0]["path"] == "recordings/video-1.webm"

    @pytest.mark.asyncio
    async def test_tags_report_artifacts(self):
        """Stage tags report files with report source."""
        from dokumen.stages.artifact import ArtifactStage

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output", "test-artifact")
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, "report.md"), "w") as f:
                f.write("# Research Report\n\nContent here.")

            ctx = _make_context()
            ctx.output_dir = output_dir
            ctx.research_report_rel_path = "report.md"

            stage = ArtifactStage()
            result = await stage.run(ctx)

            report_artifacts = [
                a for a in result.output_artifacts if a["source"] == "report"
            ]
            assert len(report_artifacts) == 1
            assert report_artifacts[0]["content"] == "# Research Report\n\nContent here."

    @pytest.mark.asyncio
    async def test_empty_output_dir(self):
        """Stage returns empty list when output dir is empty."""
        from dokumen.stages.artifact import ArtifactStage

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output", "test-artifact")
            os.makedirs(output_dir, exist_ok=True)

            ctx = _make_context()
            ctx.output_dir = output_dir

            stage = ArtifactStage()
            result = await stage.run(ctx)

            assert result.output_artifacts == []
