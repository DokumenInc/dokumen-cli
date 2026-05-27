"""Tests for BrowserSetupStage."""

import os
import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_context(**overrides):
    """Create a minimal PipelineContext for testing."""
    from dokumen.pipeline import PipelineContext

    executor = MagicMock()
    executor.tools = []

    defaults = dict(
        test_id="test-browser",
        reason="Test browser",
        executor=executor,
        judges=[],
        files=[],
        timeout=60.0,
        retries=0,
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


class TestBrowserSetupStage:
    """Tests for BrowserSetupStage."""

    def test_name(self):
        """Stage name is 'browser_setup'."""
        from dokumen.stages.browser_setup import BrowserSetupStage
        assert BrowserSetupStage().name == "browser_setup"

    @pytest.mark.asyncio
    async def test_skip_when_no_browser_tools(self):
        """Stage skips browser init when no browser tools are used."""
        from dokumen.stages.browser_setup import BrowserSetupStage

        ctx = _make_context()
        stage = BrowserSetupStage()

        with patch("dokumen.stages.browser_setup.clear_output_dir"):
            with patch("dokumen.stages.browser_setup.os.makedirs"):
                result = await stage.run(ctx)

        assert result.failed is False
        assert result.mcp_client is None
        assert result.output_dir != ""

    @pytest.mark.asyncio
    async def test_creates_output_dir(self):
        """Stage always creates the output directory."""
        from dokumen.stages.browser_setup import BrowserSetupStage

        ctx = _make_context()
        stage = BrowserSetupStage()

        with patch("dokumen.stages.browser_setup.clear_output_dir"):
            with patch("dokumen.stages.browser_setup.os.makedirs") as mock_makedirs:
                await stage.run(ctx)

        mock_makedirs.assert_called()
        assert ctx.output_dir.endswith("test-browser")

    @pytest.mark.asyncio
    async def test_fails_when_playwright_not_found(self):
        """Stage fails when Playwright is not available."""
        from dokumen.stages.browser_setup import BrowserSetupStage

        # Set up executor with a browser tool
        browser_tool = MagicMock()
        browser_tool.name = "browser_navigate"
        executor = MagicMock()
        executor.tools = [browser_tool]

        ctx = _make_context(executor=executor)
        stage = BrowserSetupStage()

        mock_client = MagicMock()
        mock_client.start = AsyncMock(
            side_effect=FileNotFoundError("playwright not found")
        )

        with patch("dokumen.stages.browser_setup.clear_output_dir"):
            with patch("dokumen.stages.browser_setup.os.makedirs"):
                with patch("dokumen.mcp_client.PlaywrightMCPClient",
                            return_value=mock_client):
                    with patch("dokumen.playwright_tools.set_shared_mcp_client"):
                        result = await stage.run(ctx)

        assert result.failed is True
        assert "Playwright not available" in result.failure_reasons[0]

    @pytest.mark.asyncio
    async def test_starts_browser_when_browser_tools_used(self):
        """Stage starts MCP client when browser tools are used."""
        from dokumen.stages.browser_setup import BrowserSetupStage

        browser_tool = MagicMock()
        browser_tool.name = "browser_navigate"
        executor = MagicMock()
        executor.tools = [browser_tool]

        ctx = _make_context(executor=executor)
        stage = BrowserSetupStage()

        mock_client = MagicMock()
        mock_client.start = AsyncMock()

        with patch("dokumen.stages.browser_setup.clear_output_dir"):
            with patch("dokumen.stages.browser_setup.os.makedirs"):
                with patch("dokumen.mcp_client.PlaywrightMCPClient",
                            return_value=mock_client):
                    with patch("dokumen.playwright_tools.set_shared_mcp_client") as mock_set:
                        result = await stage.run(ctx)

        assert result.failed is False
        assert result.mcp_client is mock_client
        mock_client.start.assert_called_once()
        mock_set.assert_called_once_with(mock_client)


class TestBrowserSetupSdkPath:
    """Tests for SDK executor browser setup path."""

    @pytest.mark.asyncio
    async def test_sdk_executor_skips_mcp_client_startup(self):
        """SDK executors don't start a shared PlaywrightMCPClient."""
        from dokumen.stages.browser_setup import BrowserSetupStage
        from dokumen.sdk.agent_wrapper import SdkExecutorWrapper

        # Create a mock SdkExecutorWrapper
        mock_sdk_executor = MagicMock(spec=SdkExecutorWrapper)
        browser_tool = MagicMock()
        browser_tool.name = "mcp__playwright__browser_navigate"
        mock_sdk_executor.tools = [browser_tool]

        ctx = _make_context(executor=mock_sdk_executor)
        stage = BrowserSetupStage()

        with patch("dokumen.stages.browser_setup.clear_output_dir"):
            with patch("dokumen.stages.browser_setup.os.makedirs") as mock_makedirs:
                result = await stage.run(ctx)

        assert result.failed is False
        assert result.mcp_client is None  # No shared MCP client
        # Recordings dir should still be created
        assert mock_makedirs.call_count >= 2  # output_dir + recordings_dir

    @pytest.mark.asyncio
    async def test_sdk_executor_creates_recordings_dir(self):
        """SDK path still creates recordings directory for video output."""
        from dokumen.stages.browser_setup import BrowserSetupStage
        from dokumen.sdk.agent_wrapper import SdkExecutorWrapper

        mock_sdk_executor = MagicMock(spec=SdkExecutorWrapper)
        browser_tool = MagicMock()
        browser_tool.name = "mcp__playwright__browser_navigate"
        mock_sdk_executor.tools = [browser_tool]

        ctx = _make_context(executor=mock_sdk_executor)
        stage = BrowserSetupStage()

        makedirs_calls = []
        def track_makedirs(path, **kwargs):
            makedirs_calls.append(path)

        with patch("dokumen.stages.browser_setup.clear_output_dir"):
            with patch("dokumen.stages.browser_setup.os.makedirs", side_effect=track_makedirs):
                await stage.run(ctx)

        recordings_calls = [c for c in makedirs_calls if "recordings" in c]
        assert len(recordings_calls) == 1


class TestUsesBrowserTools:
    """Tests for _uses_browser_tools helper."""

    def test_no_browser_tools(self):
        """Returns False when no browser tools used."""
        from dokumen.stages.browser_setup import _uses_browser_tools

        ctx = _make_context()
        assert _uses_browser_tools(ctx) is False

    def test_executor_has_browser_tool(self):
        """Returns True when executor has a browser tool."""
        from dokumen.stages.browser_setup import _uses_browser_tools

        browser_tool = MagicMock()
        browser_tool.name = "browser_click"
        executor = MagicMock()
        executor.tools = [browser_tool]

        ctx = _make_context(executor=executor)
        assert _uses_browser_tools(ctx) is True

    def test_judge_has_browser_tool(self):
        """Returns True when a judge has a browser tool."""
        from dokumen.stages.browser_setup import _uses_browser_tools

        browser_tool = MagicMock()
        browser_tool.name = "browser_screenshot"
        judge = MagicMock()
        judge.tools = [browser_tool]

        ctx = _make_context(judges=[judge])
        assert _uses_browser_tools(ctx) is True

    def test_judge_with_none_tools(self):
        """Handles judges with None tools."""
        from dokumen.stages.browser_setup import _uses_browser_tools

        judge = MagicMock()
        judge.tools = None

        ctx = _make_context(judges=[judge])
        assert _uses_browser_tools(ctx) is False

    def test_recognizes_playwright_mcp_prefix(self):
        """Detects tools with mcp__playwright__ prefix."""
        from dokumen.stages.browser_setup import _is_browser_tool_name

        assert _is_browser_tool_name("mcp__playwright__browser_navigate") is True
        assert _is_browser_tool_name("mcp__playwright__browser_click") is True
        assert _is_browser_tool_name("mcp__dokumen__browser_navigate") is True
        assert _is_browser_tool_name("browser_navigate") is True
        assert _is_browser_tool_name("Read") is False
        assert _is_browser_tool_name("Bash") is False
