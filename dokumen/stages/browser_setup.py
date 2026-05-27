"""Browser setup stage — initializes Playwright MCP client for browser tests."""

import os

from ..logging_config import get_logger
from ..pipeline import PipelineContext, PipelineStage
from ..test_object import (
    BROWSER_TOOL_NAMES,
    BrowserConfig,
    DEFAULT_BROWSER_VIDEO_SIZE,
    DEFAULT_BROWSER_VIEWPORT,
    clear_output_dir,
    resolve_browser_headless,
)

logger = get_logger(__name__)


def _is_browser_tool_name(name: str) -> bool:
    return (
        name in BROWSER_TOOL_NAMES
        or name.startswith("browser_")
        or name.startswith("mcp__dokumen__browser_")
        or name.startswith("mcp__playwright__browser_")
    )


def _uses_browser_tools(ctx: PipelineContext) -> bool:
    """Check if the test uses any browser automation tools."""
    for tool in ctx.executor.tools:
        if _is_browser_tool_name(tool.name):
            return True
    for judge in ctx.judges:
        if judge.tools:
            for tool in judge.tools:
                if _is_browser_tool_name(tool.name):
                    return True
    return False


class BrowserSetupStage(PipelineStage):
    """Initialize Playwright MCP client for browser tests.

    Clears and creates the unified output directory, then starts the
    Playwright MCP client if the test uses browser tools.
    """

    @property
    def name(self) -> str:
        return "browser_setup"

    def _is_sdk_executor(self, ctx: PipelineContext) -> bool:
        """Check if the executor uses the SDK path (manages its own MCP)."""
        try:
            from ..sdk.agent_wrapper import SdkExecutorWrapper
            return isinstance(ctx.executor, SdkExecutorWrapper)
        except ImportError:
            return False

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Set up the browser MCP client if browser tools are used.

        For SDK executors, Playwright MCP is managed by the SDK via
        McpStdioServerConfig — only output directories are created here.
        For legacy executors, the shared PlaywrightMCPClient is started.

        Args:
            ctx: The pipeline context.

        Returns:
            Updated context with mcp_client set (or None if not needed).
        """
        logger.info("stage.browser_setup.start", test_id=ctx.test_id)

        # Clear and create unified output dir BEFORE browser init
        output_dir = os.path.join(".dokumen-cache", "output", ctx.test_id)
        clear_output_dir(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        ctx.output_dir = output_dir

        if not _uses_browser_tools(ctx):
            logger.info("stage.browser_setup.skip", test_id=ctx.test_id,
                        reason="no browser tools used")
            return ctx

        # SDK executors manage Playwright MCP via McpStdioServerConfig —
        # just create the recordings directory for video output.
        if self._is_sdk_executor(ctx):
            recordings_dir = os.path.join(output_dir, "recordings")
            os.makedirs(recordings_dir, exist_ok=True)
            logger.info("stage.browser_setup.sdk_path", test_id=ctx.test_id,
                        reason="SDK manages Playwright MCP directly")
            return ctx

        from ..mcp_client import PlaywrightMCPClient
        from ..playwright_tools import set_shared_mcp_client

        logger.info("stage.browser_setup.init", test_id=ctx.test_id)
        browser_config = ctx.browser_config or BrowserConfig()
        headless = resolve_browser_headless(browser_config.headless)
        save_video = (
            browser_config.save_video
            if browser_config.save_video is not None
            else DEFAULT_BROWSER_VIDEO_SIZE
        )
        viewport_size = (
            browser_config.viewport_size
            if browser_config.viewport_size is not None
            else (save_video or DEFAULT_BROWSER_VIEWPORT)
        )
        recordings_dir = os.path.join(output_dir, "recordings")
        os.makedirs(recordings_dir, exist_ok=True)

        mcp_client = PlaywrightMCPClient(
            headless=headless,
            save_video=save_video,
            viewport_size=viewport_size,
            output_dir=recordings_dir,
        )

        try:
            await mcp_client.start()
        except FileNotFoundError as e:
            logger.error("stage.browser_setup.playwright_not_found",
                         test_id=ctx.test_id, error=str(e))
            ctx.fail(
                f"Playwright not available: {e}. "
                "Browser tests require Playwright MCP to be installed. "
                "Skipping this test."
            )
            return ctx

        set_shared_mcp_client(mcp_client)
        ctx.mcp_client = mcp_client
        logger.info("stage.browser_setup.complete", test_id=ctx.test_id)
        return ctx
