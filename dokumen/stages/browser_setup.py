"""Browser setup stage for SDK-managed browser tests."""

import os

from ..logging_config import get_logger
from ..pipeline import PipelineContext, PipelineStage
from ..test_object import (
    clear_output_dir,
)
from ..playwright_tools import BROWSER_TOOL_NAMES

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
    """Prepare output directories for browser tests.

    The Claude Agent SDK owns Playwright MCP startup. This stage keeps the
    per-test output layout deterministic so videos and screenshots land in the
    same cache location regardless of whether the test uses browser tools.
    """

    @property
    def name(self) -> str:
        return "browser_setup"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Prepare output directories for this test run.

        For browser tests, Playwright MCP is configured later on the SDK agent
        via ``McpStdioServerConfig``. No in-process MCP client is started here.

        Args:
            ctx: The pipeline context.

        Returns:
            Updated context with output directories populated.
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

        recordings_dir = os.path.join(output_dir, "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        logger.info("stage.browser_setup.sdk_path", test_id=ctx.test_id,
                    reason="SDK manages Playwright MCP directly")
        return ctx
