"""MCP tool integration for CLI executor/judge agents.

Creates ToolDefinition instances backed by InProcessMCPAdapter,
allowing CLI agents to use the same 15 Dokumen MCP tools that
Claude Code clients use — one set of tools, one code path.

Part of Issue #589: Wire Dokumen MCP tools into chat agent and CLI agents.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .logging_config import get_logger
from .tools.types import ToolDefinition, ToolResult

logger = get_logger(__name__)


def create_mcp_tool_definitions(
    gitlab_token: Optional[str] = None,
    gitlab_url: Optional[str] = None,
    project_id: Optional[int] = None,
    branch: str = "main",
) -> List[ToolDefinition]:
    """Create ToolDefinition instances backed by InProcessMCPAdapter.

    Reads credentials from environment variables if not provided.
    Returns empty list if required credentials are missing (graceful
    degradation for local dev without GitLab).

    Args:
        gitlab_token: GitLab PAT. Falls back to GITLAB_SERVICE_TOKEN env var.
        gitlab_url: GitLab URL. Falls back to GITLAB_URL env var.
        project_id: GitLab project ID. Falls back to GITLAB_PROJECT_ID env var.
        branch: Git branch for file operations.

    Returns:
        List of ToolDefinition instances for all 15 MCP tools.
    """
    token = (
        gitlab_token
        or os.environ.get("GITLAB_SERVICE_TOKEN")
        or os.environ.get("GITLAB_TOKEN")
        or ""
    )
    url = gitlab_url or os.environ.get("GITLAB_URL", "")
    pid_str = str(project_id) if project_id else os.environ.get("GITLAB_PROJECT_ID", "")
    mcp_branch = branch or os.environ.get("MCP_BRANCH", "main")

    if not token or not url or not pid_str:
        # In sandbox mode, missing credentials is a real problem — warn loudly.
        # In CLI/local mode, MCP tools are optional — log at debug level (#591).
        is_sandbox = os.environ.get("DOKUMEN_EXECUTION_MODE") == "sandbox"
        log_fn = logger.warning if is_sandbox else logger.debug
        log_fn(
            "mcp_tools.skip",
            reason="Missing GITLAB_SERVICE_TOKEN/GITLAB_TOKEN, GITLAB_URL, or GITLAB_PROJECT_ID",
            has_token=bool(token),
            has_url=bool(url),
            has_project_id=bool(pid_str),
            execution_mode=os.environ.get("DOKUMEN_EXECUTION_MODE", "cli"),
        )
        return []

    try:
        pid = int(pid_str)
    except (ValueError, TypeError):
        logger.error(
            "mcp_tools.invalid_project_id",
            project_id=pid_str,
        )
        return []

    try:
        from backend.dokumen_mcp.in_process import InProcessMCPAdapter
    except ImportError:
        logger.warning(
            "mcp_tools.import_failed",
            reason="backend.dokumen_mcp.in_process not available (expected in CI)",
        )
        return []

    adapter = InProcessMCPAdapter(
        gitlab_token=token,
        gitlab_url=url,
        project_id=pid,
        branch=mcp_branch,
        source_type="pipeline",
    )

    tool_defs = adapter.get_tool_definitions()
    tools: List[ToolDefinition] = []

    for tdef in tool_defs:
        tool_name = tdef["name"]

        # Create a closure-safe handler for each tool
        async def _handler(
            params: Dict[str, Any],
            _adapter: Any = adapter,
            _name: str = tool_name,
        ) -> ToolResult:
            """Execute MCP tool via InProcessMCPAdapter."""
            logger.info(
                "mcp_tool.execute.start",
                tool_name=_name,
                params_keys=list(params.keys()),
            )
            try:
                result = await _adapter.call_tool(_name, params)
            except Exception as e:
                logger.error(
                    "mcp_tool.execute.exception",
                    tool_name=_name,
                    error=str(e),
                )
                return ToolResult(
                    success=False,
                    output="",
                    error=f"MCP tool '{_name}' failed: {e}",
                )

            if not result.get("success", True):
                error_msg = result.get("error", "MCP tool failed")
                logger.warning(
                    "mcp_tool.execute.mcp_error",
                    tool_name=_name,
                    error=error_msg,
                )
                return ToolResult(
                    success=False,
                    output="",
                    error=error_msg,
                )

            logger.info(
                "mcp_tool.execute.complete",
                tool_name=_name,
                result_keys=list(result.keys()),
            )
            return ToolResult(
                success=True,
                output=json.dumps(result, default=str),
            )

        tools.append(
            ToolDefinition(
                name=tool_name,
                description=tdef["description"],
                parameters=tdef["inputSchema"],
                handler=_handler,
            )
        )

    logger.info(
        "mcp_tools.created",
        tool_count=len(tools),
        tool_names=[t.name for t in tools],
        gitlab_url=url,
        project_id=pid,
        branch=mcp_branch,
    )
    return tools
