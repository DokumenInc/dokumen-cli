"""
Tool mapping: Dokumen tool names -> Claude Code SDK tools.

Splits scaffold tools into three categories:
1. SDK built-ins (Read, Write, Bash, Glob, Grep, WebFetch, WebSearch)
2. Dokumen MCP tools such as read_many_files and explore
3. Playwright MCP tools (browser_*) -- served via external Playwright MCP server
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server, tool

from ..tools.types import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Runtime context needed for delegate_to_agent tool resolution.

    Carries the information that the delegation handler needs to discover
    user-defined agents, resolve their tools, and run them as subagents.

    Attributes:
        user_dirs: Directories to search for user-defined agent YAML files.
        tools_config: Project-level tools configuration from dokumen.yaml.
        base_dir: Project base directory for path resolution.
        timeout: Default subagent timeout in seconds.
        is_subagent: True when this context belongs to a subagent —
                     prevents recursive delegation.
    """

    user_dirs: Optional[List[Path]] = None
    tools_config: Optional[Any] = None
    base_dir: str = "."
    timeout: float = 120.0
    is_subagent: bool = False


# Import BROWSER_TOOLS at module level for patchability in tests.
# Falls back to empty dict if playwright_tools is unavailable.
try:
    from ..playwright_tools import BROWSER_TOOLS
except ImportError:
    BROWSER_TOOLS: Dict[str, Any] = {}
    logger.debug("playwright_tools not available, browser tools disabled")

# SDK built-in mapping (Dokumen tool name -> Claude Code SDK tool name)
SDK_MAPPING: Dict[str, str] = {
    "read_file": "Read",
    "write_file": "Write",
    "glob": "Glob",
    "search_file_content": "Grep",
    "list_directory": "Glob",  # Mapped to Glob with pattern: {dir}/*
    "run_shell_command": "Bash",
    "web_fetch": "WebFetch",
    "web_search": "WebSearch",
}

UNSUPPORTED_SDK_TOOLS: Dict[str, str] = {}

# Prefix for Playwright MCP tools when exposed through the SDK
PLAYWRIGHT_MCP_PREFIX = "mcp__playwright__"


@dataclass
class ResolvedTools:
    """Result of resolving scaffold tools for the SDK path."""

    sdk_tool_names: List[str]  # ["Read", "Glob", "Bash", ...]
    dokumen_mcp_tools: List[ToolDefinition]  # ToolDefinition objects for in-process Dokumen MCP
    playwright_mcp_config: Optional[Any] = None  # Playwright MCP server config
    playwright_tool_names: List[str] = field(
        default_factory=list
    )  # ["mcp__playwright__browser_navigate", ...]


def resolve_sdk_tools(
    scaffold_tools: List[str],
    tools_config: Optional[Dict[str, Any]] = None,
    test_name: Optional[str] = None,
    browser_config: Optional[Dict[str, Any]] = None,
    agent_context: Optional[AgentContext] = None,
    dokumen_tool_definitions: Optional[List[ToolDefinition]] = None,
) -> ResolvedTools:
    """Resolve scaffold tool names into SDK-compatible tool sets.

    For each tool name in scaffold_tools:
    - If in UNSUPPORTED_SDK_TOOLS -> raise ValueError
    - If in SDK_MAPPING -> add SDK name to sdk_tool_names
    - If a browser tool (in BROWSER_TOOLS) -> add to playwright_tool_names
    - Otherwise -> expose a pre-resolved Dokumen ToolDefinition as MCP

    If any browser tools are found, "Read" is auto-injected into sdk_tool_names.

    Args:
        scaffold_tools: List of Dokumen tool names from the test scaffold.
        tools_config: Optional tool configuration from dokumen.yaml.
        dokumen_tool_definitions: ToolDefinition objects already resolved by
            the CLI loader. These are used for Dokumen-specific MCP tools such
            as read_many_files, task tools, and explore.

    Returns:
        ResolvedTools with categorized tool sets.

    Raises:
        ValueError: If a tool is unsupported or unknown.
    """
    logger.info(
        "Resolving SDK tools",
        extra={"tool_count": len(scaffold_tools), "tools": scaffold_tools},
    )

    sdk_names: List[str] = []
    dokumen_tools: List[ToolDefinition] = []
    playwright_names: List[str] = []
    playwright_config = None

    seen_sdk: set = set()
    provided_dokumen_tools = {
        tool_def.name: tool_def for tool_def in (dokumen_tool_definitions or [])
    }

    for tool_name in scaffold_tools:
        # Check unsupported first
        if tool_name in UNSUPPORTED_SDK_TOOLS:
            logger.error(
                "Unsupported tool requested",
                extra={"tool": tool_name, "reason": UNSUPPORTED_SDK_TOOLS[tool_name]},
            )
            raise ValueError(UNSUPPORTED_SDK_TOOLS[tool_name])

        # SDK built-in mapping
        if tool_name in SDK_MAPPING:
            sdk_name = SDK_MAPPING[tool_name]
            if sdk_name not in seen_sdk:
                sdk_names.append(sdk_name)
                seen_sdk.add(sdk_name)
            logger.debug(
                "Mapped to SDK built-in",
                extra={"dokumen_tool": tool_name, "sdk_tool": sdk_name},
            )
            continue

        # Browser / Playwright tools
        if tool_name in BROWSER_TOOLS:
            prefixed = f"{PLAYWRIGHT_MCP_PREFIX}{tool_name}"
            playwright_names.append(prefixed)
            logger.debug(
                "Mapped to Playwright MCP tool",
                extra={"dokumen_tool": tool_name, "mcp_tool": prefixed},
            )
            continue

        # Agent delegation tool — handled via SDK-native Agent tool
        # The actual agent definitions are built separately by
        # build_sdk_agent_definitions() and passed to ClaudeAgentOptions.agents.
        # Here we just skip the tool name so it doesn't hit resolve_dokumen_tool().
        if tool_name == "delegate_to_agent":
            if agent_context is None:
                # No context — fall through to raise ValueError
                resolved = resolve_dokumen_tool(tool_name, tools_config=tools_config)
                dokumen_tools.append(resolved)
                continue
            if agent_context.is_subagent:
                # Subagents cannot delegate (anti-recursion)
                logger.info(
                    "Skipping delegate_to_agent for subagent",
                    extra={"tool": tool_name},
                )
                continue
            # Replace delegate_to_agent with SDK-native Agent tool
            if "Agent" not in seen_sdk:
                sdk_names.append("Agent")
                seen_sdk.add("Agent")
            logger.info(
                "Mapped delegate_to_agent to SDK-native Agent tool",
                extra={"base_dir": agent_context.base_dir},
            )
            continue

        # Dokumen-specific tools (MCP)
        resolved = provided_dokumen_tools.get(tool_name)
        if resolved is None:
            resolved = resolve_dokumen_tool(tool_name, tools_config=tools_config)
        if resolved.name not in {tool.name for tool in dokumen_tools}:
            dokumen_tools.append(resolved)
        logger.debug("Resolved as Dokumen MCP tool", extra={"tool": tool_name})

    # Auto-inject Read for browser tests (agents need to read files)
    if playwright_names and "Read" not in seen_sdk:
        sdk_names.append("Read")
        seen_sdk.add("Read")
        logger.info("Auto-injected Read tool for browser test")

    # Set up Playwright MCP config if browser tools are present
    if playwright_names:
        bc = browser_config or {}
        playwright_config = get_playwright_mcp_config(
            test_name=test_name,
            headless=bc.get("headless"),
            save_video=bc.get("save_video"),
            viewport_size=bc.get("viewport_size"),
        )

    logger.info(
        "SDK tool resolution complete",
        extra={
            "sdk_tools": sdk_names,
            "dokumen_mcp_count": len(dokumen_tools),
            "playwright_tools": playwright_names,
        },
    )

    return ResolvedTools(
        sdk_tool_names=sdk_names,
        dokumen_mcp_tools=dokumen_tools,
        playwright_mcp_config=playwright_config,
        playwright_tool_names=playwright_names,
    )


def resolve_dokumen_tool(
    name: str,
    tools_config: Optional[Dict[str, Any]] = None,
) -> ToolDefinition:
    """Look up a Dokumen-specific tool by name.

    Args:
        name: The tool name to resolve.
        tools_config: Optional tool configuration.

    Returns:
        ToolDefinition for the requested tool.

    Raises:
        ValueError: If the tool name is unknown.
    """
    # TODO: Import from loader.py registries (BUILTIN_TOOLS, STANDALONE_TOOLS,
    # SANDBOX_TOOLS) once circular import issues are resolved. For now, this is
    # a stub that raises ValueError for unknown tools.
    logger.warning(
        "Dokumen tool resolution is a stub",
        extra={"tool": name},
    )
    raise ValueError(
        f"Unknown Dokumen tool: '{name}'. SDK tool resolution for Dokumen-specific "
        f"tools is not yet implemented. Known SDK tools: {list(SDK_MAPPING.keys())}"
    )


def create_dokumen_mcp_server(
    tools: List[ToolDefinition],
    on_tool_call: Optional[Callable] = None,
) -> Any:
    """Create an in-process MCP server config for Dokumen-specific tools.

    Wraps each ToolDefinition's async handler as an SdkMcpTool and bundles
    them into an McpSdkServerConfig via create_sdk_mcp_server().

    Args:
        tools: List of ToolDefinition objects to expose as MCP tools.
        on_tool_call: Optional callback invoked on each tool call with
                      (tool_name, params, result) for logging/hooks.

    Returns:
        McpSdkServerConfig for use with ClaudeAgentOptions.mcp_servers.
    """
    logger.info(
        "Creating Dokumen MCP server",
        extra={"tool_count": len(tools), "tool_names": [t.name for t in tools]},
    )

    sdk_tools: List[SdkMcpTool] = []

    for tool_def in tools:
        sdk_tool = _wrap_tool_definition(tool_def, on_tool_call)
        sdk_tools.append(sdk_tool)

    config = create_sdk_mcp_server(
        name="dokumen-tools",
        version="1.0.0",
        tools=sdk_tools,
    )

    logger.info(
        "Dokumen MCP server created",
        extra={"server_name": "dokumen-tools", "tool_count": len(sdk_tools)},
    )

    return config


def _wrap_tool_definition(
    tool_def: ToolDefinition,
    on_tool_call: Optional[Callable] = None,
) -> SdkMcpTool:
    """Wrap a ToolDefinition into an SdkMcpTool.

    Args:
        tool_def: The Dokumen ToolDefinition to wrap.
        on_tool_call: Optional callback for tool call notifications.

    Returns:
        SdkMcpTool compatible with the SDK MCP server.
    """

    @tool(
        name=tool_def.name,
        description=tool_def.description,
        input_schema=tool_def.parameters,
    )
    async def handler(params: dict) -> dict:
        logger.info(
            "Dokumen MCP tool invoked",
            extra={"tool": tool_def.name, "params": params},
        )

        result = await tool_def.handler(params)

        if on_tool_call is not None:
            try:
                on_tool_call(tool_def.name, params, result)
            except Exception as e:
                logger.warning(
                    "on_tool_call callback failed",
                    extra={"tool": tool_def.name, "error": str(e)},
                )

        payload = {
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }
        if isinstance(result.output, str) and result.output.strip():
            payload["content"] = [{"type": "text", "text": result.output}]
        elif result.error:
            payload["content"] = [{"type": "text", "text": result.error}]
        return payload

    return handler


def get_playwright_mcp_config(
    test_name: Optional[str] = None,
    headless: Optional[bool] = None,
    save_video: Optional[str] = None,
    viewport_size: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build McpStdioServerConfig for Playwright MCP server.

    Returns a stdio MCP server config that the Claude Agent SDK will manage
    directly, eliminating the double-indirection through Dokumen MCP wrappers.

    Args:
        test_name: Test identifier (used for output directory).
        headless: Run browser in headless mode (defaults to True for CI safety).
        save_video: Video size string, e.g. "1920x1080" (None to disable).
        viewport_size: Browser viewport size, e.g. "1512x982".
        output_dir: Override output directory for recordings.

    Returns:
        McpStdioServerConfig dict for use in ClaudeAgentOptions.mcp_servers.
    """
    import os

    if headless is None:
        headless = True  # Default to headless for CI safety

    # Resolve output directory
    if not output_dir:
        output_dir = os.path.join(".dokumen-cache", "output", test_name or "default", "recordings")

    # Determine command: prefer local Playwright MCP fork, fall back to upstream
    # Path: sdk/tools.py → sdk/ → dokumen/ → dokumen-cli/ → dokumen/ (repo root)
    fork_path = os.environ.get("PLAYWRIGHT_MCP_PATH")
    if not fork_path:
        fork_path = os.path.join(
            os.path.dirname(  # dokumen/ (repo root)
                os.path.dirname(  # dokumen-cli/
                    os.path.dirname(os.path.dirname(__file__))  # dokumen/  # sdk/
                )
            ),
            "playwright-mcp-fork",
        )
    cli_path = os.path.join(fork_path, "packages", "playwright", "cli.js")
    built_program = os.path.join(fork_path, "packages", "playwright", "lib", "program.js")

    if os.path.exists(cli_path) and os.path.exists(built_program):
        command = "node"
        args = [cli_path, "run-mcp-server"]
        use_fork = True
        logger.info(
            "Playwright MCP config using fork",
            extra={"fork_path": fork_path, "cli_path": cli_path},
        )
    else:
        command = "npx"
        args = ["@playwright/mcp@latest"]
        use_fork = False
        logger.info(
            "Playwright MCP config using upstream",
            extra={"fork_path": fork_path},
        )

    # Always start with isolated profile
    args.append("--isolated")

    # Disable Chromium sandbox when running as root or explicitly requested
    sandbox_env = (os.environ.get("PLAYWRIGHT_MCP_SANDBOX") or "").lower()
    sandbox_disabled = sandbox_env in ("0", "false", "no", "off")
    sandbox_enabled = sandbox_env in ("1", "true", "yes", "on")
    if sandbox_disabled:
        args.append("--no-sandbox")
    elif hasattr(os, "geteuid") and os.geteuid() == 0 and not sandbox_enabled:
        args.append("--no-sandbox")

    if headless:
        args.append("--headless")

    if save_video:
        args.extend(["--save-video", save_video])
        args.extend(["--output-dir", output_dir])

    if viewport_size:
        args.extend(["--viewport-size", viewport_size])

    # Visual indicators (only with fork)
    if use_fork:
        args.append("--visual-indicators")

    logger.info(
        "Playwright MCP stdio config built",
        extra={
            "command": command,
            "cmd_args": args,
            "headless": headless,
            "save_video": save_video,
        },
    )

    return {
        "type": "stdio",
        "command": command,
        "args": args,
    }
