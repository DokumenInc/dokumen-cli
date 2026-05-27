"""
Tool resolution for the Documentation Unit Test Framework.

Handles resolving tool names to ToolDefinition objects, auto-injection
of required tools based on agent capabilities, allowed/blocked list filtering,
and provenance tracking.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os

from .logging_config import get_logger
from .tools_object import ToolDefinition, ToolResult
from .user_tool_overrides import is_tool_enabled_for_test

logger = get_logger(__name__)


@dataclass
class ToolProvenance:
    """Tracks where each tool came from for observability.

    Source values:
        - "scaffold": explicitly listed in the test scaffold YAML
        - "defaults": from global tools.defaults in dokumen.yaml
        - "auto:standard": auto-injected run_shell_command for standard tests
        - "auto:research": auto-injected web_search for research tests
        - "auto:browser": auto-injected browser tools for browser tests
        - "auto:cross-reference": auto-injected code repository tools
        - "explore:config": explore tools from config/defaults
        - "explore:overrides": explore tools from .dokumen/tool-definitions/ overrides
    """
    executor_tools: Dict[str, str] = field(default_factory=dict)
    judge_tools: Dict[str, Dict[str, str]] = field(default_factory=dict)
    explore_tools: Dict[str, str] = field(default_factory=dict)
    overrides_active: bool = False
    removed_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize provenance to a plain dict for progress callbacks.

        Returns copies to prevent mutation of internal state.
        """
        return {
            'executor_tools': dict(self.executor_tools),
            'judge_tools': {k: dict(v) for k, v in self.judge_tools.items()},
            'explore_tools': dict(self.explore_tools),
            'overrides_active': self.overrides_active,
            'removed_tools': list(self.removed_tools),
        }


def determine_executor_tool_names(
    scaffold_tools: Optional[List[str]],
    tools_config: Optional[Any],
    scaffold_name: str,
    provenance: ToolProvenance,
) -> List[str]:
    """Determine the base executor tool names from scaffold or defaults.

    Args:
        scaffold_tools: Tools listed in the scaffold YAML (may be None/empty)
        tools_config: Project-level ToolsConfig from dokumen.yaml
        scaffold_name: Name of the scaffold for logging
        provenance: ToolProvenance to update with sources

    Returns:
        List of tool name strings
    """
    if scaffold_tools:
        executor_tool_names = list(scaffold_tools)
        for t in executor_tool_names:
            provenance.executor_tools[t] = "scaffold"
        logger.debug("Scaffold tools override defaults", scaffold=scaffold_name, tools=executor_tool_names)
    elif tools_config and tools_config.defaults is not None:
        executor_tool_names = list(tools_config.defaults)
        for t in executor_tool_names:
            provenance.executor_tools[t] = "defaults"
        logger.info("Global defaults applied", scaffold=scaffold_name, tools=executor_tool_names)
    else:
        executor_tool_names = []
    return executor_tool_names


def enforce_allowed_list(
    tool_names: List[str],
    tools_config: Optional[Any],
    scaffold_name: str,
) -> None:
    """Enforce the tools.allowed list from dokumen.yaml.

    Raises ValueError if any tools are not in the allowed list.
    """
    if tools_config and tools_config.allowed is not None:
        disallowed = set(tool_names) - set(tools_config.allowed)
        if disallowed:
            raise ValueError(
                f"Scaffold '{scaffold_name}' uses tools not in allowed list: {sorted(disallowed)}. "
                f"Allowed tools: {sorted(tools_config.allowed)}"
            )


def auto_inject_tools(
    executor_tool_names: List[str],
    is_browser_agent: bool,
    is_research_agent: bool,
    is_code_agent: bool,
    tools_config: Optional[Any],
    scaffold_name: str,
    provenance: ToolProvenance,
) -> tuple:
    """Auto-inject required tools based on agent capabilities.

    Returns:
        Tuple of (updated_tool_names, auto_injected_tools_set)
    """
    auto_injected_tools: set = set()

    # Auto-add run_shell_command (respects allowed list)
    if not is_browser_agent and not is_research_agent and 'run_shell_command' not in executor_tool_names:
        if tools_config is None or tools_config.allowed is None or 'run_shell_command' in tools_config.allowed:
            executor_tool_names = ['run_shell_command'] + list(executor_tool_names)
            auto_injected_tools.add('run_shell_command')
            provenance.executor_tools['run_shell_command'] = "auto:standard"
        else:
            logger.warning("run_shell_command auto-injection skipped: not in allowed list", scaffold=scaffold_name)

    # Auto-add web_search for research agents (respects allowed list)
    if is_research_agent and 'web_search' not in executor_tool_names:
        if tools_config is None or tools_config.allowed is None or 'web_search' in tools_config.allowed:
            executor_tool_names.append('web_search')
            auto_injected_tools.add('web_search')
            provenance.executor_tools['web_search'] = "auto:research"
            logger.info("web_search auto-injected for research test", scaffold=scaffold_name)
        else:
            logger.warning("web_search auto-injection skipped: not in allowed list", scaffold=scaffold_name)

    # Auto-add browser tools for browser agents (bypasses allowed list)
    if is_browser_agent:
        from .playwright_tools import get_browser_tool_names
        browser_tool_names_to_inject = get_browser_tool_names() + ['read_file']
        injected = []
        for bt_name in browser_tool_names_to_inject:
            if bt_name not in executor_tool_names:
                executor_tool_names.append(bt_name)
                injected.append(bt_name)
                auto_injected_tools.add(bt_name)
                provenance.executor_tools[bt_name] = "auto:browser"
        if injected:
            logger.info("Browser tools auto-injected for browser test", scaffold=scaffold_name, tools=injected)

    # Auto-add code tools for code-reviewer agents (bypasses allowed list)
    if is_code_agent:
        code_tools_to_inject = ["code_read_file", "code_search", "code_glob"]
        injected = []
        for ct_name in code_tools_to_inject:
            if ct_name not in executor_tool_names:
                executor_tool_names.append(ct_name)
                injected.append(ct_name)
                auto_injected_tools.add(ct_name)
                provenance.executor_tools[ct_name] = "auto:cross-reference"
        if injected:
            logger.info("Code tools auto-injected for cross-reference test", scaffold=scaffold_name, tools=injected)

    # Merge agent tools from DB (when DOKUMEN_AGENT_ID is set)
    from .agent_loader import get_agent_tools
    agent_tools = get_agent_tools()
    if agent_tools:
        injected = []
        for at_name in agent_tools:
            if at_name not in executor_tool_names:
                executor_tool_names.append(at_name)
                injected.append(at_name)
                provenance.executor_tools[at_name] = "agent:db"
        if injected:
            logger.info("Agent tools merged from DB", scaffold=scaffold_name, agent_tools=injected)

    return executor_tool_names, auto_injected_tools


def filter_tools_with_overrides(
    tool_names: List[str],
    auto_injected_tools: set,
    overrides: Optional[Any],
    tools_config: Optional[Any],
    scaffold_name: str,
    scaffold_agent: Optional[str],
    provenance: ToolProvenance,
) -> List[str]:
    """Apply tool filtering via overrides or legacy blocked list.

    Args:
        tool_names: Current tool name list
        auto_injected_tools: Set of auto-injected tool names (must not be disabled)
        overrides: User tool overrides from .dokumen/tool-definitions/
        tools_config: Project-level ToolsConfig
        scaffold_name: Scaffold name for logging
        scaffold_agent: Agent name for error messages
        provenance: ToolProvenance to update

    Returns:
        Filtered list of tool names
    """
    if overrides is not None:
        # YAML override mode
        for tool_name in auto_injected_tools:
            if not is_tool_enabled_for_test(tool_name, overrides):
                raise ValueError(
                    f"Tool '{tool_name}' is required for agent '{scaffold_agent or 'standard'}' "
                    f"but disabled in .dokumen/tool-definitions/. "
                    f"Enable it or change the agent."
                )
        before_names = set(tool_names)
        tool_names = [
            t for t in tool_names
            if is_tool_enabled_for_test(t, overrides)
        ]
        removed_names = before_names - set(tool_names)
        if removed_names:
            provenance.removed_tools.extend(sorted(removed_names))
            for t in removed_names:
                provenance.executor_tools.pop(t, None)
            logger.info(
                "Overrides removed executor tools",
                scaffold=scaffold_name,
                removed_count=len(removed_names),
            )
    elif tools_config and tools_config.blocked:
        # Legacy: existing tools.blocked filter unchanged
        blocked_set = set(tools_config.blocked)
        before_names = set(tool_names)
        tool_names = [t for t in tool_names if t not in blocked_set]
        removed_names = before_names - set(tool_names)
        if removed_names:
            provenance.removed_tools.extend(sorted(removed_names))
            for t in removed_names:
                provenance.executor_tools.pop(t, None)
            logger.info(
                "Blocked tools removed from executor",
                scaffold=scaffold_name,
                blocked_tools=sorted(blocked_set),
                removed_count=len(removed_names),
            )

    return tool_names


def filter_judge_tools(
    judge_tool_names: List[str],
    auto_added_judge_tools: set,
    overrides: Optional[Any],
    tools_config: Optional[Any],
    scaffold_name: str,
    judge_name: str,
    scaffold_agent: Optional[str],
    judge_prov: Dict[str, str],
) -> List[str]:
    """Apply tool filtering to judge tools.

    Args:
        judge_tool_names: Judge tool name list
        auto_added_judge_tools: Set of auto-added judge tool names
        overrides: User tool overrides
        tools_config: Project-level ToolsConfig
        scaffold_name: Scaffold name for logging
        judge_name: Judge name for logging
        scaffold_agent: Agent name for error messages
        judge_prov: Judge provenance dict to update

    Returns:
        Filtered judge tool names
    """
    if overrides is not None:
        for tool_name in auto_added_judge_tools:
            if not is_tool_enabled_for_test(tool_name, overrides):
                raise ValueError(
                    f"Tool '{tool_name}' is auto-required for judges with agent '{scaffold_agent or 'standard'}' "
                    f"but disabled in .dokumen/tool-definitions/. "
                    f"Enable it or change the agent."
                )
        before_judge_names = set(judge_tool_names)
        judge_tool_names = [
            t for t in judge_tool_names
            if is_tool_enabled_for_test(t, overrides)
        ]
        removed_judge_names = before_judge_names - set(judge_tool_names)
        for t in removed_judge_names:
            judge_prov.pop(t, None)
    elif tools_config and tools_config.blocked:
        blocked_set = set(tools_config.blocked)
        before_judge_names = set(judge_tool_names)
        judge_tool_names = [t for t in judge_tool_names if t not in blocked_set]
        removed_judge_names = before_judge_names - set(judge_tool_names)
        for t in removed_judge_names:
            judge_prov.pop(t, None)
        if removed_judge_names:
            logger.info(
                "Blocked tools removed from judge",
                scaffold=scaffold_name,
                judge=judge_name,
                blocked_tools=sorted(blocked_set),
                removed_count=len(removed_judge_names),
            )

    return judge_tool_names


def resolve_tools(
    tool_names: List[str],
    base_dir: str = ".",
    sandbox: Any = None,
    provider: Any = None,
    parent_tools: List[ToolDefinition] = None,
    perplexity_config: Optional[dict] = None,
    tools_config: Optional[Any] = None,
    code_repos_config: Optional[List[Dict[str, Any]]] = None,
    agent_registry: Any = None,
    agent_provider: Any = None,
) -> List[ToolDefinition]:
    """
    Resolve tool names to ToolDefinition objects.

    Supports: read_file, list_directory, glob, run_shell_command,
              search_file_content, web_fetch, web_search, browser tools,
              code_read_file, code_glob, code_search, code_list_directory,
              delegate_to_agent, load_skill

    Args:
        tool_names: List of tool names to resolve
        base_dir: Base directory for file-based tools
        sandbox: Ignored in Phase 0
        provider: Ignored in Phase 0
        parent_tools: Ignored in Phase 0
        perplexity_config: Optional dict for web_search config
        tools_config: Optional project-level tool configuration
        code_repos_config: Optional list of code repo configs
        agent_registry: Optional agent registry for delegate_to_agent
        agent_provider: Optional provider for delegate_to_agent

    Returns:
        List of ToolDefinition objects

    Raises:
        ValueError: If tool cannot be resolved
    """
    from .tools_object import (
        BUILTIN_TOOLS,
        SANDBOX_TOOLS,
        STANDALONE_TOOLS,
        CONTEXT_TOOLS,
        CODE_TOOLS,
        AGENT_TOOLS,
    )
    from .playwright_tools import BROWSER_TOOLS
    from .debug import debug

    debug("[DEBUG LOADER] resolve_tools called:")
    debug(f"[DEBUG LOADER]   tool_names: {tool_names}")
    debug(f"[DEBUG LOADER]   base_dir: {base_dir}")
    debug(f"[DEBUG LOADER]   code_repos_config: {code_repos_config is not None}")

    resolved = []

    for name in tool_names:
        debug(f"[DEBUG LOADER]   Resolving tool '{name}'...")
        if name in BUILTIN_TOOLS:
            debug(f"[DEBUG LOADER]     '{name}' is a BUILTIN_TOOLS (file read-only)")
            tool_factory = BUILTIN_TOOLS[name]
            resolved.append(tool_factory(base_dir))
        elif name in SANDBOX_TOOLS:
            debug(f"[DEBUG LOADER]     '{name}' is a SANDBOX_TOOLS")
            if name == "run_shell_command":
                shell_timeout = 30.0
                if tools_config and tools_config.config:
                    shell_timeout = tools_config.config.run_shell_command.timeout
                    logger.debug("Per-tool config applied", tool=name, timeout=shell_timeout)
                debug(f"[DEBUG LOADER]     using run_shell_command in dev mode with base_dir={base_dir}")
                from .tools_object import create_bash_tool
                resolved.append(create_bash_tool(sandbox=None, base_dir=base_dir, timeout=shell_timeout))
            elif name == "search_file_content":
                debug(f"[DEBUG LOADER]     using search_file_content in dev mode with base_dir={base_dir}")
                from .tools_object import create_grep_tool
                resolved.append(create_grep_tool(sandbox=None, base_dir=base_dir))
            else:
                debug(f"[DEBUG LOADER]     creating placeholder for '{name}'")
                resolved.append(_create_placeholder_tool(name))
        elif name in STANDALONE_TOOLS:
            debug(f"[DEBUG LOADER]     '{name}' is a STANDALONE_TOOLS")
            if name == "web_search":
                from .tools_object import create_perplexity_web_search_tool
                cfg = perplexity_config or {}
                ws_model = cfg.get("model", "sonar")
                ws_max = cfg.get("max_searches", 5)
                if tools_config and tools_config.config and tools_config.config.web_search:
                    tc_ws = tools_config.config.web_search
                    if tc_ws.model is not None:
                        ws_model = tc_ws.model
                    if tc_ws.max_searches is not None:
                        ws_max = tc_ws.max_searches
                    logger.debug("Per-tool config applied", tool=name, model=ws_model, max_searches=ws_max)
                resolved.append(create_perplexity_web_search_tool(
                    api_key=cfg.get("api_key"),
                    model=ws_model,
                    max_searches=ws_max,
                ))
            elif name == "anthropic_web_search":
                from .tools_object import create_anthropic_web_search_tool
                aws_max_uses = None
                aws_allowed = None
                aws_blocked = None
                if tools_config and tools_config.config and tools_config.config.anthropic_web_search:
                    tc_aws = tools_config.config.anthropic_web_search
                    aws_max_uses = tc_aws.max_uses
                    aws_allowed = tc_aws.allowed_domains
                    aws_blocked = tc_aws.blocked_domains
                    logger.debug(
                        "Per-tool config applied", tool=name,
                        max_uses=aws_max_uses,
                        has_allowed_domains=aws_allowed is not None,
                    )
                resolved.append(create_anthropic_web_search_tool(
                    max_uses=aws_max_uses,
                    allowed_domains=aws_allowed,
                    blocked_domains=aws_blocked,
                ))
            elif name == "web_fetch":
                fetch_timeout = 30.0
                if tools_config and tools_config.config:
                    fetch_timeout = tools_config.config.web_fetch.timeout
                    logger.debug("Per-tool config applied", tool=name, timeout=fetch_timeout)
                from .tools_object import create_http_request_tool
                resolved.append(create_http_request_tool(sandbox=None, timeout=fetch_timeout))
            else:
                tool_factory = STANDALONE_TOOLS[name]
                resolved.append(tool_factory(sandbox=None))
        elif name in BROWSER_TOOLS:
            debug(f"[DEBUG LOADER]     '{name}' is a BROWSER_TOOLS placeholder")
            resolved.append(_create_placeholder_tool(name))
        elif name == "delegate_to_agent":
            if agent_registry is not None and agent_provider is not None:
                debug(f"[DEBUG LOADER]     '{name}' resolved with agent registry")
                from .tools_object import create_delegate_to_agent_tool
                delegate_timeout = 60.0
                if tools_config and tools_config.config:
                    delegate_timeout = getattr(
                        getattr(tools_config.config, 'delegate_to_agent', None),
                        'timeout', 60.0
                    ) or 60.0
                logger.info(
                    "delegate_to_agent.resolved",
                    has_registry=True,
                    has_parent_tools=parent_tools is not None,
                    timeout=delegate_timeout,
                )
                resolved.append(create_delegate_to_agent_tool(
                    registry=agent_registry,
                    provider=agent_provider,
                    sandbox=sandbox,
                    timeout=delegate_timeout,
                    parent_tools=parent_tools,
                ))
            else:
                debug(f"[DEBUG LOADER]     '{name}' has no registry - creating placeholder")
                logger.info(
                    "delegate_to_agent.placeholder",
                    has_registry=False,
                    reason="No agent_registry or agent_provider provided",
                )
                resolved.append(_create_placeholder_tool(name))
        elif name in CONTEXT_TOOLS:
            debug(f"[DEBUG LOADER]     '{name}' is a CONTEXT_TOOLS - creating placeholder")
            resolved.append(_create_placeholder_tool(name))
        elif name in CODE_TOOLS:
            debug(f"[DEBUG LOADER]     '{name}' is a CODE_TOOLS")
            if not code_repos_config:
                raise ValueError(
                    f"Tool '{name}' requires code_repos_config. "
                    "Add a code_repos section to dokumen.yaml to use code repository tools."
                )
            repo_cfg = code_repos_config[0]
            code_base_dir = repo_cfg.get("base_dir", ".")
            include_patterns = repo_cfg.get("include_patterns", [])
            exclude_patterns = repo_cfg.get("exclude_patterns", [])
            logger.info(
                "Resolving code tool",
                tool=name,
                code_repo=repo_cfg.get("name", "unknown"),
                base_dir=code_base_dir,
            )
            tool_factory = CODE_TOOLS[name]
            resolved.append(tool_factory(
                base_dir=code_base_dir,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            ))
        elif name in AGENT_TOOLS:
            debug(f"[DEBUG LOADER]     '{name}' is an AGENT_TOOLS")
            config, project_root = _get_agent_tool_config(base_dir)
            logger.info(
                "Resolving agent tool",
                tool=name,
                project_root=project_root,
            )
            tool_factory = AGENT_TOOLS[name]
            resolved.append(tool_factory(config=config, project_root=project_root))
        elif name == "load_skill":
            debug(f"[DEBUG LOADER]     '{name}' is a SKILL_TOOL")
            logger.info(
                "Resolving load_skill tool",
                tool=name,
                base_dir=base_dir,
            )
            from .tools.load_skill_tool import create_load_skill_tool
            resolved.append(create_load_skill_tool(base_dir))
        else:
            raise ValueError(f"Unknown tool: {name}")

    # --- Register MCP tools (replace built-in handlers with same name) ---
    try:
        from .mcp_tools import create_mcp_tool_definitions

        mcp_tools = create_mcp_tool_definitions(branch=os.environ.get("MCP_BRANCH", "main"))
        if mcp_tools:
            resolved_names = {t.name for t in resolved}
            replaced = []
            added = []
            for mcp_tool in mcp_tools:
                if mcp_tool.name in resolved_names:
                    resolved = [
                        mcp_tool if t.name == mcp_tool.name else t
                        for t in resolved
                    ]
                    replaced.append(mcp_tool.name)
                else:
                    resolved.append(mcp_tool)
                    added.append(mcp_tool.name)
            logger.info(
                "resolve_tools.mcp_integrated",
                replaced=replaced,
                added=added,
                total_mcp=len(mcp_tools),
            )
    except Exception as e:
        debug(f"[DEBUG LOADER] MCP tools integration skipped: {e}")
        logger.debug("resolve_tools.mcp_skipped", error=str(e))

    return resolved


def _create_placeholder_tool(name: str) -> ToolDefinition:
    """Create a placeholder tool for unavailable tools."""

    async def placeholder_handler(params):
        return ToolResult(
            success=False,
            output=None,
            error=f"Tool '{name}' is not available in Phase 0."
        )

    return ToolDefinition(
        name=name,
        description=f"[Placeholder] {name} - not available in Phase 0",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=placeholder_handler
    )


def _get_agent_tool_config(base_dir: str):
    """Load DokumenConfig and project root for agent tools (explore/ask)."""
    from .config import load_config

    logger.info("agent_tools.load_config", base_dir=base_dir)
    config = load_config(base_dir=base_dir)
    project_root = os.path.abspath(base_dir)
    return config, project_root
