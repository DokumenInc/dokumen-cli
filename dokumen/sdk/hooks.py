"""
Validation hooks for the Claude Agent SDK path.

PreToolUse hooks enforce:
- Path allowlists for Read/Write/Edit tools
- Command validation for Bash
- Shell timeout enforcement from ToolConfigMap

PostToolUse hooks:
- on_tool_call callback for SDK built-in tools (logging/tracking)
"""

import logging
import os
from typing import Any, Callable, Dict, Optional

from claude_agent_sdk import HookMatcher

logger = logging.getLogger(__name__)


def _deny(message: str) -> dict:
    """Return a deny decision matching claude agent SDK PreToolUse format."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": message,
        }
    }


def is_path_allowed(path: str, tools_config: Optional[Any] = None) -> bool:
    """Check if a file path is within the project working directory.

    Normalizes the path (resolve .., make absolute) and checks it starts
    with the project root (os.getcwd()). Can be extended with sandbox
    config later.

    Args:
        path: The file path to validate.
        tools_config: Optional ToolsConfig for future extension.

    Returns:
        True if the path is within the project root, False otherwise.
    """
    if not path:
        logger.debug("Path validation: empty path denied")
        return False

    project_root = os.path.realpath(os.getcwd())
    resolved = os.path.realpath(os.path.join(project_root, path))

    allowed = resolved.startswith(project_root + os.sep) or resolved == project_root
    logger.debug(
        "Path validation",
        extra={
            "path": path,
            "resolved": resolved,
            "project_root": project_root,
            "allowed": allowed,
        },
    )
    return allowed


def is_command_allowed(command: str, tools_config: Optional[Any] = None) -> bool:
    """Check if a shell command is allowed.

    Currently returns True — CI sandbox handles isolation. This is a
    guardrail, not a sandbox. Can be extended with command allowlists
    from tools_config later.

    Args:
        command: The shell command string.
        tools_config: Optional ToolsConfig for future extension.

    Returns:
        True if the command is allowed.
    """
    logger.debug("Command validation", extra={"command": command, "allowed": True})
    return True


def build_validation_hooks(
    tools_config: Optional[Any] = None,
    on_tool_call: Optional[Callable] = None,
) -> Dict[str, list]:
    """Build combined PreToolUse + PostToolUse hooks dict.

    Args:
        tools_config: Optional ToolsConfig with .config.run_shell_command.timeout.
        on_tool_call: Optional callback(tool_name, tool_input, tool_response)
            invoked after each tool call for logging/tracking.

    Returns:
        Dict with "PreToolUse" and optionally "PostToolUse" hook matcher lists.
    """
    logger.info(
        "Building validation hooks",
        extra={
            "has_tools_config": tools_config is not None,
            "has_on_tool_call": on_tool_call is not None,
        },
    )

    async def validate_pre_tool(input_data, tool_use_id, context):
        """PreToolUse hook: path allowlists, command validation, timeout."""
        tool_name = input_data["tool_name"]
        tool_input = input_data["tool_input"]

        # Path validation for write tools only — read tools can access anywhere
        # (mimick needs to read external repos, explore reads the whole codebase)
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            if not is_path_allowed(file_path, tools_config):
                reason = f"Path not allowed: {file_path}"
                logger.warning(
                    "PreToolUse denied file tool",
                    extra={"tool": tool_name, "path": file_path, "reason": reason},
                )
                return _deny(reason)

        # Command validation for Bash
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            if not is_command_allowed(command, tools_config):
                reason = f"Command not allowed: {command}"
                logger.warning(
                    "PreToolUse denied bash command",
                    extra={"tool": tool_name, "command": command, "reason": reason},
                )
                return _deny(reason)

            # Timeout enforcement from tools_config
            if tools_config is not None:
                try:
                    config_timeout = tools_config.config.run_shell_command.timeout
                    timeout_ms = int(config_timeout * 1000)
                    current_timeout = tool_input.get("timeout")
                    if current_timeout is None or current_timeout > timeout_ms:
                        tool_input["timeout"] = timeout_ms
                        logger.debug(
                            "Injected bash timeout from tools_config",
                            extra={
                                "config_timeout_s": config_timeout,
                                "timeout_ms": timeout_ms,
                            },
                        )
                except AttributeError:
                    logger.debug(
                        "tools_config missing run_shell_command.timeout, skipping"
                    )

        logger.debug(
            "PreToolUse allowed",
            extra={"tool": tool_name},
        )
        return {}

    hooks: Dict[str, list] = {
        "PreToolUse": [
            HookMatcher(
                matcher="Read|Write|Edit|Bash",
                hooks=[validate_pre_tool],
            )
        ],
    }

    if on_tool_call is not None:

        async def post_hook(input_data, tool_use_id, context):
            """PostToolUse hook: invoke on_tool_call callback."""
            tool_name = input_data["tool_name"]
            tool_input = input_data["tool_input"]
            tool_response = input_data.get("tool_response")

            logger.debug(
                "PostToolUse callback",
                extra={"tool": tool_name},
            )
            on_tool_call(tool_name, tool_input, tool_response)
            return {}

        hooks["PostToolUse"] = [
            HookMatcher(hooks=[post_hook]),
        ]

    # Subagent lifecycle hooks for observability
    async def on_subagent_start(input_data, tool_use_id, context):
        """SubagentStart hook: log subagent spawn event."""
        agent_id = input_data.get("agent_id", "unknown")
        agent_type = input_data.get("agent_type", "unknown")
        logger.info(
            "Subagent started",
            extra={"agent_id": agent_id, "agent_type": agent_type},
        )
        return {}

    async def on_subagent_stop(input_data, tool_use_id, context):
        """SubagentStop hook: log subagent completion event."""
        agent_id = input_data.get("agent_id", "unknown")
        agent_type = input_data.get("agent_type", "unknown")
        transcript = input_data.get("agent_transcript_path", "")
        logger.info(
            "Subagent stopped",
            extra={
                "agent_id": agent_id,
                "agent_type": agent_type,
                "transcript_path": transcript,
            },
        )
        return {}

    hooks["SubagentStart"] = [
        HookMatcher(hooks=[on_subagent_start]),
    ]
    hooks["SubagentStop"] = [
        HookMatcher(hooks=[on_subagent_stop]),
    ]

    logger.info(
        "Validation hooks built",
        extra={"hook_keys": list(hooks.keys())},
    )
    return hooks
