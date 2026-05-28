"""SDK-native subagent delegation via AgentDefinition.

Converts Dokumen agent YAML definitions into SDK AgentDefinition objects
for native subagent support. The SDK manages subprocess lifecycle, context
isolation, and result passing — no custom delegation handler needed.

Usage:
    from .delegate import build_sdk_agent_definitions
    agents = build_sdk_agent_definitions(agent_context)
    # Pass to ClaudeAgentOptions(agents=agents)
"""

import logging
from typing import Dict, List, Optional

from claude_agent_sdk import AgentDefinition

from .tools import AgentContext, SDK_MAPPING

logger = logging.getLogger(__name__)

# Internal agents that should not be exposed as subagents
_INTERNAL_AGENTS = {"compaction", "explore"}

# Reverse map: full model ID → SDK literal
_MODEL_ID_TO_SDK: Dict[str, str] = {}

# Valid SDK AgentDefinition model literals
_SDK_MODEL_LITERALS = {"sonnet", "opus", "haiku", "inherit"}


def _init_model_reverse_map() -> None:
    """Build reverse mapping from full model IDs to SDK literals."""
    global _MODEL_ID_TO_SDK
    if _MODEL_ID_TO_SDK:
        return

    try:
        from dokumen_schema.constants import KNOWN_MODEL_ALIASES
    except ImportError:
        logger.debug("dokumen_schema.constants not available, model mapping limited")
        return

    for alias, full_id in KNOWN_MODEL_ALIASES.items():
        # Only map the base aliases (sonnet, haiku, opus), not versioned ones
        if alias in _SDK_MODEL_LITERALS:
            _MODEL_ID_TO_SDK[full_id] = alias


def _map_model(model: Optional[str]) -> Optional[str]:
    """Map a Dokumen model name to an SDK AgentDefinition model literal.

    The SDK AgentDefinition.model accepts: "sonnet", "opus", "haiku", "inherit", or None.

    Args:
        model: Dokumen model alias or full model ID, or None.

    Returns:
        SDK-compatible model literal, or None to inherit from parent.
    """
    if model is None:
        return None

    # Already a valid SDK literal
    if model in _SDK_MODEL_LITERALS:
        return model

    # Try reverse mapping from full model ID
    _init_model_reverse_map()
    sdk_literal = _MODEL_ID_TO_SDK.get(model)
    if sdk_literal:
        return sdk_literal

    logger.warning(
        "Unknown model for SDK agent definition, will inherit from parent",
        extra={"model": model},
    )
    return None


def _map_tools_to_sdk(tools: List[str]) -> List[str]:
    """Map Dokumen tool names to SDK tool names.

    Uses SDK_MAPPING for known tools, passes through unknown names as-is.
    Deduplicates the result while preserving order.

    Args:
        tools: List of Dokumen tool names.

    Returns:
        List of SDK tool names (deduplicated).
    """
    result: List[str] = []
    seen: set = set()

    for tool_name in tools:
        sdk_name = SDK_MAPPING.get(tool_name, tool_name)
        if sdk_name not in seen:
            result.append(sdk_name)
            seen.add(sdk_name)

    return result


def build_sdk_agent_definitions(
    agent_context: AgentContext,
) -> Dict[str, AgentDefinition]:
    """Load all user-defined agents and convert to SDK AgentDefinition format.

    Discovers agent YAML files from the configured directories, skips
    internal agents (compaction, explore), and maps tool names and model
    aliases to SDK-compatible values.

    Args:
        agent_context: Runtime context with user_dirs for agent discovery.

    Returns:
        Dict mapping agent name to SDK AgentDefinition.
    """
    from dokumen_schema.agent_defs import list_agents, load_agent

    logger.info(
        "Building SDK agent definitions",
        extra={
            "user_dirs": [str(d) for d in (agent_context.user_dirs or [])],
            "base_dir": agent_context.base_dir,
        },
    )

    definitions: Dict[str, AgentDefinition] = {}

    agent_names = list_agents(user_dirs=agent_context.user_dirs)
    logger.info(
        "Discovered agents for delegation",
        extra={"agent_count": len(agent_names), "agents": agent_names},
    )

    for agent_name in agent_names:
        if agent_name in _INTERNAL_AGENTS:
            logger.debug(
                "Skipping internal agent",
                extra={"agent": agent_name},
            )
            continue

        agent_def = load_agent(agent_name, user_dirs=agent_context.user_dirs)
        if agent_def is None:
            logger.warning(
                "Agent failed to load, skipping",
                extra={"agent": agent_name},
            )
            continue

        sdk_tools = _map_tools_to_sdk(agent_def.tools)
        sdk_model = _map_model(agent_def.model)

        definitions[agent_def.name] = AgentDefinition(
            description=agent_def.description,
            prompt=agent_def.system_prompt or "",
            tools=sdk_tools if sdk_tools else None,
            model=sdk_model,
        )

        logger.info(
            "Converted agent to SDK definition",
            extra={
                "agent": agent_def.name,
                "sdk_tools": sdk_tools,
                "sdk_model": sdk_model,
            },
        )

    logger.info(
        "SDK agent definitions built",
        extra={"count": len(definitions), "agents": list(definitions.keys())},
    )

    return definitions
