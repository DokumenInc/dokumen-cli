"""Per-user tool override loading and validation for CLI.

Reads ``.dokumen/tool-definitions/*.yaml`` from the project root and
validates overrides.  Maps canonical tool IDs to CLI runtime names.

The ``IMPLEMENTABLE_IN`` map defines which systems actually have executable
code for each tool.  This is mirrored from the backend module but kept
standalone so the CLI has no backend dependency.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .logging_config import get_logger

logger = get_logger(__name__)

# Canonical -> CLI name mapping (4 tools with different names in CLI)
CANONICAL_TO_CLI: dict[str, str] = {
    "glob_files": "glob",
    "list_files": "list_directory",
    "search_files": "search_file_content",
}
CLI_TO_CANONICAL: dict[str, str] = {v: k for k, v in CANONICAL_TO_CLI.items()}

# Valid systems for user overrides (mcp excluded -- not user-controllable)
VALID_OVERRIDE_SYSTEMS = frozenset({"chat", "test", "explore"})

# Tools that are safe for the explore phase (read-only, no side effects).
# Canonical IDs only -- CLI maps internally via CANONICAL_TO_CLI.
EXPLORE_SAFE_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "search_files",
        "glob_files",
        "explore_code",
    }
)

# Which systems have executable runtime code for each tool.
# Mirrored from backend/tools/user_tool_overrides.py -- kept standalone
# so the CLI has no backend dependency.
IMPLEMENTABLE_IN: dict[str, frozenset[str]] = {
    # Chat + CLI read-only
    "read_file": frozenset({"chat", "test", "explore"}),
    "list_files": frozenset({"chat", "test", "explore"}),
    "search_files": frozenset({"chat", "test", "explore"}),
    "glob_files": frozenset({"chat", "test", "explore"}),
    # Write tools (write_file has CLI runtime; others chat-only)
    "write_file": frozenset({"chat", "test"}),
    "edit_file": frozenset({"chat"}),
    "delete_file": frozenset({"chat"}),
    "apply_patch": frozenset({"chat"}),
    "get_git_changes": frozenset({"chat"}),
    "spawn_agent": frozenset({"chat"}),
    "batch": frozenset({"chat"}),
    # CLI-only delegation tool (executors + judges)
    "delegate_to_agent": frozenset({"test"}),
    # Chat + CLI execution
    "run_shell_command": frozenset({"chat", "test"}),
    # CLI-only read tools
    "read_many_files": frozenset({"test"}),
    # CLI-only network tools
    "web_fetch": frozenset({"test"}),
    "web_search": frozenset({"test"}),
    # CLI-only browser tools
    "browser_navigate": frozenset({"test"}),
    "browser_snapshot": frozenset({"test"}),
    "browser_click": frozenset({"test"}),
    "browser_type": frozenset({"test"}),
    "browser_wait": frozenset({"test"}),
    "browser_screenshot": frozenset({"test"}),
    "browser_take_screenshot": frozenset({"test"}),
    "browser_evaluate": frozenset({"test"}),
    "browser_close": frozenset({"test"}),
    # Chat explore-only
    "explore_code": frozenset({"chat", "explore"}),
    # Chat API tools (backend-only handlers, no CLI runtime)
    "list_tests": frozenset({"chat"}),
    "trigger_tests": frozenset({"chat"}),
    "validate_test": frozenset({"chat"}),
    "cancel_pipeline": frozenset({"chat"}),
    "get_company_stats": frozenset({"chat"}),
    "get_coverage": frozenset({"chat"}),
    "get_file_coverage": frozenset({"chat"}),
    "get_job_log": frozenset({"chat"}),
    "get_latest_results": frozenset({"chat"}),
    "get_pipeline_results": frozenset({"chat"}),
    "get_pipeline_status": frozenset({"chat"}),
    "get_test_details": frozenset({"chat"}),
    "list_tasks": frozenset({"chat"}),
}


@dataclass(frozen=True)
class ToolOverridesResult:
    """Validated tool overrides from user YAML files.

    Attributes:
        overrides: Mapping of canonical tool name to its effective
            ``available_in`` systems.  Empty frozenset means disabled.
        errors: Validation errors encountered (non-fatal -- invalid entries
            are skipped, valid entries are kept).
    """

    overrides: dict[str, frozenset[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def validate_tool_overrides(
    raw: dict[str, list[str]],
) -> ToolOverridesResult:
    """Validate raw tool override data from user YAML files.

    Args:
        raw: Mapping of tool name to list of system strings
            (e.g. ``{"read_file": ["chat", "test", "explore"]}``).
            ``available_in: []`` is valid and means disabled.

    Returns:
        ToolOverridesResult with validated overrides and any errors.
    """
    logger.info(
        "user_tool_overrides.validate",
        override_count=len(raw),
    )
    overrides: dict[str, frozenset[str]] = {}
    errors: list[str] = []

    for tool_name, systems_list in raw.items():
        # Unknown tool check
        if tool_name not in IMPLEMENTABLE_IN:
            errors.append(f"Unknown tool: {tool_name}")
            logger.warning(
                "user_tool_overrides.unknown_tool",
                tool=tool_name,
            )
            continue

        # Empty available_in is valid (disabled)
        if not systems_list:
            overrides[tool_name] = frozenset()
            logger.debug(
                "user_tool_overrides.disabled",
                tool=tool_name,
            )
            continue

        # Validate each system value
        valid_systems: set[str] = set()
        implementable = IMPLEMENTABLE_IN[tool_name]

        for system in systems_list:
            if system not in VALID_OVERRIDE_SYSTEMS:
                errors.append(
                    f"Invalid system '{system}' for tool '{tool_name}' "
                    f"(valid: {sorted(VALID_OVERRIDE_SYSTEMS)})"
                )
                continue

            # Runtime-compatibility check
            if system not in implementable:
                errors.append(
                    f"No {system} implementation for '{tool_name}' "
                    f"(implementable in: {sorted(implementable)})"
                )
                continue

            # Explore safety check
            if system == "explore" and tool_name not in EXPLORE_SAFE_TOOLS:
                errors.append(
                    f"Tool '{tool_name}' is not safe for explore " f"(only read-only tools allowed)"
                )
                continue

            valid_systems.add(system)

        overrides[tool_name] = frozenset(valid_systems)

    logger.info(
        "user_tool_overrides.validated",
        valid_count=len(overrides),
        error_count=len(errors),
    )
    return ToolOverridesResult(overrides=overrides, errors=errors)


def load_overrides_from_dir(project_root: str) -> Optional[ToolOverridesResult]:
    """Read ``.dokumen/tool-definitions/*.yaml`` from project root.

    Returns ``None`` if directory doesn't exist (legacy mode).
    Returns ``ToolOverridesResult`` if directory exists (YAML mode).

    Args:
        project_root: Path to the project root directory.

    Returns:
        ToolOverridesResult if directory exists, None otherwise.
    """
    overrides_dir = Path(project_root) / ".dokumen" / "tool-definitions"
    if not overrides_dir.is_dir():
        logger.debug(
            "user_tool_overrides.no_dir",
            path=str(overrides_dir),
        )
        return None

    logger.info(
        "user_tool_overrides.loading",
        path=str(overrides_dir),
    )

    raw: dict[str, list[str]] = {}
    parse_errors: list[str] = []

    for yaml_path in sorted(list(overrides_dir.glob("*.yaml")) + list(overrides_dir.glob("*.yml"))):
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                parse_errors.append(f"Invalid YAML in {yaml_path.name}: not a mapping")
                logger.warning(
                    "user_tool_overrides.not_mapping",
                    file=yaml_path.name,
                )
                continue
            name = data.get("name", yaml_path.stem)
            available_in = data.get("available_in", [])
            if not isinstance(available_in, list):
                parse_errors.append(
                    f"Invalid available_in in {yaml_path.name}: "
                    f"expected list, got {type(available_in).__name__}"
                )
                logger.warning(
                    "user_tool_overrides.bad_available_in",
                    file=yaml_path.name,
                    type=type(available_in).__name__,
                )
                continue
            raw[name] = available_in
            logger.debug(
                "user_tool_overrides.parsed_file",
                file=yaml_path.name,
                tool=name,
                available_in=available_in,
            )
        except Exception as e:
            parse_errors.append(f"Error parsing {yaml_path.name}: {e}")
            logger.warning(
                "user_tool_overrides.parse_error",
                file=yaml_path.name,
                error=str(e),
            )

    result = validate_tool_overrides(raw)
    # Merge parse errors with validation errors
    all_errors = parse_errors + list(result.errors)

    logger.info(
        "user_tool_overrides.loaded",
        tool_count=len(result.overrides),
        error_count=len(all_errors),
    )
    return ToolOverridesResult(overrides=result.overrides, errors=all_errors)


def is_tool_enabled_for_test(
    tool_cli_name: str,
    overrides: Optional[ToolOverridesResult],
) -> bool:
    """Check if a CLI tool is enabled for the test system.

    Maps CLI name to canonical name before checking overrides.

    Args:
        tool_cli_name: The CLI-side tool name (e.g. ``"glob"``).
        overrides: Validated overrides, or ``None`` for legacy mode
            (all tools enabled).

    Returns:
        True if the tool should be available for test execution.
    """
    if overrides is None:
        return True  # Legacy mode: all tools enabled (tools.blocked handles filtering)

    canonical = CLI_TO_CANONICAL.get(tool_cli_name, tool_cli_name)
    if canonical in overrides.overrides:
        enabled = "test" in overrides.overrides[canonical]
        logger.debug(
            "user_tool_overrides.is_enabled_for_test",
            cli_name=tool_cli_name,
            canonical=canonical,
            enabled=enabled,
        )
        return enabled
    # Not overridden: tool uses its default (enabled)
    return True


def get_effective_tools_for_system(
    system: str,
    overrides: ToolOverridesResult,
) -> set[str]:
    """Get canonical tool names enabled for a system.

    For tools present in overrides, uses the override. For tools NOT
    in overrides, falls back to ``IMPLEMENTABLE_IN`` defaults (the
    tool's default system availability).

    Args:
        system: One of ``"chat"``, ``"test"``, ``"explore"``.
        overrides: Validated overrides result.

    Returns:
        Set of canonical tool names where *system* is in the
        effective ``available_in``.
    """
    logger.debug(
        "user_tool_overrides.get_effective",
        system=system,
    )
    result: set[str] = set()
    for tool_name, default_systems in IMPLEMENTABLE_IN.items():
        if tool_name in overrides.overrides:
            effective = overrides.overrides[tool_name]
        else:
            effective = default_systems
        if system in effective:
            result.add(tool_name)
    return result
