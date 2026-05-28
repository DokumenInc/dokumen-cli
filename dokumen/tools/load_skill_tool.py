"""Load Skill tool for CLI executor agents.

Provides a runtime tool that lets agents dynamically discover and load
workspace skills from SKILL.md files. This enables agents to access
specialized instructions and workflows during test execution.

The tool scans configurable directories for SKILL.md files with YAML
frontmatter, similar to how ``read_file`` provides file access.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dokumen.logging_config import get_logger
from dokumen.tools.types import ToolDefinition, ToolResult

logger = get_logger(__name__)

# Default directories to scan for SKILL.md files
_DEFAULT_SKILL_PATHS = [".skills/", "skills/", ".dokumen/skills/", "docs/skills/"]

# Maximum number of skills to discover
_MAX_SKILLS = 50


@dataclass(frozen=True)
class SkillInfo:
    """Discovered skill metadata and content.

    Attributes:
        name: Unique skill identifier from YAML frontmatter.
        description: Short description for catalog display.
        file_path: Relative path to the SKILL.md file within the workspace.
        content: Full markdown body (after frontmatter).
        argument_hint: Hint for arguments (e.g., "[issue-number]").
    """

    name: str
    description: str
    file_path: str
    content: str
    argument_hint: Optional[str] = None


def _parse_skill_file(raw_content: str, file_path: str) -> Optional[SkillInfo]:
    """Parse a SKILL.md file's YAML frontmatter and body.

    Expects ``---`` delimited YAML frontmatter with at least ``name``
    and ``description`` fields. Returns None if frontmatter is missing
    or invalid.

    Args:
        raw_content: Full text content of the SKILL.md file.
        file_path: Relative path for logging and SkillInfo.

    Returns:
        SkillInfo if valid, None otherwise.
    """
    stripped = raw_content.strip()
    if not stripped.startswith("---"):
        logger.debug("skill.parse.no_frontmatter", path=file_path)
        return None

    # Find closing ---
    end_idx = stripped.find("---", 3)
    if end_idx == -1:
        logger.debug("skill.parse.unclosed_frontmatter", path=file_path)
        return None

    frontmatter_text = stripped[3:end_idx].strip()
    body = stripped[end_idx + 3 :].strip()

    # Parse YAML frontmatter (simple key: value pairs, no external deps)
    fields: Dict[str, str] = {}
    current_key: Optional[str] = None
    multiline_value: List[str] = []

    for line in frontmatter_text.splitlines():
        # Check for new key: value pair
        if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
            # Flush previous multiline value
            if current_key and multiline_value:
                fields[current_key] = "\n".join(multiline_value).strip()
                multiline_value = []

            colon_idx = line.index(":")
            key = line[:colon_idx].strip()
            value = line[colon_idx + 1 :].strip()

            # Handle YAML block scalar indicators (| or >)
            if value in ("|", ">", "|-", ">-"):
                current_key = key
                multiline_value = []
            else:
                # Strip surrounding quotes
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                fields[key] = value
                current_key = None
        elif current_key:
            # Continuation of multiline value
            multiline_value.append(line.rstrip())

    # Flush last multiline value
    if current_key and multiline_value:
        fields[current_key] = "\n".join(multiline_value).strip()

    name = fields.get("name", "").strip()
    description = fields.get("description", "").strip()

    if not name:
        logger.debug("skill.parse.missing_name", path=file_path)
        return None
    if not description:
        logger.debug("skill.parse.missing_description", path=file_path)
        return None

    argument_hint = fields.get("argument-hint", "").strip() or None

    logger.info(
        "skill.parsed",
        skill_name=name,
        path=file_path,
        body_length=len(body),
    )

    return SkillInfo(
        name=name,
        description=description,
        file_path=file_path,
        content=body,
        argument_hint=argument_hint,
    )


def _substitute_arguments(content: str, arguments: str) -> str:
    """Substitute $ARGUMENTS placeholders in skill content.

    Supports:
    - ``$ARGUMENTS`` -- full argument string
    - ``$ARGUMENTS[N]`` -- positional access (0-based)
    - ``$N`` -- shorthand for ``$ARGUMENTS[N]`` (single digit)

    If ``$ARGUMENTS`` is not present in content and arguments are provided,
    they are appended as ``ARGUMENTS: <value>``.

    Args:
        content: Skill content body.
        arguments: User-provided argument string.

    Returns:
        Content with substitutions applied.
    """
    parts = arguments.split() if arguments else []

    has_arguments_placeholder = "$ARGUMENTS" in content or re.search(r"\$\d\b", content)

    # $ARGUMENTS[N] positional
    def _replace_positional(match: re.Match) -> str:
        idx = int(match.group(1))
        return parts[idx] if idx < len(parts) else ""

    result = re.sub(r"\$ARGUMENTS\[(\d+)\]", _replace_positional, content)

    # $N shorthand (single digit, not preceded by ARGUMENTS[)
    def _replace_shorthand(match: re.Match) -> str:
        idx = int(match.group(1))
        return parts[idx] if idx < len(parts) else ""

    result = re.sub(r"(?<!\[)\$(\d)(?!\])", _replace_shorthand, result)

    # $ARGUMENTS full replacement (after positional to avoid double-replace)
    result = result.replace("$ARGUMENTS", arguments)

    # If no placeholder was found and args provided, append them
    if not has_arguments_placeholder and arguments.strip():
        result = f"{result}\n\nARGUMENTS: {arguments}"

    return result


def _discover_skills(workspace_dir: str) -> List[SkillInfo]:
    """Scan workspace for SKILL.md files and return discovered skills.

    Args:
        workspace_dir: Absolute path to the workspace root.

    Returns:
        List of SkillInfo, deduplicated by name (first found wins),
        capped at _MAX_SKILLS.
    """
    workspace_path = Path(workspace_dir).resolve()
    seen_names: set = set()
    skills: List[SkillInfo] = []

    logger.info(
        "load_skill.discover.start",
        workspace=workspace_dir,
        scan_paths=_DEFAULT_SKILL_PATHS,
    )

    for scan_dir in _DEFAULT_SKILL_PATHS:
        if len(skills) >= _MAX_SKILLS:
            break

        base = workspace_path / scan_dir
        if not base.is_dir():
            continue

        for skill_file in sorted(base.rglob("SKILL.md")):
            if len(skills) >= _MAX_SKILLS:
                break

            # Workspace containment: resolve symlinks and verify
            try:
                resolved = skill_file.resolve()
                if not resolved.is_relative_to(workspace_path):
                    logger.warning(
                        "load_skill.discover.symlink_escape",
                        path=str(skill_file),
                        resolved=str(resolved),
                        workspace=workspace_dir,
                    )
                    continue
            except (OSError, ValueError) as e:
                logger.warning(
                    "load_skill.discover.resolve_error",
                    path=str(skill_file),
                    error=str(e),
                )
                continue

            # Compute relative path
            try:
                rel_path = str(skill_file.relative_to(workspace_path))
            except ValueError:
                logger.warning(
                    "load_skill.discover.relative_path_error",
                    path=str(skill_file),
                    workspace=workspace_dir,
                )
                continue

            # Read and parse
            try:
                raw = skill_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(
                    "load_skill.discover.read_error",
                    path=rel_path,
                    error=str(e),
                )
                continue

            info = _parse_skill_file(raw, rel_path)
            if info is None:
                continue

            # Dedup by name (first wins)
            if info.name in seen_names:
                logger.info(
                    "load_skill.discover.duplicate",
                    skill_name=info.name,
                    path=rel_path,
                )
                continue

            seen_names.add(info.name)
            skills.append(info)

    logger.info(
        "load_skill.discover.complete",
        workspace=workspace_dir,
        skills_found=len(skills),
        skill_names=[s.name for s in skills],
    )

    return skills


def create_load_skill_tool(workspace_dir: str) -> ToolDefinition:
    """Create a load_skill tool bound to a workspace directory.

    The tool discovers SKILL.md files in standard directories within
    the workspace and allows agents to list or load them by name.

    Args:
        workspace_dir: Absolute path to the workspace root directory.

    Returns:
        ToolDefinition for the load_skill tool.
    """
    logger.info("load_skill.create", workspace_dir=workspace_dir)

    async def handler(params: Dict[str, Any]) -> ToolResult:
        """Handle load_skill tool invocations.

        Args:
            params: Tool parameters with optional 'name' and 'arguments'.

        Returns:
            ToolResult with skill listing or content.
        """
        skill_name = params.get("name", "").strip()
        arguments = params.get("arguments", "").strip()

        logger.info(
            "load_skill.handler.entry",
            workspace_dir=workspace_dir,
            skill_name=skill_name or "(list mode)",
            has_arguments=bool(arguments),
        )

        # Validate workspace exists
        workspace_path = Path(workspace_dir)
        if not workspace_path.is_dir():
            logger.error(
                "load_skill.handler.workspace_missing",
                workspace_dir=workspace_dir,
            )
            return ToolResult(
                success=False,
                output=None,
                error=f"Workspace directory does not exist: {workspace_dir}",
            )

        # Discover skills
        skills = _discover_skills(workspace_dir)
        skills_by_name = {s.name: s for s in skills}

        if not skill_name:
            # List mode
            logger.info(
                "load_skill.handler.list",
                skills_count=len(skills),
            )
            if not skills:
                return ToolResult(
                    success=True,
                    output="No skills available in this workspace.",
                )

            lines = ["## Available Skills\n"]
            for s in skills:
                hint = ""
                if s.argument_hint:
                    hint = f" {s.argument_hint}"
                lines.append(f"- **{s.name}**{hint}: {s.description}")

            return ToolResult(success=True, output="\n".join(lines))

        # Load mode
        skill = skills_by_name.get(skill_name)
        if skill is None:
            available = ", ".join(skills_by_name.keys()) or "none"
            logger.warning(
                "load_skill.handler.not_found",
                skill_name=skill_name,
                available=available,
            )
            return ToolResult(
                success=False,
                output=None,
                error=f'Skill "{skill_name}" not found. Available skills: {available}',
            )

        # Apply argument substitution
        content = skill.content.strip()
        if arguments:
            content = _substitute_arguments(content, arguments)

        logger.info(
            "load_skill.handler.loaded",
            skill_name=skill_name,
            content_length=len(content),
            has_arguments=bool(arguments),
            source=skill.file_path,
        )

        output_lines = [
            f"## Skill: {skill.name}",
            "",
            f"**Source**: {skill.file_path}",
            "",
            content,
        ]

        return ToolResult(success=True, output="\n".join(output_lines))

    return ToolDefinition(
        name="load_skill",
        description=(
            "Discover and load workspace skills from SKILL.md files. "
            "Skills provide specialized instructions and workflows. "
            "Call with an empty name to list available skills, "
            "or provide a skill name to load its content."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Skill name to load. Leave empty to list all available skills."
                    ),
                },
                "arguments": {
                    "type": "string",
                    "description": (
                        "Arguments to substitute into $ARGUMENTS placeholders " "in skill content."
                    ),
                },
            },
        },
        handler=handler,
    )
