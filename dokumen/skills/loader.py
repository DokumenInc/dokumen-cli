"""
skill loader — loads skills from disk and bundles defaults.

skills are prompt-based commands that get injected into agent
system messages at startup. they can run inline (injected into prompt)
or as forked sub-agents.

inspired by claude code's skill architecture but built originally.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """how a skill runs."""

    INLINE = "inline"  # injected into prompt
    FORK = "fork"  # runs as separate agent


@dataclass
class SkillDefinition:
    """a loadable skill definition.

    skills live as .yaml or .md files in the skills/ directory.
    yaml format:
        name: my-skill
        description: what this skill does
        mode: inline  # or fork
        prompt: |
          the skill prompt text...
        tools: [read_file, glob]  # optional tool allowlist
    """

    name: str
    description: str = ""
    prompt: str = ""
    mode: ExecutionMode = ExecutionMode.INLINE
    tools: List[str] = field(default_factory=list)
    model: Optional[str] = None  # override model for fork mode
    when_to_use: str = ""  # hint for agent about when to use this
    effectiveness: float = 1.0  # decay over time if not helpful
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "mode": self.mode.value,
            "tools": self.tools,
            "model": self.model,
            "when_to_use": self.when_to_use,
            "effectiveness": self.effectiveness,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillDefinition":
        return cls(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            prompt=d.get("prompt", ""),
            mode=ExecutionMode(d.get("mode", "inline")),
            tools=d.get("tools", []),
            model=d.get("model"),
            when_to_use=d.get("when_to_use", ""),
            effectiveness=d.get("effectiveness", 1.0),
            metadata=d.get("metadata", {}),
        )


def load_skill_file(filepath: str) -> Optional[SkillDefinition]:
    """load a skill from a yaml or markdown file.

    yaml files are parsed as structured skill defs.
    markdown files use the filename as name and content as prompt.
    """
    path = Path(filepath)
    if not path.exists():
        return None

    try:
        content = path.read_text(encoding="utf-8")
    except (IOError, OSError) as e:
        logger.warning("failed to read skill file", extra={"path": filepath, "error": str(e)})
        return None

    if path.suffix in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                return SkillDefinition.from_dict(data)
        except yaml.YAMLError as e:
            logger.warning("failed to parse skill yaml", extra={"path": filepath, "error": str(e)})
            return None

    elif path.suffix == ".md":
        # markdown: filename is name, content is prompt
        name = path.stem
        # check for yaml frontmatter
        if content.startswith("---"):
            import re

            match = re.match(r"^---\n(.*?)\n---\n\n?(.*)", content, re.DOTALL)
            if match:
                try:
                    frontmatter = yaml.safe_load(match.group(1))
                    prompt = match.group(2).strip()
                    if isinstance(frontmatter, dict):
                        return SkillDefinition(
                            name=frontmatter.get("name", name),
                            description=frontmatter.get("description", ""),
                            prompt=prompt,
                            mode=ExecutionMode(frontmatter.get("mode", "inline")),
                            tools=frontmatter.get("tools", []),
                        )
                except yaml.YAMLError:
                    pass

        return SkillDefinition(name=name, prompt=content)

    return None


def load_skills_from_directory(directory: str) -> List[SkillDefinition]:
    """load all skills from a directory.

    scans for .yaml, .yml, and .md files.
    """
    skills = []
    path = Path(directory)

    if not path.exists() or not path.is_dir():
        logger.debug("skills directory does not exist", extra={"directory": directory})
        return skills

    for filepath in sorted(path.iterdir()):
        if filepath.suffix in (".yaml", ".yml", ".md"):
            skill = load_skill_file(str(filepath))
            if skill:
                skills.append(skill)
                logger.debug("loaded skill", extra={"name": skill.name, "mode": skill.mode.value})

    logger.info(
        "loaded skills from directory",
        extra={"directory": directory, "count": len(skills)},
    )
    return skills


# ── system skills ──
# loaded from dokumen/skills/*.yaml at import time
# these ship with dokumen. project skills (in user's repo) override by name.

_SYSTEM_SKILLS_DIR = Path(__file__).parent


def _load_system_skills() -> List[SkillDefinition]:
    """load system skills from yaml files alongside this module."""
    return load_skills_from_directory(str(_SYSTEM_SKILLS_DIR))


SYSTEM_SKILLS = _load_system_skills()


def get_all_skills(
    project_skills_dir: Optional[str] = None,
    include_system: bool = True,
) -> List[SkillDefinition]:
    """get all available skills: system + project-specific.

    project skills override system skills with the same name.
    """
    skills_by_name: Dict[str, SkillDefinition] = {}

    if include_system:
        for skill in SYSTEM_SKILLS:
            skills_by_name[skill.name] = skill

    if project_skills_dir:
        for skill in load_skills_from_directory(project_skills_dir):
            skills_by_name[skill.name] = skill  # project overrides system

    return list(skills_by_name.values())
