"""Workspace instruction discovery used by scaffold SOP/skill references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SkillInfo:
    """A discovered reusable instruction file."""

    name: str
    description: str
    file_path: str
    content: str


class SkillLoader:
    """Load reusable instructions from conventional workspace directories."""

    def __init__(self, paths: Optional[list[str]] = None):
        self._paths = paths or ["sops", "skills", ".dokumen/sops", ".dokumen/skills"]

    def load_skills(self, base_dir: str = ".") -> list[SkillInfo]:
        """Discover markdown and YAML reusable instruction files below base_dir."""
        root = Path(base_dir)
        skills: list[SkillInfo] = []
        seen: set[str] = set()

        for relative in self._paths:
            directory = root / relative
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.suffix.lower() not in {".md", ".yaml", ".yml"}:
                    continue
                info = self._load_skill(path, root)
                if info and info.name not in seen:
                    skills.append(info)
                    seen.add(info.name)

        return skills

    def _load_skill(self, path: Path, root: Path) -> Optional[SkillInfo]:
        """Parse a single skill file."""
        try:
            content = path.read_text(encoding="utf-8")
            rel = str(path.relative_to(root))

            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(content) or {}
                name = data.get("name") or path.stem
                description = data.get("description") or ""
                body = data.get("content") or data.get("prompt") or content
                return SkillInfo(name=name, description=description, file_path=rel, content=body)

            name = path.parent.name if path.name.upper() == "SKILL.MD" else path.stem
            description = _first_heading_or_line(content)
            return SkillInfo(name=name, description=description, file_path=rel, content=content)
        except Exception:
            return None


def _first_heading_or_line(content: str) -> str:
    """Extract a compact description from markdown content."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return ""
