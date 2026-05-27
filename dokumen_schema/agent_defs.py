"""Agent definition loading for built-in and workspace agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import BrowserScaffoldConfig


class ResearchConfig(BaseModel):
    """Research-agent defaults."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True


class AgentDefinition(BaseModel):
    """Reusable agent defaults loaded from YAML or built-ins."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    browser: Optional[BrowserScaffoldConfig] = None
    research: Optional[ResearchConfig] = None


_BUILTINS: dict[str, AgentDefinition] = {
    "general": AgentDefinition(
        name="general",
        description="Default documentation executor.",
        system_prompt=(
            "You are a Dokumen executor agent. Use the available documentation "
            "and tools to complete the user's task accurately."
        ),
        tools=["read_file", "glob", "search_file_content"],
        capabilities=["standard"],
    ),
    "browser-tester": AgentDefinition(
        name="browser-tester",
        description="Browser-oriented skill test executor.",
        system_prompt=(
            "You are a Dokumen browser test executor. Follow the documentation, "
            "interact with the browser, and report observable results."
        ),
        tools=["browser_navigate", "browser_click", "browser_type", "read_file"],
        capabilities=["browser"],
        browser=BrowserScaffoldConfig(headless=False, viewport="1512x982"),
    ),
    "researcher": AgentDefinition(
        name="researcher",
        description="Research executor with web search.",
        system_prompt=(
            "You are a Dokumen research executor. Answer with grounded claims "
            "and cite the sources you use."
        ),
        tools=["web_search", "web_fetch", "read_file"],
        capabilities=["research"],
        research=ResearchConfig(enabled=True),
    ),
    "code-reviewer": AgentDefinition(
        name="code-reviewer",
        description="Code-aware documentation review executor.",
        system_prompt=(
            "You are a Dokumen code review executor. Cross-reference docs with "
            "the linked source code and report implementation-grounded findings."
        ),
        tools=["code_read_file", "code_search", "code_glob"],
        capabilities=["code"],
    ),
}


def list_agents(user_dirs: Optional[list[Path]] = None) -> list[str]:
    """List built-in and user-defined agent names."""
    names = set(_BUILTINS)
    for directory in user_dirs or []:
        path = Path(directory)
        if path.exists():
            names.update(file.stem for file in path.glob("*.yaml"))
            names.update(file.stem for file in path.glob("*.yml"))
    return sorted(names)


def load_agent(name: str, user_dirs: Optional[list[Path]] = None) -> Optional[AgentDefinition]:
    """Load an agent by name from user directories, then built-ins."""
    for directory in user_dirs or []:
        path = Path(directory)
        for suffix in (".yaml", ".yml"):
            candidate = path / f"{name}{suffix}"
            if candidate.exists():
                return _load_agent_file(candidate)
    return _BUILTINS.get(name)


def _load_agent_file(path: Path) -> Optional[AgentDefinition]:
    """Parse one agent YAML file."""
    try:
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("name", path.stem)
        return AgentDefinition(**data)
    except Exception:
        return None
