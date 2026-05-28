"""Local explore result types and response parsing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

VALID_EXPLORE_TYPES = {"docs", "code", "both"}


@dataclass
class FileDiscovery:
    """A file discovered during the explore phase."""

    path: str
    summary: str = ""
    relevance: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the discovery for JSON output."""
        return {
            "path": self.path,
            "summary": self.summary,
            "relevance": self.relevance,
        }


@dataclass
class ExploreToolRecord:
    """A compact record of an explore tool call."""

    name: str
    input: dict[str, Any] = field(default_factory=dict)
    output: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool call for diagnostics."""
        return {
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class ExploreResult:
    """Result from the explore phase."""

    files: list[FileDiscovery]
    duration: float
    tool_calls_count: int
    success: bool
    error: Optional[str] = None
    summary: str = ""
    tool_history: list[Any] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    model: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for CLI JSON output."""
        return {
            "success": self.success,
            "files": [f.to_dict() for f in self.files],
            "duration": self.duration,
            "tool_calls_count": self.tool_calls_count,
            "error": self.error,
            "summary": self.summary,
            "tool_history": [
                item.to_dict() if hasattr(item, "to_dict") else item for item in self.tool_history
            ],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "model": self.model,
        }

    def to_context_block(self) -> str:
        """Render discovered context for executor prompts."""
        if not self.summary and not self.files:
            return ""

        lines = ["## Pre-discovered Documentation"]
        if self.summary:
            lines.extend(["", self.summary.strip()])

        if self.files:
            lines.extend(["", "Relevant files:"])
            for file in self.files:
                suffix = f" - {file.summary}" if file.summary else ""
                lines.append(f"- {file.path}{suffix}")

        return "\n".join(lines)


_PATH_RE = re.compile(
    r"(?<![\w./-])"
    r"((?:\.{1,2}/)?[\w@./-]+?\."
    r"(?:md|mdx|txt|rst|yaml|yml|json|py|js|jsx|ts|tsx|html|css|pdf|csv|toml|ini|sh|sql))"
    r"(?![\w/-])",
    re.IGNORECASE,
)


def extract_paths_from_response(content: str) -> tuple[list[FileDiscovery], str]:
    """Extract file paths from natural-language explore output."""
    discoveries: list[FileDiscovery] = []
    seen: set[str] = set()

    for match in _PATH_RE.finditer(content):
        path = match.group(1).strip("`'\".,;:)")
        if path in seen:
            continue
        seen.add(path)
        summary = _line_for_offset(content, match.start()).strip()
        discoveries.append(
            FileDiscovery(
                path=path,
                summary=summary,
                relevance=max(0.1, 1.0 - (len(discoveries) * 0.05)),
            )
        )

    return discoveries, content.strip()


def _line_for_offset(content: str, offset: int) -> str:
    """Return the source line that contains an offset."""
    start = content.rfind("\n", 0, offset) + 1
    end = content.find("\n", offset)
    if end == -1:
        end = len(content)
    return content[start:end]
