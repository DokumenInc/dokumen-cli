"""
session memory — tier 1 of the three-tier memory system.

in-conversation working memory that tracks current test context,
discovered files, and intermediate findings. auto-summarizes when
it gets too long.

protocol-based (rule 2.6) so backing store is swappable.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# default threshold: summarize when entries exceed this count
DEFAULT_SUMMARIZE_THRESHOLD = 50
# max chars before triggering compression
DEFAULT_MAX_CHARS = 50_000


@dataclass
class SessionEntry:
    """single entry in session working memory."""

    content: str
    category: str = "general"  # general, tool_result, finding, decision, error
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "category": self.category,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionEntry":
        return cls(
            content=d["content"],
            category=d.get("category", "general"),
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )


class SessionSummarizer(Protocol):
    """protocol for summarizing session entries."""

    async def summarize(self, entries: List[SessionEntry]) -> str:
        """summarize a list of entries into a compact string."""
        ...


class DefaultSummarizer:
    """simple summarizer that concatenates entries with truncation."""

    async def summarize(self, entries: List[SessionEntry], max_chars: int = 5000) -> str:
        lines = []
        for e in entries:
            prefix = f"[{e.category}]" if e.category != "general" else ""
            lines.append(f"{prefix} {e.content}")

        combined = "\n".join(lines)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + f"\n... ({len(entries)} entries summarized)"
        return combined


class SessionMemory:
    """in-conversation working memory with auto-summarization.

    tracks entries by category. when entry count or total chars exceed
    thresholds, older entries are summarized into a compact block.

    usage:
        session = SessionMemory()
        session.add("found 3 relevant files", category="finding")
        session.add("read_file returned 500 lines", category="tool_result")

        # when building prompts, inject the session context
        context = session.get_context()
    """

    def __init__(
        self,
        summarizer: Optional[SessionSummarizer] = None,
        max_entries: int = DEFAULT_SUMMARIZE_THRESHOLD,
        max_chars: int = DEFAULT_MAX_CHARS,
    ):
        self._entries: List[SessionEntry] = []
        self._summaries: List[str] = []  # previous compacted summaries
        self._summarizer = summarizer or DefaultSummarizer()
        self._max_entries = max_entries
        self._max_chars = max_chars
        self._total_chars = 0

        logger.info(
            "session memory initialized",
            extra={"max_entries": max_entries, "max_chars": max_chars},
        )

    def add(
        self, content: str, category: str = "general", metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """add an entry to session memory."""
        entry = SessionEntry(
            content=content,
            category=category,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._total_chars += len(content)

        logger.debug(
            "session entry added",
            extra={
                "category": category,
                "entries": len(self._entries),
                "total_chars": self._total_chars,
            },
        )

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def total_chars(self) -> int:
        return self._total_chars

    @property
    def needs_compaction(self) -> bool:
        return len(self._entries) >= self._max_entries or self._total_chars >= self._max_chars

    async def compact(self) -> str:
        """compact older entries into a summary, keeping recent ones.

        returns the generated summary.
        """
        if len(self._entries) <= 5:
            return ""

        # keep the 10 most recent entries, summarize the rest
        keep_count = min(10, len(self._entries) // 2)
        to_summarize = self._entries[:-keep_count]
        to_keep = self._entries[-keep_count:]

        logger.info(
            "compacting session memory",
            extra={
                "summarizing": len(to_summarize),
                "keeping": len(to_keep),
            },
        )

        summary = await self._summarizer.summarize(to_summarize)
        self._summaries.append(summary)
        self._entries = to_keep
        self._total_chars = sum(len(e.content) for e in to_keep)

        return summary

    def get_context(self, max_chars: int = 10_000) -> str:
        """get the current session context for prompt injection.

        includes previous summaries + current entries.
        """
        parts = []

        if self._summaries:
            parts.append("## previous context (summarized)")
            for s in self._summaries:
                parts.append(s)
            parts.append("")

        if self._entries:
            parts.append("## current session")
            for e in self._entries:
                prefix = f"[{e.category}]" if e.category != "general" else ""
                parts.append(f"{prefix} {e.content}")

        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[-max_chars:]  # keep most recent
        return result

    def get_entries(self, category: Optional[str] = None) -> List[SessionEntry]:
        """get entries, optionally filtered by category."""
        if category is None:
            return list(self._entries)
        return [e for e in self._entries if e.category == category]

    def clear(self) -> None:
        """clear all entries and summaries."""
        self._entries.clear()
        self._summaries.clear()
        self._total_chars = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self._entries],
            "summaries": self._summaries,
            "total_chars": self._total_chars,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **kwargs) -> "SessionMemory":
        session = cls(**kwargs)
        session._entries = [SessionEntry.from_dict(e) for e in d.get("entries", [])]
        session._summaries = d.get("summaries", [])
        session._total_chars = d.get("total_chars", 0)
        return session
