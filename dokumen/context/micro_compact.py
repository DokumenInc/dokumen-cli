"""
micro compactor — time-based clearing of verbose tool results.

older tool outputs get trimmed to just key findings. configurable
per-tool via max_tool_result_chars. this runs on individual tool
results, not on the full conversation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# default: tool results older than 5 minutes get truncated
DEFAULT_AGE_THRESHOLD = 300  # seconds
DEFAULT_TRUNCATE_TO = 500  # chars after truncation


@dataclass
class ToolResultEntry:
    """tracked tool result for micro-compaction."""

    tool_name: str
    content: str
    original_length: int
    timestamp: float = field(default_factory=time.time)
    truncated: bool = False

    @property
    def age(self) -> float:
        return time.time() - self.timestamp


class MicroCompactor:
    """micro-compacts old tool results to save context space.

    tracks tool results by timestamp. when asked, truncates old results
    to a short summary. different tools can have different truncation limits.

    usage:
        mc = MicroCompactor(age_threshold=300)
        mc.track("read_file", full_content)  # track when result arrives

        # later, when building context:
        compacted = mc.get_compacted_results()
    """

    def __init__(
        self,
        age_threshold: float = DEFAULT_AGE_THRESHOLD,
        default_truncate_to: int = DEFAULT_TRUNCATE_TO,
        per_tool_limits: Optional[Dict[str, int]] = None,
    ):
        self._entries: List[ToolResultEntry] = []
        self._age_threshold = age_threshold
        self._default_truncate_to = default_truncate_to
        self._per_tool = per_tool_limits or {}

        logger.info(
            "micro compactor initialized",
            extra={
                "age_threshold": age_threshold,
                "default_truncate_to": default_truncate_to,
            },
        )

    def track(self, tool_name: str, content: str) -> None:
        """track a new tool result."""
        self._entries.append(
            ToolResultEntry(
                tool_name=tool_name,
                content=content,
                original_length=len(content),
            )
        )

    def compact(self) -> int:
        """compact old tool results in place.

        returns number of entries truncated.
        """
        truncated = 0
        now = time.time()

        for entry in self._entries:
            if entry.truncated:
                continue
            age = now - entry.timestamp
            if age < self._age_threshold:
                continue

            limit = self._per_tool.get(entry.tool_name, self._default_truncate_to)
            if len(entry.content) > limit:
                entry.content = (
                    entry.content[:limit] + f"\n... [truncated from {entry.original_length} chars]"
                )
                entry.truncated = True
                truncated += 1

        if truncated:
            logger.info(
                "micro-compacted tool results",
                extra={"truncated": truncated, "total": len(self._entries)},
            )

        return truncated

    def get_results(self) -> List[ToolResultEntry]:
        """get all tracked results (may be truncated)."""
        return list(self._entries)

    def get_result(self, index: int) -> Optional[ToolResultEntry]:
        """get a specific result by index."""
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    @property
    def total_chars(self) -> int:
        return sum(len(e.content) for e in self._entries)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """clear all tracked results."""
        self._entries.clear()

    def stats(self) -> Dict[str, Any]:
        truncated = sum(1 for e in self._entries if e.truncated)
        return {
            "total_entries": len(self._entries),
            "truncated_entries": truncated,
            "total_chars": self.total_chars,
            "original_chars": sum(e.original_length for e in self._entries),
        }
