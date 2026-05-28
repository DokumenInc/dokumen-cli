"""
context compactor — auto-compact for long agent conversations.

when a conversation nears the token limit, older turns are summarized
into a compact block. recent turns stay intact. critical context
(system prompt, discovered files, active skills) is re-injected
after compaction.

inspired by anthropic's harness design research:
- context resets with structured handoff > compaction alone
- but compaction is still useful within a single session
- for cross-session work, use progress artifacts instead
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .archive import ArchiveStore

logger = logging.getLogger(__name__)

# rough estimate: 1 token ≈ 4 chars
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 180_000  # ~720k chars, leave headroom for 200k context
DEFAULT_COMPACT_THRESHOLD = 0.75  # compact when 75% of budget used
DEFAULT_KEEP_RECENT = 10  # keep last N turns intact


@dataclass
class Turn:
    """a single conversation turn."""

    role: str  # user, assistant, tool_result
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // CHARS_PER_TOKEN


@dataclass
class CompactionResult:
    """result of a compaction operation."""

    summary: str
    turns_removed: int
    turns_kept: int
    tokens_before: int
    tokens_after: int
    compaction_time: float = 0.0

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "turns_removed": self.turns_removed,
            "turns_kept": self.turns_kept,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "compaction_time": self.compaction_time,
        }


class Summarizer(Protocol):
    """protocol for summarizing turns into a compact string."""

    async def summarize(self, turns: List[Turn]) -> str: ...


class RuleSummarizer:
    """simple rule-based summarizer that extracts key points.

    for production, swap with an llm-based summarizer.
    """

    def __init__(self, max_summary_chars: int = 5000):
        self._max_chars = max_summary_chars

    async def summarize(self, turns: List[Turn]) -> str:
        points = []

        for turn in turns:
            if turn.role == "assistant":
                # keep first line of assistant responses as key decisions
                first_line = turn.content.strip().split("\n")[0][:200]
                if first_line:
                    points.append(f"- {first_line}")

            elif turn.role == "tool_result":
                # keep tool name + short result
                tool_name = turn.metadata.get("tool", "tool")
                result_preview = turn.content[:100].replace("\n", " ")
                points.append(f"- [{tool_name}]: {result_preview}")

        combined = "\n".join(points)
        if len(combined) > self._max_chars:
            combined = combined[: self._max_chars] + "\n... (truncated)"

        return f"## compacted context ({len(turns)} turns)\n\n{combined}"


class ContextCompactor:
    """manages conversation context with auto-compaction.

    tracks turns and their token estimates. when total tokens exceed
    the threshold, older turns are summarized and replaced with a
    compact summary block.

    usage:
        compactor = ContextCompactor(max_tokens=100000)
        compactor.add_turn("user", "read the docs")
        compactor.add_turn("assistant", "i found 3 files...")

        if compactor.needs_compaction:
            result = await compactor.compact()

        # get current turns for API call
        messages = compactor.get_messages()
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        compact_threshold: float = DEFAULT_COMPACT_THRESHOLD,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        summarizer: Optional[Summarizer] = None,
        system_prompt: str = "",
        reinject_context: Optional[List[str]] = None,
        archive: Optional[ArchiveStore] = None,
        session_id: str = "",
    ):
        self._turns: List[Turn] = []
        self._max_tokens = max_tokens
        self._threshold = compact_threshold
        self._keep_recent = keep_recent
        self._summarizer = summarizer or RuleSummarizer()
        self._system_prompt = system_prompt
        self._reinject = reinject_context or []  # strings to re-inject after compaction
        self._archive = archive
        self._session_id = session_id
        self._compaction_count = 0
        self._total_tokens_processed = 0

        logger.info(
            "context compactor initialized",
            extra={
                "max_tokens": max_tokens,
                "threshold": compact_threshold,
                "keep_recent": keep_recent,
            },
        )

    def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """add a turn to the conversation."""
        turn = Turn(role=role, content=content, metadata=metadata or {})
        self._turns.append(turn)
        self._total_tokens_processed += turn.token_estimate

    @property
    def current_tokens(self) -> int:
        return sum(t.token_estimate for t in self._turns)

    @property
    def token_budget(self) -> int:
        return self._max_tokens

    @property
    def needs_compaction(self) -> bool:
        return self.current_tokens >= int(self._max_tokens * self._threshold)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    async def compact(self) -> CompactionResult:
        """compact older turns into a summary.

        keeps the most recent turns intact and summarizes the rest.
        re-injects critical context after compaction.
        """
        start = time.time()
        tokens_before = self.current_tokens

        if len(self._turns) <= self._keep_recent:
            return CompactionResult(
                summary="",
                turns_removed=0,
                turns_kept=len(self._turns),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        to_compact = self._turns[: -self._keep_recent]
        to_keep = self._turns[-self._keep_recent :]

        logger.info(
            "starting compaction",
            extra={
                "compacting": len(to_compact),
                "keeping": len(to_keep),
                "tokens_before": tokens_before,
            },
        )

        summary = await self._summarizer.summarize(to_compact)

        # archive the turns being discarded before they're gone
        if self._archive is not None:
            try:
                turn_dicts = [
                    {
                        "role": t.role,
                        "content": t.content,
                        "timestamp": t.timestamp,
                        "metadata": t.metadata,
                    }
                    for t in to_compact
                ]
                self._archive.save(
                    session_id=self._session_id,
                    turns=turn_dicts,
                    summary=summary,
                    metadata={
                        "compaction_number": self._compaction_count + 1,
                        "tokens_before": tokens_before,
                    },
                )
                logger.info(
                    "compacted turns archived",
                    extra={
                        "session_id": self._session_id,
                        "turn_count": len(to_compact),
                    },
                )
            except Exception as e:
                # archiving is best-effort — never block compaction
                logger.warning(
                    "archive save failed, continuing without archive",
                    extra={"error": str(e)},
                    exc_info=True,
                )

        # build new turn list: summary + re-injected context + recent turns
        new_turns = [Turn(role="system", content=summary, metadata={"compacted": True})]

        for ctx in self._reinject:
            new_turns.append(Turn(role="system", content=ctx, metadata={"reinjected": True}))

        new_turns.extend(to_keep)

        self._turns = new_turns
        self._compaction_count += 1
        tokens_after = self.current_tokens
        elapsed = time.time() - start

        logger.info(
            "compaction complete",
            extra={
                "turns_removed": len(to_compact),
                "turns_kept": len(to_keep),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": tokens_before - tokens_after,
                "compaction_number": self._compaction_count,
            },
        )

        return CompactionResult(
            summary=summary,
            turns_removed=len(to_compact),
            turns_kept=len(to_keep) + 1 + len(self._reinject),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compaction_time=elapsed,
        )

    def get_messages(self) -> List[Dict[str, str]]:
        """get current conversation as a list of message dicts.

        suitable for passing to an LLM API.
        """
        messages = []
        for turn in self._turns:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def get_turns(self) -> List[Turn]:
        """get raw Turn objects."""
        return list(self._turns)

    def clear(self) -> None:
        """clear all turns."""
        self._turns.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "turn_count": len(self._turns),
            "current_tokens": self.current_tokens,
            "max_tokens": self._max_tokens,
            "usage_pct": round(self.current_tokens / self._max_tokens * 100, 1),
            "compactions": self._compaction_count,
            "total_tokens_processed": self._total_tokens_processed,
        }
