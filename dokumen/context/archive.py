"""
context archive — persists compacted turns so they're not lost forever.

when the compactor discards old turns, this module can save them to disk
(or memory for testing) before they're gone. useful for auditing,
debugging, and replaying sessions.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# default archive directory relative to cwd
DEFAULT_ARCHIVE_DIR = ".dokumen-cache/archives"


@dataclass
class ArchiveEntry:
    """a single compaction event saved to the archive."""

    session_id: str
    timestamp: float
    turns: List[Dict[str, Any]]  # full turn content at compaction time
    summary: str  # the summary that replaced these turns
    turn_count: int
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchiveEntry":
        return cls(**data)


@runtime_checkable
class ArchiveStore(Protocol):
    """protocol for archive backends — swappable per rule 2.6."""

    def save(
        self,
        session_id: str,
        turns: List[Dict[str, Any]],
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArchiveEntry:
        """persist a compaction event and return the saved entry."""
        ...

    def load(self, session_id: str) -> List[ArchiveEntry]:
        """load all archive entries for a given session."""
        ...

    def list_sessions(self) -> List[str]:
        """return all session ids that have archived entries."""
        ...


class FileArchiveStore:
    """writes archive entries to .dokumen-cache/archives/ as json files.

    each compaction event gets its own file named:
        {session_id}_{timestamp}.json
    """

    def __init__(self, archive_dir: str = DEFAULT_ARCHIVE_DIR):
        self._dir = Path(archive_dir)
        logger.info("file archive store initialized", extra={"archive_dir": str(self._dir)})

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        session_id: str,
        turns: List[Dict[str, Any]],
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArchiveEntry:
        """save a compaction event to disk."""
        self._ensure_dir()

        ts = time.time()
        token_estimate = sum(len(t.get("content", "")) // 4 for t in turns)

        entry = ArchiveEntry(
            session_id=session_id,
            timestamp=ts,
            turns=turns,
            summary=summary,
            turn_count=len(turns),
            token_estimate=token_estimate,
            metadata=metadata or {},
        )

        # safe filename — colons break windows paths, dots in session ids are fine
        filename = f"{session_id}_{ts:.6f}.json".replace(":", "-")
        path = self._dir / filename

        try:
            path.write_text(json.dumps(entry.to_dict(), indent=2), encoding="utf-8")
            logger.info(
                "archive entry saved",
                extra={
                    "session_id": session_id,
                    "path": str(path),
                    "turn_count": len(turns),
                    "token_estimate": token_estimate,
                },
            )
        except OSError as e:
            logger.error(
                "failed to write archive entry",
                extra={"path": str(path), "error": str(e)},
                exc_info=True,
            )
            raise

        return entry

    def load(self, session_id: str) -> List[ArchiveEntry]:
        """load all archive entries for a session, sorted by timestamp."""
        if not self._dir.exists():
            logger.debug(
                "archive dir does not exist, returning empty", extra={"dir": str(self._dir)}
            )
            return []

        # match files that start with the session_id prefix
        prefix = f"{session_id}_"
        entries: List[ArchiveEntry] = []

        for path in self._dir.glob("*.json"):
            if not path.name.startswith(prefix):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entries.append(ArchiveEntry.from_dict(data))
            except (OSError, json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    "skipping unreadable archive file",
                    extra={"path": str(path), "error": str(e)},
                )

        entries.sort(key=lambda e: e.timestamp)
        logger.debug(
            "loaded archive entries",
            extra={"session_id": session_id, "count": len(entries)},
        )
        return entries

    def list_sessions(self) -> List[str]:
        """return unique session ids found in the archive directory."""
        if not self._dir.exists():
            return []

        sessions = set()
        for path in self._dir.glob("*.json"):
            # filename is {session_id}_{timestamp}.json
            # session_id itself may contain underscores, so split from the right
            # using the timestamp pattern (digits + dot + digits before .json)
            name = path.stem  # strip .json
            # find the last underscore before what looks like a float timestamp
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                sessions.add(parts[0])

        result = sorted(sessions)
        logger.debug("listed archive sessions", extra={"count": len(result)})
        return result


class InMemoryArchiveStore:
    """in-memory archive store for testing — no disk i/o."""

    def __init__(self) -> None:
        self._entries: List[ArchiveEntry] = []
        logger.debug("in-memory archive store initialized")

    def save(
        self,
        session_id: str,
        turns: List[Dict[str, Any]],
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArchiveEntry:
        token_estimate = sum(len(t.get("content", "")) // 4 for t in turns)

        entry = ArchiveEntry(
            session_id=session_id,
            timestamp=time.time(),
            turns=turns,
            summary=summary,
            turn_count=len(turns),
            token_estimate=token_estimate,
            metadata=metadata or {},
        )

        self._entries.append(entry)
        logger.debug(
            "in-memory archive entry saved",
            extra={"session_id": session_id, "turn_count": len(turns)},
        )
        return entry

    def load(self, session_id: str) -> List[ArchiveEntry]:
        result = [e for e in self._entries if e.session_id == session_id]
        result.sort(key=lambda e: e.timestamp)
        return result

    def list_sessions(self) -> List[str]:
        return sorted({e.session_id for e in self._entries})
