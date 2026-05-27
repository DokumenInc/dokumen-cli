"""
memdir — tier 3 persistent file-based memory store.

stores memories as individual markdown files with yaml frontmatter.
supports typed memories (user, feedback, project, reference) and
maintains an index file (MEMORY.md) for quick lookup.

inspired by file-based memory architectures — but built from scratch
for dokumen's needs.
"""
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .base import MemoryStore
from .schemas import Memory

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """types of persistent memories."""
    USER = "user"           # about the user/client
    FEEDBACK = "feedback"   # corrections and confirmations
    PROJECT = "project"     # ongoing work, goals, deadlines
    REFERENCE = "reference" # pointers to external resources


@dataclass
class MemdirEntry:
    """a single memdir entry with frontmatter metadata."""
    id: str
    name: str
    description: str
    memory_type: MemoryType
    content: str
    filename: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_markdown(self) -> str:
        """render as markdown with yaml frontmatter."""
        frontmatter = {
            "name": self.name,
            "description": self.description,
            "type": self.memory_type.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        fm_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip()
        return f"---\n{fm_str}\n---\n\n{self.content}\n"

    @classmethod
    def from_markdown(cls, text: str, filename: str) -> Optional["MemdirEntry"]:
        """parse a memdir entry from markdown with frontmatter."""
        match = re.match(r"^---\n(.*?)\n---\n\n?(.*)", text, re.DOTALL)
        if not match:
            return None

        try:
            frontmatter = yaml.safe_load(match.group(1))
            content = match.group(2).strip()
        except yaml.YAMLError:
            return None

        if not isinstance(frontmatter, dict):
            return None

        try:
            mem_type = MemoryType(frontmatter.get("type", "project"))
        except ValueError:
            mem_type = MemoryType.PROJECT

        return cls(
            id=os.path.splitext(filename)[0],
            name=frontmatter.get("name", filename),
            description=frontmatter.get("description", ""),
            memory_type=mem_type,
            content=content,
            filename=filename,
            created_at=frontmatter.get("created_at", time.time()),
            updated_at=frontmatter.get("updated_at", time.time()),
        )

    def to_memory(self) -> Memory:
        """convert to base Memory for compatibility."""
        return Memory(
            id=self.id,
            content=self.content,
            metadata={
                "name": self.name,
                "description": self.description,
                "type": self.memory_type.value,
                "filename": self.filename,
            },
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class MemdirStore:
    """file-based persistent memory store.

    each memory is a markdown file with yaml frontmatter in a directory.
    an index file (MEMORY.md) provides a quick overview.

    implements the MemoryStore protocol for compatibility.

    usage:
        store = MemdirStore("/path/to/memory/")
        store.save(MemdirEntry(...))
        entries = store.load_all()
        store.update_index()
    """

    INDEX_FILENAME = "MEMORY.md"

    def __init__(self, directory: str):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "memdir store initialized",
            extra={"directory": str(self._dir)},
        )

    @property
    def directory(self) -> str:
        return str(self._dir)

    def save(self, entry: MemdirEntry) -> str:
        """save a memdir entry to disk.

        returns the file path.
        """
        filepath = self._dir / entry.filename
        filepath.write_text(entry.to_markdown(), encoding="utf-8")

        logger.info(
            "memdir entry saved",
            extra={"id": entry.id, "filename": entry.filename, "type": entry.memory_type.value},
        )

        self.update_index()
        return str(filepath)

    def load(self, filename: str) -> Optional[MemdirEntry]:
        """load a single entry by filename."""
        filepath = self._dir / filename
        if not filepath.exists():
            return None

        try:
            text = filepath.read_text(encoding="utf-8")
            return MemdirEntry.from_markdown(text, filename)
        except (IOError, OSError) as e:
            logger.warning("failed to load memdir entry", extra={"filename": filename, "error": str(e)})
            return None

    def load_all(self) -> List[MemdirEntry]:
        """load all memdir entries from disk."""
        entries = []
        for filepath in sorted(self._dir.glob("*.md")):
            if filepath.name == self.INDEX_FILENAME:
                continue
            entry = self.load(filepath.name)
            if entry:
                entries.append(entry)

        logger.debug("loaded memdir entries", extra={"count": len(entries)})
        return entries

    def delete(self, filename: str) -> bool:
        """delete a memdir entry."""
        filepath = self._dir / filename
        if filepath.exists():
            filepath.unlink()
            logger.info("memdir entry deleted", extra={"filename": filename})
            self.update_index()
            return True
        return False

    def find_by_type(self, memory_type: MemoryType) -> List[MemdirEntry]:
        """find all entries of a given type."""
        return [e for e in self.load_all() if e.memory_type == memory_type]

    def search_by_content(self, query: str, max_results: int = 10) -> List[MemdirEntry]:
        """simple text search across all entries.

        for production use, use embedding-based search via MemoryStore protocol.
        """
        query_lower = query.lower()
        scored = []
        for entry in self.load_all():
            searchable = f"{entry.name} {entry.description} {entry.content}".lower()
            # simple keyword overlap score
            query_words = set(query_lower.split())
            match_count = sum(1 for w in query_words if w in searchable)
            if match_count > 0:
                scored.append((entry, match_count / len(query_words)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:max_results]]

    def update_index(self) -> None:
        """regenerate the MEMORY.md index file."""
        entries = self.load_all()
        lines = ["# Memory Index\n"]

        # group by type
        by_type: Dict[str, List[MemdirEntry]] = {}
        for entry in entries:
            key = entry.memory_type.value
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(entry)

        for type_name in ["user", "feedback", "project", "reference"]:
            type_entries = by_type.get(type_name, [])
            if not type_entries:
                continue
            lines.append(f"\n## {type_name}\n")
            for entry in type_entries:
                desc = entry.description[:100] if entry.description else entry.name
                lines.append(f"- [{entry.name}]({entry.filename}) — {desc}")

        index_path = self._dir / self.INDEX_FILENAME
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        logger.debug("memdir index updated", extra={"entries": len(entries)})

    # ── MemoryStore protocol compatibility ──

    def add(self, memory: Memory) -> None:
        """add a Memory via the MemoryStore protocol."""
        mem_type = MemoryType(memory.metadata.get("type", "project"))
        name = memory.metadata.get("name", memory.id)
        desc = memory.metadata.get("description", memory.content[:100])
        filename = memory.metadata.get("filename", f"{memory.id}.md")

        entry = MemdirEntry(
            id=memory.id,
            name=name,
            description=desc,
            memory_type=mem_type,
            content=memory.content,
            filename=filename,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )
        self.save(entry)

    def update(self, memory_id: str, content: str, embedding=None) -> None:
        """update a memory's content."""
        # find by id
        for entry in self.load_all():
            if entry.id == memory_id:
                entry.content = content
                entry.updated_at = time.time()
                self.save(entry)
                return

    def get_all(self) -> List[Memory]:
        """return all memories via the MemoryStore protocol."""
        return [e.to_memory() for e in self.load_all()]

    def search(self, query_embedding, top_k=10, threshold=0.0):
        """embedding search — requires external embedding provider."""
        # for memdir, text search is the fallback
        # full embedding search should be done at a higher level
        logger.warning("memdir search called with embeddings — use text search or wrap with embedding provider")
        return []
