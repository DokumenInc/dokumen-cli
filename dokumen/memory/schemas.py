"""memory data types."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryOperation(Enum):
    """operations the LLM can choose during the update phase."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


@dataclass
class Memory:
    """single memory unit — a fact extracted from agent conversations."""

    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Memory":
        return cls(
            id=d["id"],
            content=d["content"],
            embedding=d.get("embedding"),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )
