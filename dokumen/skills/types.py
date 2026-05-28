"""types for the skill framework."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SkillCategory(Enum):
    """categories of learned skills."""

    EXECUTOR = "executor"  # tips for executor agents
    JUDGE = "judge"  # tips for judge agents
    TOOL_USE = "tool_use"  # tips about tool usage patterns
    DOMAIN = "domain"  # domain-specific knowledge
    ERROR_RECOVERY = "error_recovery"  # how to recover from common errors


@dataclass
class SkillEntry:
    """a single learned skill/tip.

    skills are extracted from test run failures and successes.
    they encode reusable knowledge that improves future runs.

    example:
        SkillEntry(
            id="sk-abc123",
            content="when checking OAuth docs, always verify both access_token and refresh_token flows",
            category=SkillCategory.DOMAIN,
            source_test="oauth-validation",
            source_judge="accuracy",
            tags=["oauth", "auth"],
            times_used=3,
            times_helpful=2,
        )
    """

    id: str = field(default_factory=lambda: f"sk-{uuid.uuid4().hex[:12]}")
    content: str = ""
    category: SkillCategory = SkillCategory.EXECUTOR
    # where this skill was learned from
    source_test: str = ""
    source_judge: str = ""
    # for retrieval
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    # effectiveness tracking
    times_used: int = 0
    times_helpful: int = 0  # times injected and test subsequently passed
    # metadata
    client_id: str = ""  # per-client isolation
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def effectiveness(self) -> float:
        """fraction of times this skill led to improvement."""
        if self.times_used == 0:
            return 0.0
        return self.times_helpful / self.times_used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category.value,
            "source_test": self.source_test,
            "source_judge": self.source_judge,
            "embedding": self.embedding,
            "tags": self.tags,
            "times_used": self.times_used,
            "times_helpful": self.times_helpful,
            "client_id": self.client_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillEntry":
        return cls(
            id=d.get("id", f"sk-{uuid.uuid4().hex[:12]}"),
            content=d.get("content", ""),
            category=SkillCategory(d.get("category", "executor")),
            source_test=d.get("source_test", ""),
            source_judge=d.get("source_judge", ""),
            embedding=d.get("embedding"),
            tags=d.get("tags", []),
            times_used=d.get("times_used", 0),
            times_helpful=d.get("times_helpful", 0),
            client_id=d.get("client_id", ""),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            metadata=d.get("metadata", {}),
        )
