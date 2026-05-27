"""types for the task system."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """task lifecycle states."""

    BLOCKED = "blocked"  # waiting on dependencies
    PENDING = "pending"  # ready to run
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskOutput:
    """output from a task execution step."""

    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskOutput":
        return cls(
            content=d["content"],
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Task:
    """a trackable subtask within a test run.

    tasks go through a lifecycle: pending -> in_progress -> completed/failed/cancelled.
    each task can have multiple output entries tracking progress.
    """

    id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None  # for subtask hierarchies
    depends_on: List[str] = field(default_factory=list)  # task ids this task is blocked by
    outputs: List[TaskOutput] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """whether this task is in a terminal state."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def is_blocked(self) -> bool:
        """whether this task is waiting on dependencies."""
        return self.status == TaskStatus.BLOCKED

    @property
    def duration(self) -> Optional[float]:
        """duration in seconds if completed."""
        if self.completed_at:
            return self.completed_at - self.created_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "depends_on": list(self.depends_on),
            "outputs": [o.to_dict() for o in self.outputs],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        return cls(
            id=d.get("id", f"task-{uuid.uuid4().hex[:8]}"),
            name=d.get("name", ""),
            description=d.get("description", ""),
            status=TaskStatus(d.get("status", "pending")),
            parent_id=d.get("parent_id"),
            depends_on=list(d.get("depends_on", [])),
            outputs=[TaskOutput.from_dict(o) for o in d.get("outputs", [])],
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            completed_at=d.get("completed_at"),
            error=d.get("error"),
            metadata=d.get("metadata", {}),
        )
