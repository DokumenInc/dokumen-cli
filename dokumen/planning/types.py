"""types for the planning system."""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PlanStatus(Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class PlanStep:
    """a single step in a plan."""
    id: str = field(default_factory=lambda: f"step-{uuid.uuid4().hex[:6]}")
    description: str = ""
    tools_needed: List[str] = field(default_factory=list)
    done_criteria: str = ""  # how to verify this step is done (sprint contract)
    status: PlanStatus = PlanStatus.DRAFT
    output: str = ""
    error: Optional[str] = None
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "tools_needed": self.tools_needed,
            "done_criteria": self.done_criteria,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlanStep":
        return cls(
            id=d.get("id", f"step-{uuid.uuid4().hex[:6]}"),
            description=d.get("description", ""),
            tools_needed=d.get("tools_needed", []),
            done_criteria=d.get("done_criteria", ""),
            status=PlanStatus(d.get("status", "draft")),
            output=d.get("output", ""),
            error=d.get("error"),
            order=d.get("order", 0),
        )


@dataclass
class Plan:
    """a structured execution plan.

    plans decompose complex tasks into ordered steps, each with
    clear done criteria (sprint contracts). this prevents the agent
    from trying to one-shot complex tasks.
    """
    id: str = field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    goal: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == PlanStatus.COMPLETED)

    @property
    def progress(self) -> float:
        """completion percentage 0.0-1.0."""
        if not self.steps:
            return 0.0
        return self.completed_steps / self.total_steps

    @property
    def current_step(self) -> Optional[PlanStep]:
        """get the next step that needs work."""
        for step in sorted(self.steps, key=lambda s: s.order):
            if step.status in (PlanStatus.DRAFT, PlanStatus.APPROVED, PlanStatus.IN_PROGRESS):
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Plan":
        return cls(
            id=d.get("id", f"plan-{uuid.uuid4().hex[:8]}"),
            goal=d.get("goal", ""),
            steps=[PlanStep.from_dict(s) for s in d.get("steps", [])],
            status=PlanStatus(d.get("status", "draft")),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            metadata=d.get("metadata", {}),
        )
