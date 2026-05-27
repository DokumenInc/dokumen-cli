"""
plan manager — create, update, and track execution plans.

plans decompose complex tasks into discrete steps with done criteria
(sprint contracts). this addresses the common failure mode of agents
trying to one-shot complex tasks.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import Plan, PlanStatus, PlanStep

logger = logging.getLogger(__name__)


class PlanManager:
    """manages execution plans.

    usage:
        pm = PlanManager()

        # create a plan
        plan = pm.create("validate all api docs", steps=[
            {"description": "list all api doc files", "done_criteria": "file list obtained"},
            {"description": "check each endpoint doc", "done_criteria": "all endpoints verified"},
        ])

        # execute step by step
        pm.start_step(plan.id, plan.steps[0].id)
        pm.complete_step(plan.id, plan.steps[0].id, output="found 12 files")
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._plans: Dict[str, Plan] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def create(
        self,
        goal: str,
        steps: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        """create a new plan.

        args:
            goal: the main goal
            steps: list of step dicts with {description, tools_needed, done_criteria}
            metadata: optional metadata
        """
        plan_steps = []
        for i, step_data in enumerate(steps or []):
            plan_steps.append(PlanStep(
                description=step_data.get("description", ""),
                tools_needed=step_data.get("tools_needed", []),
                done_criteria=step_data.get("done_criteria", ""),
                order=i,
            ))

        plan = Plan(
            goal=goal,
            steps=plan_steps,
            metadata=metadata or {},
        )
        self._plans[plan.id] = plan
        self._persist(plan)

        logger.info(
            "plan created",
            extra={"plan_id": plan.id, "goal": goal[:100], "steps": len(plan_steps)},
        )
        return plan

    def get(self, plan_id: str) -> Optional[Plan]:
        return self._plans.get(plan_id)

    def list(self, status: Optional[PlanStatus] = None) -> List[Plan]:
        plans = list(self._plans.values())
        if status is not None:
            plans = [p for p in plans if p.status == status]
        return sorted(plans, key=lambda p: p.created_at, reverse=True)

    def approve(self, plan_id: str) -> Optional[Plan]:
        """approve a draft plan for execution."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return None
        if plan.status != PlanStatus.DRAFT:
            logger.warning("can only approve draft plans", extra={"plan_id": plan_id, "status": plan.status.value})
            return plan

        plan.status = PlanStatus.APPROVED
        plan.updated_at = time.time()
        for step in plan.steps:
            step.status = PlanStatus.APPROVED
        self._persist(plan)

        logger.info("plan approved", extra={"plan_id": plan_id})
        return plan

    def start_step(self, plan_id: str, step_id: str) -> Optional[PlanStep]:
        """mark a step as in progress."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return None

        step = self._find_step(plan, step_id)
        if step is None:
            return None

        step.status = PlanStatus.IN_PROGRESS
        plan.status = PlanStatus.IN_PROGRESS
        plan.updated_at = time.time()
        self._persist(plan)

        logger.info("step started", extra={"plan_id": plan_id, "step_id": step_id})
        return step

    def complete_step(self, plan_id: str, step_id: str, output: str = "") -> Optional[PlanStep]:
        """mark a step as completed."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return None

        step = self._find_step(plan, step_id)
        if step is None:
            return None

        step.status = PlanStatus.COMPLETED
        step.output = output
        plan.updated_at = time.time()

        # check if all steps are done
        if plan.completed_steps == plan.total_steps:
            plan.status = PlanStatus.COMPLETED
            logger.info("plan completed", extra={"plan_id": plan_id})

        self._persist(plan)
        logger.info("step completed", extra={"plan_id": plan_id, "step_id": step_id})
        return step

    def fail_step(self, plan_id: str, step_id: str, error: str = "") -> Optional[PlanStep]:
        """mark a step as failed."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return None

        step = self._find_step(plan, step_id)
        if step is None:
            return None

        step.status = PlanStatus.FAILED
        step.error = error
        plan.status = PlanStatus.FAILED
        plan.updated_at = time.time()
        self._persist(plan)

        logger.info("step failed", extra={"plan_id": plan_id, "step_id": step_id, "error": error})
        return step

    def add_step(self, plan_id: str, description: str, done_criteria: str = "", tools_needed: Optional[List[str]] = None) -> Optional[PlanStep]:
        """add a new step to an existing plan."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return None

        order = max((s.order for s in plan.steps), default=-1) + 1
        step = PlanStep(
            description=description,
            done_criteria=done_criteria,
            tools_needed=tools_needed or [],
            order=order,
            status=plan.status if plan.status in (PlanStatus.APPROVED, PlanStatus.IN_PROGRESS) else PlanStatus.DRAFT,
        )
        plan.steps.append(step)
        plan.updated_at = time.time()
        self._persist(plan)

        logger.info("step added", extra={"plan_id": plan_id, "step_id": step.id})
        return step

    def delete(self, plan_id: str) -> bool:
        if plan_id in self._plans:
            del self._plans[plan_id]
            if self._persist_dir:
                filepath = self._persist_dir / f"{plan_id}.json"
                if filepath.exists():
                    filepath.unlink()
            return True
        return False

    def get_progress_summary(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """get a summary of plan progress for handoff artifacts."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return None

        return {
            "plan_id": plan.id,
            "goal": plan.goal,
            "status": plan.status.value,
            "progress": round(plan.progress * 100, 1),
            "total_steps": plan.total_steps,
            "completed_steps": plan.completed_steps,
            "current_step": plan.current_step.description if plan.current_step else None,
            "completed_outputs": [
                {"step": s.description, "output": s.output[:200]}
                for s in plan.steps if s.status == PlanStatus.COMPLETED
            ],
        }

    def _find_step(self, plan: Plan, step_id: str) -> Optional[PlanStep]:
        for step in plan.steps:
            if step.id == step_id:
                return step
        return None

    def _persist(self, plan: Plan) -> None:
        if self._persist_dir:
            filepath = self._persist_dir / f"{plan.id}.json"
            filepath.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    def _load_from_disk(self) -> None:
        for filepath in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                plan = Plan.from_dict(data)
                self._plans[plan.id] = plan
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning("failed to load plan", extra={"path": str(filepath), "error": str(e)})
