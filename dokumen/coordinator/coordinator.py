"""
coordinator agent — orchestrates parallel workers with DAG scheduling.

the coordinator:
1. auto-decomposes a goal into tasks with dependencies (or uses a pre-built plan)
2. resolves execution order via topological sort
3. spawns worker agents in parallel (respecting dependency ordering)
4. workers communicate via message bus + shared memory
5. collects results and synthesizes findings

follows the planner -> generator -> evaluator pattern from
anthropic's harness design research.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from .messaging import MessageBus
from .shared_memory import SharedMemory
from .types import CoordinatorPlan, WorkerResult, WorkerStatus, WorkerTask
from .worker import WorkerAgent

logger = logging.getLogger(__name__)


# ── auto-decomposition prompt ──

DECOMPOSE_SYSTEM = """you are a task decomposition agent. given a goal and available workers, break the goal into concrete tasks.

each task needs: title, description, assignee (worker name), depends_on (list of titles that must complete first, or empty list).

split work so independent tasks can run in parallel. keep the number of tasks reasonable (2-6)."""


# json schema for structured output — guarantees parseable response
DECOMPOSE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "assignee": {"type": "string"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["title", "description", "assignee", "depends_on"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

SYNTHESIS_SYSTEM = """you are a synthesis agent. given the outputs from multiple workers who each completed
a subtask, produce a unified result that combines their findings coherently.
be concise. preserve all important findings. note any contradictions between workers."""


def _parse_task_specs(text: str) -> List[Dict[str, Any]]:
    """parse task specs from LLM output as YAML. falls back to JSON."""
    import yaml

    # strip markdown fences if present
    clean = text.strip()
    fenced = re.search(r"```(?:ya?ml|json)?\s*\n?(.*?)```", clean, re.DOTALL)
    if fenced:
        clean = fenced.group(1).strip()

    # try YAML first
    try:
        parsed = yaml.safe_load(clean)
        if isinstance(parsed, list) and parsed:
            return parsed
    except Exception:
        pass

    # try JSON fallback
    try:
        start = clean.find("[")
        end = clean.rfind("]")
        if start >= 0 and end > start:
            parsed = json.loads(clean[start : end + 1])
            if isinstance(parsed, list):
                return parsed
    except Exception:
        pass

    logger.warning(
        "_parse_task_specs: could not parse tasks",
        extra={"text_length": len(text), "text_preview": text[:300]},
    )
    return []


class CoordinatorAgent:
    """orchestrates multiple worker agents with DAG scheduling.

    usage:
        coordinator = CoordinatorAgent(provider=provider)

        # option 1: provide a pre-built plan
        plan = CoordinatorPlan(
            main_goal="validate all docs",
            worker_tasks=[
                WorkerTask(name="check-api", goal="validate api docs"),
                WorkerTask(name="check-guides", goal="validate guide docs", depends_on=["check-api"]),
            ],
        )
        results = await coordinator.execute_plan(plan)

        # option 2: auto-decompose via LLM
        result = await coordinator.run("validate all documentation files")
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        tools_config: Optional[Any] = None,
        max_workers: int = 5,
        default_timeout: float = 60.0,
        synthesis_strategy: str = "merge",
        decompose_timeout: float = 60.0,
        decompose_model: Optional[str] = None,
        executor_mode: str = "sdk",
        base_dir: str = ".",
        worker_model: Optional[str] = None,
    ):
        self._provider = provider
        self._tools_config = tools_config
        self._max_workers = max_workers
        self._default_timeout = default_timeout
        self._synthesis_strategy = synthesis_strategy
        self._decompose_timeout = decompose_timeout
        self._decompose_model = decompose_model
        self._executor_mode = executor_mode
        self._base_dir = base_dir
        self._worker_model = worker_model
        self._bus = MessageBus()
        self._shared_memory = SharedMemory()

        logger.info(
            "coordinator initialized",
            extra={"max_workers": max_workers, "default_timeout": default_timeout},
        )

    @property
    def bus(self) -> MessageBus:
        return self._bus

    @property
    def shared_memory(self) -> SharedMemory:
        return self._shared_memory

    async def auto_decompose(
        self,
        goal: str,
        worker_names: List[str],
    ) -> CoordinatorPlan:
        """use LLM to decompose a goal into tasks with dependencies.

        falls back to one-task-per-worker if parsing fails.
        """
        if self._provider is None:
            # no LLM — create one task per worker
            logger.info("auto_decompose: no provider, creating one task per worker")
            return self._fallback_plan(goal, worker_names)

        roster = ", ".join(worker_names) if worker_names else "worker-0"
        prompt = f"goal: {goal}\n\navailable workers: {roster}\n\ndecompose this goal into tasks."

        try:
            # call provider for decomposition — short timeout, this is just planning
            # temporarily swap model if decompose_model is set
            original_model = getattr(self._provider, "model", None)
            if self._decompose_model:
                self._provider.model = self._decompose_model

            result = await asyncio.wait_for(
                self._provider.complete(
                    system_prompt=DECOMPOSE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={
                        "format": {
                            "type": "json_schema",
                            "schema": DECOMPOSE_OUTPUT_SCHEMA,
                        }
                    },
                ),
                timeout=self._decompose_timeout,
            )

            # restore original model
            if self._decompose_model and original_model is not None:
                self._provider.model = original_model

            # extract text from provider response
            if isinstance(result, dict):
                response_text = result.get("content", "") or result.get("text", "") or str(result)
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)

            logger.info(
                "auto_decompose: got response",
                extra={
                    "response_length": len(response_text),
                    "response_preview": response_text[:200],
                },
            )

            # structured output guarantees valid JSON — parse directly
            parsed = json.loads(response_text)
            specs = (
                parsed.get("tasks", [])
                if isinstance(parsed, dict)
                else _parse_task_specs(response_text)
            )

            if not specs:
                logger.warning("auto_decompose: failed to parse tasks, using fallback")
                return self._fallback_plan(goal, worker_names)

            # resolve title-based dependencies to task objects
            tasks = []
            title_to_task: Dict[str, WorkerTask] = {}

            # first pass: create all tasks
            for spec in specs:
                task = WorkerTask(
                    name=spec.get("assignee", worker_names[0] if worker_names else "worker-0"),
                    goal=spec.get("description", spec.get("title", "")),
                    tools=[],
                    context=f"task: {spec.get('title', '')}",
                    timeout=self._default_timeout,
                    metadata={"title": spec.get("title", "")},
                )
                title_to_task[spec.get("title", "")] = task
                tasks.append((spec, task))

            # second pass: resolve depends_on by title
            for spec, task in tasks:
                dep_titles = spec.get("depends_on", [])
                resolved_deps = []
                for dep_title in dep_titles:
                    dep_task = title_to_task.get(dep_title)
                    if dep_task:
                        resolved_deps.append(dep_task.id)
                    else:
                        # case-insensitive fuzzy match
                        for title, t in title_to_task.items():
                            if title.lower().strip() == dep_title.lower().strip():
                                resolved_deps.append(t.id)
                                break
                task.depends_on = resolved_deps

            plan = CoordinatorPlan(
                main_goal=goal,
                worker_tasks=[t for _, t in tasks],
                synthesis_strategy=self._synthesis_strategy,
            )

            logger.info(
                "auto_decompose: created plan",
                extra={"task_count": len(plan.worker_tasks), "goal": goal[:100]},
            )
            return plan

        except Exception as e:
            logger.warning(
                "auto_decompose: LLM call failed, using fallback",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            print(f"  ⚠ auto-decompose failed: {type(e).__name__}: {e}", flush=True)
            return self._fallback_plan(goal, worker_names)

    def _fallback_plan(self, goal: str, worker_names: List[str]) -> CoordinatorPlan:
        """create a simple single-worker plan as fallback.

        when auto-decompose fails, run the full goal as one task
        rather than duplicating it across workers (which wastes tokens).
        """
        logger.info("using fallback plan: single worker for full goal")
        return CoordinatorPlan(
            main_goal=goal,
            worker_tasks=[
                WorkerTask(
                    name=worker_names[0] if worker_names else "worker-0",
                    goal=goal,
                    timeout=self._default_timeout,
                )
            ],
            synthesis_strategy=self._synthesis_strategy,
        )

    async def execute_plan(self, plan: CoordinatorPlan) -> List[WorkerResult]:
        """execute a plan respecting dependency ordering.

        tasks with no dependencies run in parallel. tasks with dependencies
        wait until all deps are completed before starting.
        """
        logger.info(
            "executing plan",
            extra={
                "plan_id": plan.id,
                "worker_count": plan.worker_count,
                "strategy": plan.synthesis_strategy,
            },
        )

        start = time.time()
        results: Dict[str, WorkerResult] = {}
        task_map = {t.id: t for t in plan.worker_tasks}

        # build dependency-aware execution waves
        remaining = set(task_map.keys())
        completed = set()

        while remaining:
            # find tasks whose deps are all completed
            ready = []
            for tid in list(remaining):
                task = task_map[tid]
                deps = getattr(task, "depends_on", []) or []
                if all(d in completed or d not in task_map for d in deps):
                    ready.append(task)

            if not ready:
                # deadlock — all remaining tasks have unmet deps
                logger.warning("execution deadlock — breaking with remaining tasks")
                for tid in remaining:
                    results[tid] = WorkerResult(
                        task_id=tid,
                        status=WorkerStatus.FAILED,
                        error="deadlock: unresolvable dependencies",
                        duration=time.time() - start,
                    )
                break

            # run this wave in parallel
            sem = asyncio.Semaphore(self._max_workers)

            async def _run_task(task: WorkerTask) -> WorkerResult:
                async with sem:
                    print(f"  ▶ worker {task.name}: {task.goal[:80]}...", flush=True)
                    # inject shared memory context + unread messages into worker
                    context_parts = []
                    mem_summary = self._shared_memory.get_summary()
                    if mem_summary:
                        context_parts.append(mem_summary)
                    msg_summary = self._bus.get_summary()
                    if msg_summary:
                        context_parts.append(msg_summary)
                    unread = self._bus.get_unread(task.name)
                    if unread:
                        context_parts.append("\n## messages for you\n")
                        for msg in unread:
                            context_parts.append(f"- from {msg.sender}: {msg.content}")
                        self._bus.mark_read(task.name)

                    # inject upstream task results for dependencies
                    deps = getattr(task, "depends_on", []) or []
                    for dep_id in deps:
                        dep_result = self._shared_memory.get_task_result(dep_id)
                        if dep_result:
                            dep_task = task_map.get(dep_id)
                            dep_name = dep_task.name if dep_task else dep_id
                            context_parts.append(f"\n## result from {dep_name}\n{dep_result}")

                    if context_parts:
                        task.context = (task.context or "") + "\n\n" + "\n".join(context_parts)

                    # swap model for workers if worker_model is set
                    original_model = getattr(self._provider, "model", None)
                    if self._worker_model:
                        self._provider.model = self._worker_model

                    worker = WorkerAgent(
                        provider=self._provider,
                        tools_config=self._tools_config,
                        executor_mode=self._executor_mode,
                        base_dir=self._base_dir,
                    )
                    result = await worker.run(task)

                    # restore original model
                    if self._worker_model and original_model is not None:
                        self._provider.model = original_model

                    # persist result to shared memory
                    if result.success:
                        self._shared_memory.write_task_result(task.id, task.name, result.output)

                    if result.success:
                        print(f"  ✓ worker {task.name} completed", flush=True)
                    else:
                        print(
                            f"  ✗ worker {task.name} failed: {result.error or 'unknown'}",
                            flush=True,
                        )
                    return result

            wave_tasks = [_run_task(t) for t in ready]
            wave_results = await asyncio.gather(*wave_tasks, return_exceptions=True)

            for i, task in enumerate(ready):
                r = wave_results[i]
                if isinstance(r, Exception):
                    print(f"  ✗ worker {task.name} failed: {r}", flush=True)
                    logger.error("worker exception", extra={"task_id": task.id, "error": str(r)})
                    results[task.id] = WorkerResult(
                        task_id=task.id,
                        status=WorkerStatus.FAILED,
                        error=str(r),
                        duration=time.time() - start,
                    )
                else:
                    results[task.id] = r

                remaining.discard(task.id)
                if isinstance(r, WorkerResult) and r.success:
                    completed.add(task.id)
                else:
                    # cascade: mark tasks that depend on this failed task
                    cascade_ids = set()
                    for tid in list(remaining):
                        deps = getattr(task_map[tid], "depends_on", []) or []
                        if task.id in deps:
                            cascade_ids.add(tid)

                    for cid in cascade_ids:
                        remaining.discard(cid)
                        results[cid] = WorkerResult(
                            task_id=cid,
                            status=WorkerStatus.FAILED,
                            error=f"cancelled: dependency {task.id} failed",
                            duration=time.time() - start,
                        )

        elapsed = time.time() - start
        final_results = [results[t.id] for t in plan.worker_tasks if t.id in results]
        succeeded = sum(1 for r in final_results if r.success)

        logger.info(
            "plan execution complete",
            extra={
                "plan_id": plan.id,
                "total": len(final_results),
                "succeeded": succeeded,
                "failed": len(final_results) - succeeded,
                "elapsed": round(elapsed, 2),
            },
        )

        return final_results

    def synthesize(
        self,
        results: List[WorkerResult],
        strategy: str = "merge",
    ) -> str:
        """synthesize worker results into a unified output.

        strategies:
        - merge: concatenate all outputs
        - vote: take majority finding
        - chain: use results sequentially
        """
        successful = [r for r in results if r.success]

        if not successful:
            return "all workers failed — no results to synthesize"

        if strategy == "vote":
            return self._synthesize_vote(successful)
        elif strategy == "chain":
            return self._synthesize_chain(successful)
        else:  # merge
            return self._synthesize_merge(successful)

    def _synthesize_merge(self, results: List[WorkerResult]) -> str:
        parts = []
        for r in results:
            parts.append(f"## {r.task_id}\n\n{r.output}")
            if r.findings:
                parts.append("\n**findings:**")
                for f in r.findings:
                    parts.append(f"- {f}")
        return "\n\n".join(parts)

    def _synthesize_vote(self, results: List[WorkerResult]) -> str:
        all_findings = []
        for r in results:
            all_findings.extend(r.findings)

        if not all_findings:
            return self._synthesize_merge(results)

        from collections import Counter

        counts = Counter(all_findings)
        ranked = counts.most_common()

        parts = ["## findings (by agreement)\n"]
        for finding, count in ranked:
            agreement = f"({count}/{len(results)} workers)"
            parts.append(f"- {finding} {agreement}")

        return "\n".join(parts)

    def _synthesize_chain(self, results: List[WorkerResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"### step {i}: {r.task_id}\n\n{r.output}")
        return "\n\n".join(parts)

    async def run(
        self,
        goal: str = "",
        plan: Optional[CoordinatorPlan] = None,
        system_prompt: str = "",
        worker_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """run the full coordinator flow: plan -> execute -> synthesize.

        args:
            goal: the main goal (used if no plan provided)
            plan: optional pre-built plan
            system_prompt: optional system prompt for workers
            worker_configs: optional list of {name/id, goal, tools, context, timeout}

        returns:
            dict with success, synthesis, worker results, etc.
        """
        # build or use plan
        if plan is not None:
            exec_plan = plan
        elif worker_configs:
            tasks = [
                WorkerTask(
                    name=wc.get("name", wc.get("id", f"worker-{i}")),
                    goal=wc.get("goal", goal),
                    tools=wc.get("tools", []),
                    context=wc.get("context", ""),
                    timeout=wc.get("timeout", self._default_timeout),
                )
                for i, wc in enumerate(worker_configs)
            ]
            exec_plan = CoordinatorPlan(
                main_goal=goal,
                worker_tasks=tasks,
                synthesis_strategy=self._synthesis_strategy,
            )
        else:
            # auto-decompose the goal into tasks via LLM
            worker_names = [f"worker-{i}" for i in range(self._max_workers)]
            exec_plan = await self.auto_decompose(goal, worker_names)

        results = await self.execute_plan(exec_plan)
        synthesis = self.synthesize(results, exec_plan.synthesis_strategy)
        succeeded = sum(1 for r in results if r.success)

        return {
            "success": succeeded > 0,
            "synthesis": synthesis,
            "results": [r.to_dict() for r in results],
            "workers_succeeded": succeeded,
            "workers_failed": len(results) - succeeded,
            "error": None if succeeded > 0 else "all workers failed",
        }
