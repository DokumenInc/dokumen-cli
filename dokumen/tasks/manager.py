"""
task manager — create, update, list, and track subtasks with DAG dependencies.

protocol-based store (rule 2.6) — currently backed by in-memory dict
with optional disk persistence to .dokumen-cache/tasks/.

supports dependency graphs: tasks can declare depends_on to other tasks.
the manager handles topological ordering, auto-unblocking when dependencies
complete, and cascade failure when dependencies fail.
"""

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from .types import Task, TaskOutput, TaskStatus

logger = logging.getLogger(__name__)


class TaskStore(Protocol):
    """Abstract task store implementation."""

    def save(self, task: Task) -> None: ...
    def load(self, task_id: str) -> Optional[Task]: ...
    def load_all(self) -> List[Task]: ...
    def delete(self, task_id: str) -> bool: ...


class InMemoryTaskStore:
    """in-memory task store with optional disk persistence."""

    def __init__(self, persist_dir: Optional[str] = None):
        self._tasks: Dict[str, Task] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def save(self, task: Task) -> None:
        self._tasks[task.id] = task
        if self._persist_dir:
            self._save_to_disk(task)

    def load(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def load_all(self) -> List[Task]:
        return list(self._tasks.values())

    def delete(self, task_id: str) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            if self._persist_dir:
                filepath = self._persist_dir / f"{task_id}.json"
                if filepath.exists():
                    filepath.unlink()
            return True
        return False

    def _save_to_disk(self, task: Task) -> None:
        filepath = self._persist_dir / f"{task.id}.json"
        filepath.write_text(json.dumps(task.to_dict(), indent=2), encoding="utf-8")

    def _load_from_disk(self) -> None:
        for filepath in self._persist_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                task = Task.from_dict(data)
                self._tasks[task.id] = task
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    "failed to load task from disk", extra={"path": str(filepath), "error": str(e)}
                )


# ── event types ──

TASK_READY = "task:ready"
TASK_COMPLETE = "task:complete"
TASK_FAILED = "task:failed"
ALL_COMPLETE = "all:complete"

TaskEventCallback = Callable[[Task], None]
AllCompleteCallback = Callable[[], None]


# ── DAG utilities ──


def topological_sort(tasks: List[Task]) -> List[Task]:
    """kahn's algorithm — returns tasks in dependency order.

    tasks with no dependencies come first. if there's a cycle,
    the cycled tasks are omitted from the result.
    """
    task_map = {t.id: t for t in tasks}
    in_degree: Dict[str, int] = {t.id: 0 for t in tasks}
    successors: Dict[str, List[str]] = {t.id: [] for t in tasks}

    for t in tasks:
        for dep_id in t.depends_on:
            if dep_id in task_map:
                in_degree[t.id] += 1
                successors[dep_id].append(t.id)

    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    result = []

    while queue:
        tid = queue.popleft()
        result.append(task_map[tid])
        for succ_id in successors[tid]:
            in_degree[succ_id] -= 1
            if in_degree[succ_id] == 0:
                queue.append(succ_id)

    return result


def validate_dependencies(tasks: List[Task]) -> Dict[str, List[str]]:
    """validate task dependency graph.

    returns dict of errors. empty dict means valid.
    checks: unknown refs, self-refs, cycles.
    """
    errors: Dict[str, List[str]] = {}
    task_ids = {t.id for t in tasks}

    for t in tasks:
        task_errors = []

        # self-reference
        if t.id in t.depends_on:
            task_errors.append(f"self-reference: {t.id} depends on itself")

        # unknown references
        for dep_id in t.depends_on:
            if dep_id not in task_ids:
                task_errors.append(f"unknown dependency: {dep_id}")

        if task_errors:
            errors[t.id] = task_errors

    # cycle detection via DFS three-color marking
    WHITE, GREY, BLACK = 0, 1, 2
    color = {t.id: WHITE for t in tasks}
    task_map = {t.id: t for t in tasks}

    def _dfs(tid: str, path: List[str]) -> Optional[str]:
        color[tid] = GREY
        path.append(tid)
        for dep_id in task_map.get(tid, Task()).depends_on:
            if dep_id not in color:
                continue
            if color[dep_id] == GREY:
                # found cycle — trace back
                cycle_start = path.index(dep_id)
                cycle_path = " -> ".join(path[cycle_start:] + [dep_id])
                return cycle_path
            if color[dep_id] == WHITE:
                result = _dfs(dep_id, path)
                if result:
                    return result
        color[tid] = BLACK
        path.pop()
        return None

    for t in tasks:
        if color[t.id] == WHITE:
            cycle = _dfs(t.id, [])
            if cycle:
                errors.setdefault("_cycles", []).append(cycle)

    return errors


def is_task_ready(task: Task, all_tasks: Dict[str, Task]) -> bool:
    """check if a task's dependencies are all completed."""
    if task.status != TaskStatus.BLOCKED and task.status != TaskStatus.PENDING:
        return False
    if not task.depends_on:
        return True
    return all(
        all_tasks.get(dep_id, Task()).status == TaskStatus.COMPLETED for dep_id in task.depends_on
    )


class TaskManager:
    """manages task lifecycle with DAG dependency support.

    usage:
        manager = TaskManager()
        a = manager.create("step a", description="first step")
        b = manager.create("step b", description="needs a", depends_on=[a.id])
        c = manager.create("step c", description="needs a", depends_on=[a.id])
        d = manager.create("step d", description="needs b+c", depends_on=[b.id, c.id])

        # b, c, d start as BLOCKED. a starts as PENDING.
        ready = manager.get_ready_tasks()  # [a]
        manager.start(a.id)
        manager.complete(a.id)  # auto-unblocks b and c
        ready = manager.get_ready_tasks()  # [b, c]
    """

    def __init__(self, store: Optional[TaskStore] = None, persist_dir: Optional[str] = None):
        self._store = store or InMemoryTaskStore(persist_dir=persist_dir)
        self._listeners: Dict[str, List[Callable]] = {
            TASK_READY: [],
            TASK_COMPLETE: [],
            TASK_FAILED: [],
            ALL_COMPLETE: [],
        }

        logger.info(
            "task manager initialized",
            extra={"store_type": type(self._store).__name__},
        )

    # ── event system ──

    def on(self, event: str, callback: Callable) -> Callable:
        """subscribe to a task event. returns an unsubscribe function."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)

        def unsubscribe():
            try:
                self._listeners[event].remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def _emit(self, event: str, data: Any = None) -> None:
        """fire all listeners for an event."""
        for cb in self._listeners.get(event, []):
            try:
                cb(data) if data is not None else cb()
            except Exception as e:
                logger.warning("event callback error", extra={"event": event, "error": str(e)})

    # ── core CRUD ──

    def create(
        self,
        name: str,
        description: str = "",
        parent_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """create a new task. auto-resolves initial status based on dependencies."""
        task = Task(
            name=name,
            description=description,
            parent_id=parent_id,
            depends_on=list(depends_on) if depends_on else [],
            metadata=metadata or {},
        )

        # resolve initial status: if has unfinished deps, start as BLOCKED
        if task.depends_on:
            all_tasks = {t.id: t for t in self._store.load_all()}
            all_tasks[task.id] = task  # include self for consistency
            if not is_task_ready(task, all_tasks):
                task.status = TaskStatus.BLOCKED

        self._store.save(task)

        logger.info(
            "task created",
            extra={
                "task_id": task.id,
                "name": name,
                "parent_id": parent_id,
                "depends_on": task.depends_on,
                "status": task.status.value,
            },
        )

        if task.status == TaskStatus.PENDING:
            self._emit(TASK_READY, task)

        return task

    def get(self, task_id: str) -> Optional[Task]:
        """get a task by id."""
        return self._store.load(task_id)

    def list(
        self,
        status: Optional[TaskStatus] = None,
        parent_id: Optional[str] = None,
    ) -> List[Task]:
        """list tasks, optionally filtered by status or parent."""
        tasks = self._store.load_all()

        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if parent_id is not None:
            tasks = [t for t in tasks if t.parent_id == parent_id]

        return sorted(tasks, key=lambda t: t.created_at)

    def start(self, task_id: str) -> Optional[Task]:
        """transition task to in_progress."""
        task = self._store.load(task_id)
        if task is None:
            return None
        if task.is_terminal:
            logger.warning(
                "cannot start terminal task",
                extra={"task_id": task_id, "status": task.status.value},
            )
            return task
        if task.status == TaskStatus.BLOCKED:
            logger.warning("cannot start blocked task", extra={"task_id": task_id})
            return task

        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = time.time()
        self._store.save(task)

        logger.info("task started", extra={"task_id": task_id})
        return task

    def add_output(
        self, task_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        """add an output entry to a task."""
        task = self._store.load(task_id)
        if task is None:
            return None

        task.outputs.append(TaskOutput(content=content, metadata=metadata or {}))
        task.updated_at = time.time()
        self._store.save(task)

        logger.debug(
            "task output added",
            extra={"task_id": task_id, "output_count": len(task.outputs)},
        )
        return task

    def complete(self, task_id: str) -> Optional[Task]:
        """mark task as completed. auto-unblocks dependents."""
        task = self._store.load(task_id)
        if task is None:
            return None

        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.updated_at = time.time()
        self._store.save(task)

        logger.info(
            "task completed",
            extra={"task_id": task_id, "duration": task.duration},
        )

        self._emit(TASK_COMPLETE, task)
        self._unblock_dependents(task_id)
        self._check_all_complete()

        return task

    def fail(self, task_id: str, error: str = "") -> Optional[Task]:
        """mark task as failed. cascades failure to dependents."""
        task = self._store.load(task_id)
        if task is None:
            return None

        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = time.time()
        task.updated_at = time.time()
        self._store.save(task)

        logger.info(
            "task failed",
            extra={"task_id": task_id, "error": error},
        )

        self._emit(TASK_FAILED, task)
        self._cascade_failure(task_id)
        self._check_all_complete()

        return task

    def cancel(self, task_id: str) -> Optional[Task]:
        """cancel a task."""
        task = self._store.load(task_id)
        if task is None:
            return None
        if task.is_terminal:
            return task

        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        task.updated_at = time.time()
        self._store.save(task)

        logger.info("task cancelled", extra={"task_id": task_id})
        return task

    def delete(self, task_id: str) -> bool:
        """delete a task entirely."""
        return self._store.delete(task_id)

    def get_subtasks(self, parent_id: str) -> List[Task]:
        """get all subtasks of a parent task."""
        return self.list(parent_id=parent_id)

    def summary(self) -> Dict[str, Any]:
        """get a summary of all tasks."""
        tasks = self._store.load_all()
        by_status = {}
        for t in tasks:
            key = t.status.value
            by_status[key] = by_status.get(key, 0) + 1

        return {
            "total": len(tasks),
            "by_status": by_status,
        }

    # ── DAG operations ──

    def get_ready_tasks(self) -> List[Task]:
        """get all tasks that are ready to execute (pending, deps satisfied)."""
        all_tasks = {t.id: t for t in self._store.load_all()}
        ready = []
        for t in all_tasks.values():
            if t.status == TaskStatus.PENDING:
                ready.append(t)
            elif t.status == TaskStatus.BLOCKED and is_task_ready(t, all_tasks):
                # auto-promote to pending
                t.status = TaskStatus.PENDING
                t.updated_at = time.time()
                self._store.save(t)
                self._emit(TASK_READY, t)
                ready.append(t)
        return sorted(ready, key=lambda t: t.created_at)

    def get_execution_order(self) -> List[Task]:
        """get tasks in topological order (respecting dependencies)."""
        tasks = self._store.load_all()
        return topological_sort(tasks)

    def validate(self) -> Dict[str, List[str]]:
        """validate the dependency graph. returns errors dict (empty = valid)."""
        tasks = self._store.load_all()
        return validate_dependencies(tasks)

    def get_dependents(self, task_id: str) -> List[Task]:
        """get all tasks that directly depend on the given task."""
        return [t for t in self._store.load_all() if task_id in t.depends_on]

    def get_blocked_count(self, task_id: str) -> int:
        """count how many tasks are transitively blocked by this task.

        useful for critical path analysis — tasks blocking more downstream
        work should be prioritized.
        """
        all_tasks = self._store.load_all()
        # build reverse adjacency: task_id -> list of tasks that depend on it
        successors: Dict[str, List[str]] = {}
        for t in all_tasks:
            for dep_id in t.depends_on:
                successors.setdefault(dep_id, []).append(t.id)

        # BFS from task_id
        visited = set()
        queue = deque(successors.get(task_id, []))
        while queue:
            tid = queue.popleft()
            if tid in visited:
                continue
            visited.add(tid)
            queue.extend(successors.get(tid, []))

        return len(visited)

    def add_batch(self, tasks: List[Task]) -> List[Task]:
        """add multiple tasks at once, resolving initial statuses.

        tasks are added in topological order so dependency resolution
        works correctly within the batch.
        """
        sorted_tasks = topological_sort(tasks)
        added = []

        for t in sorted_tasks:
            # build the map including already-stored tasks + batch so far
            all_tasks = {st.id: st for st in self._store.load_all()}
            all_tasks[t.id] = t

            if t.depends_on and not is_task_ready(t, all_tasks):
                t.status = TaskStatus.BLOCKED

            self._store.save(t)
            added.append(t)

            if t.status == TaskStatus.PENDING:
                self._emit(TASK_READY, t)

        logger.info("batch added", extra={"count": len(added)})
        return added

    # ── internal DAG methods ──

    def _unblock_dependents(self, completed_task_id: str) -> None:
        """when a task completes, check if any blocked tasks can now run."""
        all_tasks = {t.id: t for t in self._store.load_all()}
        unblocked = []

        for t in all_tasks.values():
            if t.status != TaskStatus.BLOCKED:
                continue
            if completed_task_id not in t.depends_on:
                continue
            if is_task_ready(t, all_tasks):
                t.status = TaskStatus.PENDING
                t.updated_at = time.time()
                self._store.save(t)
                all_tasks[t.id] = t  # update map for subsequent checks
                unblocked.append(t)

        for t in unblocked:
            logger.info(
                "task unblocked", extra={"task_id": t.id, "unblocked_by": completed_task_id}
            )
            self._emit(TASK_READY, t)

    def _cascade_failure(self, failed_task_id: str) -> None:
        """when a task fails, recursively fail all tasks that depend on it."""
        all_tasks = {t.id: t for t in self._store.load_all()}
        to_fail = deque()

        # find direct dependents
        for t in all_tasks.values():
            if failed_task_id in t.depends_on and not t.is_terminal:
                to_fail.append(t.id)

        failed_ids = set()
        while to_fail:
            tid = to_fail.popleft()
            if tid in failed_ids:
                continue
            failed_ids.add(tid)

            task = all_tasks.get(tid)
            if task is None or task.is_terminal:
                continue

            task.status = TaskStatus.FAILED
            task.error = f"cancelled: dependency {failed_task_id} failed"
            task.completed_at = time.time()
            task.updated_at = time.time()
            self._store.save(task)

            logger.info(
                "task cascade failed",
                extra={"task_id": tid, "caused_by": failed_task_id},
            )
            self._emit(TASK_FAILED, task)

            # find transitive dependents
            for t in all_tasks.values():
                if tid in t.depends_on and not t.is_terminal and t.id not in failed_ids:
                    to_fail.append(t.id)

    def _check_all_complete(self) -> None:
        """check if all tasks are in terminal states."""
        tasks = self._store.load_all()
        if not tasks:
            return
        if all(t.is_terminal for t in tasks):
            logger.info("all tasks complete")
            self._emit(ALL_COMPLETE)
