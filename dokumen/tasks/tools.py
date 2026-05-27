"""task tools — executor-callable tools for task tracking.

these tools let the executor agent create and manage subtasks
during test execution. registered in tools_object when tasks are enabled.
"""
import logging
from typing import Any, Dict

from .manager import TaskManager
from .types import TaskStatus

logger = logging.getLogger(__name__)

# module-level task manager instance, set by register_task_tools
_task_manager: TaskManager = None


def _get_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


def set_task_manager(manager: TaskManager) -> None:
    """set the task manager instance used by task tools."""
    global _task_manager
    _task_manager = manager


async def handle_task_create(params: Dict[str, Any]) -> Dict[str, Any]:
    """create a new task."""
    mgr = _get_manager()
    description = params.get("description", "")
    name = params.get("name", description[:50] or "unnamed")
    parent_id = params.get("parent_id")

    if not description:
        return {"success": False, "error": "description is required"}

    task = mgr.create(name=name, description=description, parent_id=parent_id)
    logger.info("task_tool.create", extra={"task_id": task.id, "description": description[:80]})
    return {"success": True, "task_id": task.id, "status": task.status.value}


async def handle_task_update(params: Dict[str, Any]) -> Dict[str, Any]:
    """update a task's status."""
    mgr = _get_manager()
    task_id = params.get("task_id", "")
    status = params.get("status", "")

    if not task_id or not status:
        return {"success": False, "error": "task_id and status are required"}

    try:
        status_enum = TaskStatus(status)
    except ValueError:
        return {"success": False, "error": f"invalid status: {status}. use: pending, in_progress, completed, failed, cancelled"}

    task = mgr.get(task_id)
    if task is None:
        return {"success": False, "error": f"task not found: {task_id}"}

    if status_enum == TaskStatus.IN_PROGRESS:
        mgr.start(task_id)
    elif status_enum == TaskStatus.COMPLETED:
        mgr.complete(task_id)
    elif status_enum == TaskStatus.FAILED:
        mgr.fail(task_id, error=params.get("error"))
    elif status_enum == TaskStatus.CANCELLED:
        mgr.cancel(task_id)

    logger.info("task_tool.update", extra={"task_id": task_id, "status": status})
    return {"success": True, "task_id": task_id, "status": status}


async def handle_task_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """list all tasks."""
    mgr = _get_manager()
    tasks = mgr.list()
    return {
        "success": True,
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "status": t.status.value,
                "parent_id": t.parent_id,
            }
            for t in tasks
        ],
        "count": len(tasks),
    }


async def handle_task_output(params: Dict[str, Any]) -> Dict[str, Any]:
    """add output to a task."""
    mgr = _get_manager()
    task_id = params.get("task_id", "")
    content = params.get("content", "")
    output_type = params.get("type", "text")

    if not task_id or not content:
        return {"success": False, "error": "task_id and content are required"}

    task = mgr.get(task_id)
    if task is None:
        return {"success": False, "error": f"task not found: {task_id}"}

    mgr.add_output(task_id, content=content, metadata={"type": output_type})
    logger.info("task_tool.output", extra={"task_id": task_id, "type": output_type})
    return {"success": True, "task_id": task_id}


# tool definitions for registration
TASK_TOOL_DEFINITIONS = [
    {
        "name": "task_create",
        "description": "create a new subtask to track work. use this to break complex tasks into trackable pieces.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "what this task does"},
                "name": {"type": "string", "description": "short task name (defaults to description)"},
                "parent_id": {"type": "string", "description": "parent task id (optional)"},
            },
            "required": ["description"],
        },
        "handler": handle_task_create,
    },
    {
        "name": "task_update",
        "description": "update the status of a task. statuses: pending, in_progress, completed, failed, cancelled.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "the task id to update"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed", "cancelled"]},
                "error": {"type": "string", "description": "error message (for failed status)"},
            },
            "required": ["task_id", "status"],
        },
        "handler": handle_task_update,
    },
    {
        "name": "task_list",
        "description": "list all tracked tasks and their statuses.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
        "handler": handle_task_list,
    },
    {
        "name": "task_output",
        "description": "add output content to a task (findings, results, artifacts).",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "the task id"},
                "content": {"type": "string", "description": "the output content"},
                "type": {"type": "string", "description": "output type (text, json, code)", "default": "text"},
            },
            "required": ["task_id", "content"],
        },
        "handler": handle_task_output,
    },
]
