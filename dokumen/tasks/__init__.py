"""
task system for dokumen-cli.

tracks subtasks within test runs. useful for complex executor
scenarios where work needs to be broken into discrete steps.
"""

from .types import Task, TaskStatus, TaskOutput
from .manager import TaskManager

__all__ = [
    "Task",
    "TaskStatus",
    "TaskOutput",
    "TaskManager",
]
