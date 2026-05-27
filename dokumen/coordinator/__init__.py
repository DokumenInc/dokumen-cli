"""
coordinator mode — multi-agent orchestration for dokumen-cli.

a master coordinator agent spawns worker agents for parallel investigation.
workers report back via structured results. coordinator synthesizes findings.

inspired by anthropic's harness design: planner -> generator -> evaluator.
"""
from .types import WorkerTask, WorkerResult, CoordinatorPlan
from .coordinator import CoordinatorAgent
from .worker import WorkerAgent
from .messaging import MessageBus, Message
from .shared_memory import SharedMemory

__all__ = [
    "WorkerTask",
    "WorkerResult",
    "CoordinatorPlan",
    "CoordinatorAgent",
    "WorkerAgent",
    "MessageBus",
    "Message",
    "SharedMemory",
]
