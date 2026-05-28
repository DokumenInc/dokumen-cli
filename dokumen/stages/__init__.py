"""Pipeline stages for test execution.

Each stage implements PipelineStage and performs a discrete unit of work.
"""

from .browser_setup import BrowserSetupStage
from .setup import SetupStage
from .explore import ExploreStage
from .executor import ExecutorStage
from .judge import JudgeStage
from .artifact import ArtifactStage
from .memory import MemoryStage
from .compaction import CompactionStage
from .coordinator import CoordinatorStage

__all__ = [
    "BrowserSetupStage",
    "SetupStage",
    "ExploreStage",
    "ExecutorStage",
    "JudgeStage",
    "ArtifactStage",
    "MemoryStage",
    "CompactionStage",
    "CoordinatorStage",
]
