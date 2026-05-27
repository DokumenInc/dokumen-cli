"""
context management for dokumen-cli.

handles context compaction (auto-compact + micro-compact) to keep
agent conversations within token limits while preserving critical context.
"""

from .compactor import ContextCompactor, CompactionResult
from .micro_compact import MicroCompactor

__all__ = [
    "ContextCompactor",
    "CompactionResult",
    "MicroCompactor",
]
