"""
Dokumen Agent SDK integration package.

Provides executor/judge agents built on the Claude Agent SDK.
This is the only execution path for executor and judge agents.

Canonical result types (ExecutorResult, JudgeVerdict) live in sdk/types.py.
Legacy aliases (SdkExecutorResult, SdkJudgeResult) are re-exported for
backward compatibility.
"""

from .types import (
    ExecutorResult,
    JudgeVerdict,
    QueryResult,
    SdkExecutorResult,
    SdkJudgeResult,
    Verdict,
)

__all__ = [
    # Canonical types
    "ExecutorResult",
    "JudgeVerdict",
    "QueryResult",
    "Verdict",
    # Backward-compatible aliases
    "SdkExecutorResult",
    "SdkJudgeResult",
]
