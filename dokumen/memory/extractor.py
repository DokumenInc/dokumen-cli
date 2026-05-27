"""
memory extractor — tier 2 of the three-tier memory system.

post-run background extraction: after each test run, extracts learnings
(failure patterns, useful tool sequences, domain knowledge) and writes
them to persistent memory.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schemas import Memory

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """result of extracting memories from a test run."""
    memories: List[Memory] = field(default_factory=list)
    skipped: int = 0  # duplicates or low-quality
    source_test: str = ""
    extraction_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "skipped": self.skipped,
            "source_test": self.source_test,
            "extraction_time": self.extraction_time,
        }


class MemoryExtractor:
    """extracts persistent memories from test run results.

    uses rule-based extraction (no llm needed). looks for:
    - failure patterns (judge failures with reasons)
    - successful tool sequences
    - domain-specific findings
    - error recovery patterns

    usage:
        extractor = MemoryExtractor()
        result = extractor.extract_from_run(test_result)
        for memory in result.memories:
            store.add(memory)
    """

    def __init__(
        self,
        min_reason_length: int = 20,
        max_memories_per_run: int = 10,
    ):
        self._min_reason_length = min_reason_length
        self._max_memories_per_run = max_memories_per_run

    def extract_from_run(self, run_data: Dict[str, Any]) -> ExtractionResult:
        """extract memories from a test run result dict.

        expected keys in run_data:
            test_id: str
            passed: bool
            judges: list of {judge_id, passed, reason, confidence, sub_assertions}
            tool_calls: list of {tool, args, result}
            executor_response: str (optional)
        """
        start = time.time()
        memories = []
        skipped = 0
        test_id = run_data.get("test_id", "unknown")

        logger.info(
            "extracting memories from run",
            extra={"test_id": test_id, "passed": run_data.get("passed")},
        )

        # extract from judge failures
        for judge in run_data.get("judges", []):
            if not judge.get("passed", True):
                mem = self._extract_from_failure(test_id, judge)
                if mem:
                    memories.append(mem)
                else:
                    skipped += 1

            # extract from sub-assertions
            for sub in judge.get("sub_assertions", []):
                if not sub.get("passed", True):
                    mem = self._extract_from_sub_assertion(test_id, judge.get("judge_id", ""), sub)
                    if mem:
                        memories.append(mem)
                    else:
                        skipped += 1

        # extract from tool call patterns on success
        if run_data.get("passed", False):
            tool_mem = self._extract_tool_pattern(test_id, run_data.get("tool_calls", []))
            if tool_mem:
                memories.append(tool_mem)

        # cap at max
        if len(memories) > self._max_memories_per_run:
            skipped += len(memories) - self._max_memories_per_run
            memories = memories[:self._max_memories_per_run]

        elapsed = time.time() - start
        logger.info(
            "extraction complete",
            extra={
                "test_id": test_id,
                "extracted": len(memories),
                "skipped": skipped,
                "elapsed": round(elapsed, 3),
            },
        )

        return ExtractionResult(
            memories=memories,
            skipped=skipped,
            source_test=test_id,
            extraction_time=elapsed,
        )

    def _extract_from_failure(self, test_id: str, judge: Dict[str, Any]) -> Optional[Memory]:
        """extract a memory from a judge failure."""
        reason = judge.get("reason", "") or judge.get("failure_reason", "")
        if len(reason) < self._min_reason_length:
            return None

        judge_id = judge.get("judge_id", "unknown")
        content = f"test '{test_id}' failed judge '{judge_id}': {reason}"

        return Memory(
            id=f"mem-{test_id}-{judge_id}-{int(time.time())}",
            content=content,
            metadata={
                "type": "failure_pattern",
                "test_id": test_id,
                "judge_id": judge_id,
                "confidence": judge.get("confidence", 0.0),
            },
        )

    def _extract_from_sub_assertion(
        self, test_id: str, judge_id: str, sub: Dict[str, Any]
    ) -> Optional[Memory]:
        """extract a memory from a failed sub-assertion."""
        reason = sub.get("reason", "")
        question = sub.get("question", "")
        if len(reason) < self._min_reason_length:
            return None

        content = f"sub-assertion failed in '{test_id}/{judge_id}': {question} — {reason}"

        return Memory(
            id=f"mem-{test_id}-{judge_id}-sub-{int(time.time())}",
            content=content,
            metadata={
                "type": "sub_assertion_failure",
                "test_id": test_id,
                "judge_id": judge_id,
                "question": question,
            },
        )

    def _extract_tool_pattern(self, test_id: str, tool_calls: List[Dict[str, Any]]) -> Optional[Memory]:
        """extract a useful tool sequence from a successful run."""
        if len(tool_calls) < 2:
            return None

        tool_names = [tc.get("tool", "unknown") for tc in tool_calls]
        # deduplicate consecutive same tools
        deduped = [tool_names[0]]
        for t in tool_names[1:]:
            if t != deduped[-1]:
                deduped.append(t)

        if len(deduped) < 2:
            return None

        sequence = " → ".join(deduped[:10])
        content = f"successful tool sequence for '{test_id}': {sequence}"

        return Memory(
            id=f"mem-{test_id}-tools-{int(time.time())}",
            content=content,
            metadata={
                "type": "tool_pattern",
                "test_id": test_id,
                "tool_count": len(tool_calls),
                "unique_tools": list(set(tool_names)),
            },
        )
