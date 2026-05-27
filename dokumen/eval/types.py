"""types for the eval harness."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalCase:
    """a single eval case with ground truth."""

    id: str
    # the executor output to judge (pre-computed or raw text)
    executor_response: str
    # ground truth: should the judge pass or fail this?
    expected_pass: bool
    # optional context
    system_prompt: str = ""
    user_prompt: str = ""
    assertion: str = ""
    # metadata for filtering/grouping
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "executor_response": self.executor_response,
            "expected_pass": self.expected_pass,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "assertion": self.assertion,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalCase":
        return cls(
            id=d["id"],
            executor_response=d["executor_response"],
            expected_pass=d["expected_pass"],
            system_prompt=d.get("system_prompt", ""),
            user_prompt=d.get("user_prompt", ""),
            assertion=d.get("assertion", ""),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )


@dataclass
class EvalResult:
    """result of running a judge on one eval case."""

    case_id: str
    expected_pass: bool
    actual_pass: bool
    # decomposed sub-assertion details
    sub_assertions: List[Dict[str, Any]] = field(default_factory=list)
    # timing and cost
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # raw judge response for debugging
    raw_response: str = ""
    error: Optional[str] = None

    @property
    def correct(self) -> bool:
        return self.expected_pass == self.actual_pass

    @property
    def is_true_positive(self) -> bool:
        return self.expected_pass and self.actual_pass

    @property
    def is_true_negative(self) -> bool:
        return not self.expected_pass and not self.actual_pass

    @property
    def is_false_positive(self) -> bool:
        return not self.expected_pass and self.actual_pass

    @property
    def is_false_negative(self) -> bool:
        return self.expected_pass and not self.actual_pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_pass": self.expected_pass,
            "actual_pass": self.actual_pass,
            "correct": self.correct,
            "sub_assertions": self.sub_assertions,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "raw_response": self.raw_response,
            "error": self.error,
        }


@dataclass
class EvalSummary:
    """aggregated metrics from an eval run."""

    dataset_name: str
    judge_model: str
    total: int
    correct: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    # confusion matrix
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    # cost
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: int = 0
    # per-result details
    results: List[EvalResult] = field(default_factory=list)
    # when this ran
    timestamp: float = field(default_factory=time.time)
    # optional breakdown by tag
    tag_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "judge_model": self.judge_model,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_duration_ms": self.total_duration_ms,
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp,
            "tag_metrics": self.tag_metrics,
        }
