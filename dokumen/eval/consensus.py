"""multi-judge consensus — run a judge N times, derive confidence from agreement."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """result of running a judge multiple times for consensus."""

    passed: bool
    confidence: float
    # how many runs agreed with the final verdict
    agreement_ratio: float
    # individual run results
    runs: List[Dict[str, Any]] = field(default_factory=list)
    # total cost across all runs
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "confidence": self.confidence,
            "agreement_ratio": self.agreement_ratio,
            "runs": self.runs,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_runs": len(self.runs),
        }


def run_consensus(
    judge_fn: Callable,
    n_runs: int = 3,
    threshold: float = 0.5,
    **judge_kwargs,
) -> ConsensusResult:
    """run a judge function N times and derive consensus.

    args:
        judge_fn: callable that returns {"passed": bool, "confidence": float, ...}
        n_runs: number of times to run the judge
        threshold: fraction of runs that must pass for overall pass
        **judge_kwargs: passed to judge_fn on each call

    returns:
        ConsensusResult with agreement-based confidence
    """
    logger.info("starting consensus run", extra={"n_runs": n_runs, "threshold": threshold})

    runs = []
    pass_count = 0
    total_input = 0
    total_output = 0

    for i in range(n_runs):
        try:
            result = judge_fn(**judge_kwargs)
            is_pass = bool(result.get("passed", False))
            if is_pass:
                pass_count += 1
            runs.append({
                "run": i + 1,
                "passed": is_pass,
                "confidence": float(result.get("confidence", 0.0)),
                "error": None,
            })
            total_input += result.get("input_tokens", 0)
            total_output += result.get("output_tokens", 0)
        except Exception as e:
            logger.error("consensus run failed", extra={"run": i + 1, "error": str(e)})
            runs.append({
                "run": i + 1,
                "passed": False,
                "confidence": 0.0,
                "error": str(e),
            })

    # agreement ratio = fraction that agree with majority
    agreement_ratio = max(pass_count, n_runs - pass_count) / n_runs if n_runs > 0 else 0.0
    # overall pass = fraction of passes meets threshold
    pass_ratio = pass_count / n_runs if n_runs > 0 else 0.0
    overall_pass = pass_ratio >= threshold

    # confidence = agreement ratio (how much judges agree)
    # if all 3 agree → 1.0, if 2/3 agree → 0.667
    confidence = agreement_ratio

    logger.info(
        "consensus complete",
        extra={
            "passed": overall_pass,
            "pass_count": pass_count,
            "n_runs": n_runs,
            "agreement_ratio": agreement_ratio,
            "confidence": confidence,
        },
    )

    return ConsensusResult(
        passed=overall_pass,
        confidence=confidence,
        agreement_ratio=agreement_ratio,
        runs=runs,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )
