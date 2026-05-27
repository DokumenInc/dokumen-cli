"""eval harness — runs judges against ground truth datasets."""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .types import EvalCase, EvalResult, EvalSummary
from . import metrics as m

logger = logging.getLogger(__name__)


@dataclass
class EvalHarness:
    """runs a judge function against a dataset and computes metrics.

    the judge_fn takes (executor_response, assertion, system_prompt, user_prompt)
    and returns a dict with at minimum: {"passed": bool, "confidence": float}.
    optionally includes "sub_assertions", "raw_response", token counts, etc.

    this keeps the harness decoupled from JudgeAgent — you can pass any callable,
    including a mock for offline testing.
    """

    judge_fn: Callable
    judge_model: str = "unknown"
    dataset_name: str = "unnamed"

    def run_case(self, case: EvalCase) -> EvalResult:
        """evaluate a single case synchronously."""
        start = time.time()
        error = None
        actual_pass = False
        confidence = 0.0
        sub_assertions = []
        raw_response = ""
        input_tokens = 0
        output_tokens = 0

        try:
            result = self.judge_fn(
                executor_response=case.executor_response,
                assertion=case.assertion,
                system_prompt=case.system_prompt,
                user_prompt=case.user_prompt,
            )
            actual_pass = bool(result.get("passed", False))
            confidence = float(result.get("confidence", 0.0))
            sub_assertions = result.get("sub_assertions", [])
            raw_response = result.get("raw_response", "")
            input_tokens = result.get("input_tokens", 0)
            output_tokens = result.get("output_tokens", 0)
        except Exception as e:
            error = str(e)
            logger.error("judge failed on case", extra={"case_id": case.id, "error": error})

        duration_ms = int((time.time() - start) * 1000)

        return EvalResult(
            case_id=case.id,
            expected_pass=case.expected_pass,
            actual_pass=actual_pass,
            confidence=confidence,
            sub_assertions=sub_assertions,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=raw_response,
            error=error,
        )

    def run(self, cases: List[EvalCase], tags_filter: Optional[List[str]] = None) -> EvalSummary:
        """run all cases and compute aggregate metrics."""
        logger.info(
            "starting eval run",
            extra={"dataset": self.dataset_name, "model": self.judge_model, "cases": len(cases)},
        )

        # filter by tags if requested
        if tags_filter:
            tag_set = set(tags_filter)
            cases = [c for c in cases if tag_set.intersection(c.tags)]
            logger.info("filtered to cases with tags", extra={"tags": tags_filter, "remaining": len(cases)})

        results: List[EvalResult] = []
        for case in cases:
            result = self.run_case(case)
            results.append(result)

        return self._summarize(results)

    def _summarize(self, results: List[EvalResult]) -> EvalSummary:
        """compute metrics from results."""
        tp = sum(1 for r in results if r.is_true_positive)
        tn = sum(1 for r in results if r.is_true_negative)
        fp = sum(1 for r in results if r.is_false_positive)
        fn = sum(1 for r in results if r.is_false_negative)
        correct = tp + tn
        total = len(results)

        prec = m.precision(tp, fp)
        rec = m.recall(tp, fn)

        # calibration: (confidence, was_correct) pairs
        calibration_pairs = [(r.confidence, r.correct) for r in results if r.error is None]
        ece = m.expected_calibration_error(calibration_pairs)

        # per-tag breakdown
        tag_results = defaultdict(list)
        for r in results:
            # find the case tags from the result
            for case_tag in self._get_case_tags(r.case_id, results):
                tag_results[case_tag].append(r)

        tag_metrics = {}
        for tag, tag_rs in tag_results.items():
            t_tp = sum(1 for r in tag_rs if r.is_true_positive)
            t_tn = sum(1 for r in tag_rs if r.is_true_negative)
            t_fp = sum(1 for r in tag_rs if r.is_false_positive)
            t_fn = sum(1 for r in tag_rs if r.is_false_negative)
            t_prec = m.precision(t_tp, t_fp)
            t_rec = m.recall(t_tp, t_fn)
            tag_metrics[tag] = {
                "accuracy": m.accuracy(t_tp + t_tn, len(tag_rs)),
                "precision": t_prec,
                "recall": t_rec,
                "f1": m.f1(t_prec, t_rec),
                "count": len(tag_rs),
            }

        summary = EvalSummary(
            dataset_name=self.dataset_name,
            judge_model=self.judge_model,
            total=total,
            correct=correct,
            accuracy=m.accuracy(correct, total),
            precision=prec,
            recall=rec,
            f1=m.f1(prec, rec),
            expected_calibration_error=ece,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total_input_tokens=sum(r.input_tokens for r in results),
            total_output_tokens=sum(r.output_tokens for r in results),
            total_duration_ms=sum(r.duration_ms for r in results),
            results=results,
            tag_metrics=tag_metrics,
        )

        logger.info(
            "eval run complete",
            extra={
                "accuracy": summary.accuracy,
                "precision": summary.precision,
                "recall": summary.recall,
                "f1": summary.f1,
                "ece": summary.expected_calibration_error,
            },
        )

        return summary

    def _get_case_tags(self, case_id: str, results: List[EvalResult]) -> List[str]:
        """helper to retrieve tags — stored in _case_tags if available."""
        return getattr(self, "_case_tags", {}).get(case_id, [])

    def run_with_cases(self, cases: List[EvalCase], tags_filter: Optional[List[str]] = None) -> EvalSummary:
        """run with full case objects so we can track tags."""
        # stash tags for summarization
        self._case_tags = {c.id: c.tags for c in cases}

        if tags_filter:
            tag_set = set(tags_filter)
            cases = [c for c in cases if tag_set.intersection(c.tags)]

        results = [self.run_case(case) for case in cases]
        summary = self._summarize(results)

        # cleanup
        del self._case_tags
        return summary

    def save_results(self, summary: EvalSummary, path: str) -> None:
        """save eval results to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info("saved eval results", extra={"path": path})
