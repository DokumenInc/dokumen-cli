"""dataset loading and saving for eval harness."""

import json
import logging
import os
from typing import List, Optional

from .types import EvalCase

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> List[EvalCase]:
    """load eval cases from a JSON file.

    expected format:
    {
        "name": "dataset-name",
        "cases": [
            {
                "id": "case-1",
                "executor_response": "the refund policy is 30 days",
                "expected_pass": true,
                "assertion": "response mentions 30-day refund window",
                "tags": ["refund", "policy"]
            },
            ...
        ]
    }
    """
    logger.info("loading eval dataset", extra={"path": path})

    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    cases = []
    raw_cases = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(raw_cases, list):
        raise ValueError(f"expected list of cases, got {type(raw_cases).__name__}")

    for raw in raw_cases:
        cases.append(EvalCase.from_dict(raw))

    logger.info("loaded eval dataset", extra={"path": path, "count": len(cases)})
    return cases


def save_dataset(cases: List[EvalCase], path: str, name: str = "unnamed") -> None:
    """save eval cases to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "name": name,
        "cases": [c.to_dict() for c in cases],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("saved eval dataset", extra={"path": path, "count": len(cases)})


def list_datasets(directory: str) -> List[str]:
    """list all .json dataset files in a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".json")
    )
