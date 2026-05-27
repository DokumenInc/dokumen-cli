"""distillation pipeline for training open-source models from judge feedback.

collects preference pairs from test runs:
- GRPO: (prompt, response, scalar_reward) — simpler, reward = pass/fail
- SDPO: (prompt, good_response, bad_response, structured_reason) — richer signal

the decomposed binary judging (#3) gives us structured reasons for free.
"""

from .types import PreferencePair, TrainingExample, RewardSignal
from .collector import TrainingDataCollector
from .exporter import export_grpo_jsonl, export_sdpo_jsonl, export_dpo_jsonl

__all__ = [
    "PreferencePair",
    "TrainingExample",
    "RewardSignal",
    "TrainingDataCollector",
    "export_grpo_jsonl",
    "export_sdpo_jsonl",
    "export_dpo_jsonl",
]
