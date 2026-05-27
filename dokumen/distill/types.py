"""types for the distillation pipeline."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RewardSignal(Enum):
    """how the reward was derived."""
    BINARY = "binary"              # simple pass/fail
    DECOMPOSED = "decomposed"      # from sub-assertion pass rate
    CONSENSUS = "consensus"        # from multi-judge agreement
    CALIBRATED = "calibrated"      # post-calibration confidence


@dataclass
class TrainingExample:
    """a single (prompt, response, reward) tuple for GRPO.

    grpo uses scalar rewards — we get these from:
    - judge verdict: 1.0 if pass, 0.0 if fail
    - decomposed confidence: fraction of sub-assertions passed
    - consensus confidence: agreement ratio across runs
    """

    id: str
    # the prompt that was given to the model
    prompt: str
    # the model's response
    response: str
    # scalar reward [0.0, 1.0]
    reward: float
    # how the reward was computed
    reward_signal: RewardSignal = RewardSignal.BINARY
    # which role this trains (executor or judge)
    role: str = "executor"
    # source metadata
    test_id: str = ""
    judge_id: str = ""
    model: str = ""
    # structured reward breakdown (for SDPO)
    sub_rewards: List[Dict[str, Any]] = field(default_factory=list)
    # timestamp
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "reward": self.reward,
            "reward_signal": self.reward_signal.value,
            "role": self.role,
            "test_id": self.test_id,
            "judge_id": self.judge_id,
            "model": self.model,
            "sub_rewards": self.sub_rewards,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingExample":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            response=d["response"],
            reward=d["reward"],
            reward_signal=RewardSignal(d.get("reward_signal", "binary")),
            role=d.get("role", "executor"),
            test_id=d.get("test_id", ""),
            judge_id=d.get("judge_id", ""),
            model=d.get("model", ""),
            sub_rewards=d.get("sub_rewards", []),
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class PreferencePair:
    """a (chosen, rejected) pair for DPO/SDPO training.

    for SDPO we also include the structured reason why chosen > rejected,
    which gives the model richer signal than just "this one is better".
    """

    id: str
    prompt: str
    chosen: str       # the better response
    rejected: str     # the worse response
    # why chosen > rejected (from decomposed sub-assertions)
    reason: str = ""
    # structured breakdown of what's different
    chosen_sub_assertions: List[Dict[str, Any]] = field(default_factory=list)
    rejected_sub_assertions: List[Dict[str, Any]] = field(default_factory=list)
    # metadata
    test_id: str = ""
    role: str = "executor"
    model: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "reason": self.reason,
            "chosen_sub_assertions": self.chosen_sub_assertions,
            "rejected_sub_assertions": self.rejected_sub_assertions,
            "test_id": self.test_id,
            "role": self.role,
            "model": self.model,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreferencePair":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            chosen=d["chosen"],
            rejected=d["rejected"],
            reason=d.get("reason", ""),
            chosen_sub_assertions=d.get("chosen_sub_assertions", []),
            rejected_sub_assertions=d.get("rejected_sub_assertions", []),
            test_id=d.get("test_id", ""),
            role=d.get("role", "executor"),
            model=d.get("model", ""),
            created_at=d.get("created_at", time.time()),
        )
