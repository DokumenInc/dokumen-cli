"""export training data to standard formats for model training.

formats:
- GRPO JSONL: {"prompt": "...", "response": "...", "reward": 0.8}
- DPO JSONL: {"prompt": "...", "chosen": "...", "rejected": "..."}
- SDPO JSONL: DPO + structured reason + sub-assertion breakdown

these can be fed directly into trl (huggingface), openrlhf, or
custom training loops.
"""

import json
import logging
import os
from typing import List

from .types import TrainingExample, PreferencePair

logger = logging.getLogger(__name__)


def export_grpo_jsonl(examples: List[TrainingExample], path: str) -> int:
    """export training examples in GRPO format.

    format per line:
    {"prompt": "...", "response": "...", "reward": 0.8, "metadata": {...}}

    compatible with trl's RewardTrainer and openrlhf.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for ex in examples:
            line = {
                "prompt": ex.prompt,
                "response": ex.response,
                "reward": ex.reward,
                "metadata": {
                    "id": ex.id,
                    "test_id": ex.test_id,
                    "judge_id": ex.judge_id,
                    "model": ex.model,
                    "reward_signal": ex.reward_signal.value,
                    "role": ex.role,
                },
            }
            # include sub-rewards for GRPO with structured rewards
            if ex.sub_rewards:
                line["sub_rewards"] = ex.sub_rewards
            f.write(json.dumps(line) + "\n")
            count += 1

    logger.info("exported GRPO dataset", extra={"path": path, "count": count})
    return count


def export_dpo_jsonl(pairs: List[PreferencePair], path: str) -> int:
    """export preference pairs in standard DPO format.

    format per line:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

    compatible with trl's DPOTrainer.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for pair in pairs:
            line = {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
            }
            f.write(json.dumps(line) + "\n")
            count += 1

    logger.info("exported DPO dataset", extra={"path": path, "count": count})
    return count


def export_sdpo_jsonl(pairs: List[PreferencePair], path: str) -> int:
    """export preference pairs in SDPO format (DPO + structured reasons).

    format per line:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "...",
        "reason": "structured explanation of why chosen > rejected",
        "chosen_sub_assertions": [...],
        "rejected_sub_assertions": [...]
    }

    the structured reason is the key differentiator from regular DPO.
    it tells the model not just WHAT is better but WHY, enabling
    self-distilled policy optimization.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for pair in pairs:
            line = {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "reason": pair.reason,
                "chosen_sub_assertions": pair.chosen_sub_assertions,
                "rejected_sub_assertions": pair.rejected_sub_assertions,
                "metadata": {
                    "id": pair.id,
                    "test_id": pair.test_id,
                    "model": pair.model,
                    "role": pair.role,
                },
            }
            f.write(json.dumps(line) + "\n")
            count += 1

    logger.info("exported SDPO dataset", extra={"path": path, "count": count})
    return count
