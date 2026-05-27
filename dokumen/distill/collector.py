"""training data collector — extracts training examples from test runs.

sits in the pipeline as a callback. after each test run completes,
it extracts (prompt, response, reward) tuples and optionally
preference pairs when we have multiple runs of the same test.

data flow:
  test run → executor result + judge verdicts
  → collector.collect_from_run(executor_result, judge_verdicts)
  → TrainingExample(s) for GRPO
  → PreferencePair(s) for SDPO (when we have pass+fail for same prompt)
"""

import logging
import os
import uuid
from typing import Any, Dict, List

from .types import TrainingExample, PreferencePair, RewardSignal

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """collects training data from test runs.

    usage:
        collector = TrainingDataCollector(output_dir="./training_data")

        # after each test run:
        examples = collector.collect_from_run(
            test_id="oauth-test",
            executor_prompt="...",
            executor_response="...",
            judge_verdicts=[...],
        )

        # after enough data, export:
        collector.export_grpo("grpo_dataset.jsonl")
        collector.export_sdpo("sdpo_dataset.jsonl")
    """

    def __init__(self, output_dir: str = "./training_data"):
        self._output_dir = output_dir
        self._examples: List[TrainingExample] = []
        self._pairs: List[PreferencePair] = []
        # track pass/fail responses per prompt for pair generation
        self._prompt_responses: Dict[str, Dict[str, List[Dict]]] = {}
        os.makedirs(output_dir, exist_ok=True)

    def collect_from_run(
        self,
        test_id: str,
        executor_prompt: str,
        executor_response: str,
        judge_verdicts: List[Dict[str, Any]],
        model: str = "",
    ) -> List[TrainingExample]:
        """extract training examples from a single test run.

        creates one TrainingExample per judge verdict. the reward comes from:
        - binary: 1.0 if passed, 0.0 if failed
        - decomposed: fraction of sub-assertions that passed
        - both are stored so we can use either for training
        """
        examples = []

        for verdict in judge_verdicts:
            passed = verdict.get("passed", False)
            sub_assertions = verdict.get("sub_assertions", [])
            judge_id = verdict.get("judge_id", "unknown")

            # determine reward signal type and value
            if sub_assertions:
                n_passed = sum(1 for sa in sub_assertions if sa.get("passed", False))
                reward = n_passed / len(sub_assertions) if sub_assertions else 0.0
                signal = RewardSignal.DECOMPOSED
                sub_rewards = [
                    {"question": sa.get("question", ""), "passed": sa.get("passed", False), "reason": sa.get("reason", "")}
                    for sa in sub_assertions
                ]
            else:
                reward = 1.0 if passed else 0.0
                signal = RewardSignal.BINARY
                sub_rewards = []

            example = TrainingExample(
                id=f"te-{uuid.uuid4().hex[:12]}",
                prompt=executor_prompt,
                response=executor_response,
                reward=reward,
                reward_signal=signal,
                role="executor",
                test_id=test_id,
                judge_id=judge_id,
                model=model,
                sub_rewards=sub_rewards,
            )
            examples.append(example)
            self._examples.append(example)

            # track for preference pair generation
            prompt_key = f"{test_id}:{judge_id}"
            if prompt_key not in self._prompt_responses:
                self._prompt_responses[prompt_key] = {"pass": [], "fail": []}

            bucket = "pass" if passed else "fail"
            self._prompt_responses[prompt_key][bucket].append({
                "response": executor_response,
                "reward": reward,
                "sub_assertions": sub_assertions,
                "model": model,
            })

        logger.info(
            "collected training examples",
            extra={"test_id": test_id, "count": len(examples)},
        )

        # try to generate preference pairs
        self._generate_pairs(test_id)

        return examples

    def _generate_pairs(self, test_id: str) -> None:
        """generate preference pairs from accumulated pass/fail responses."""
        for prompt_key, responses in self._prompt_responses.items():
            if not responses["pass"] or not responses["fail"]:
                continue

            # pair each pass with each fail
            for good in responses["pass"]:
                for bad in responses["fail"]:
                    # build reason from sub-assertion differences
                    reason = self._build_reason(good.get("sub_assertions", []), bad.get("sub_assertions", []))

                    pair = PreferencePair(
                        id=f"pp-{uuid.uuid4().hex[:12]}",
                        prompt=f"[test: {test_id}]",  # we'd need the full prompt stored
                        chosen=good["response"],
                        rejected=bad["response"],
                        reason=reason,
                        chosen_sub_assertions=good.get("sub_assertions", []),
                        rejected_sub_assertions=bad.get("sub_assertions", []),
                        test_id=test_id,
                        model=good.get("model", ""),
                    )
                    self._pairs.append(pair)

            # clear after pairing to avoid duplicates
            responses["pass"] = []
            responses["fail"] = []

    def _build_reason(
        self,
        good_subs: List[Dict],
        bad_subs: List[Dict],
    ) -> str:
        """build a structured reason for why chosen > rejected."""
        if not good_subs or not bad_subs:
            return "chosen response passed judge evaluation, rejected did not"

        # find assertions that differ
        diffs = []
        good_map = {sa.get("question", ""): sa for sa in good_subs}
        for bad_sa in bad_subs:
            q = bad_sa.get("question", "")
            if q in good_map and good_map[q].get("passed") and not bad_sa.get("passed"):
                diffs.append(f"{q}: chosen passed ({good_map[q].get('reason', '')}), rejected failed ({bad_sa.get('reason', '')})")

        return "; ".join(diffs) if diffs else "chosen response scored higher on sub-assertions"

    @property
    def examples(self) -> List[TrainingExample]:
        return list(self._examples)

    @property
    def pairs(self) -> List[PreferencePair]:
        return list(self._pairs)

    @property
    def stats(self) -> Dict[str, Any]:
        """summary stats of collected data."""
        rewards = [e.reward for e in self._examples]
        return {
            "total_examples": len(self._examples),
            "total_pairs": len(self._pairs),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "pass_rate": sum(1 for r in rewards if r >= 0.5) / len(rewards) if rewards else 0.0,
            "decomposed_count": sum(1 for e in self._examples if e.reward_signal == RewardSignal.DECOMPOSED),
            "binary_count": sum(1 for e in self._examples if e.reward_signal == RewardSignal.BINARY),
        }
