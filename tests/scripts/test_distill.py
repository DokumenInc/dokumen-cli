#!/usr/bin/env python3
"""standalone tests for distillation pipeline (GRPO/SDPO/DPO).

run: python3 tests/scripts/test_distill.py
"""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dokumen.distill.types import TrainingExample, PreferencePair, RewardSignal
from dokumen.distill.collector import TrainingDataCollector
from dokumen.distill.exporter import export_grpo_jsonl, export_sdpo_jsonl, export_dpo_jsonl

passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}")


def approx(a, b, tol=0.001):
    return abs(a - b) < tol


# ── TrainingExample ──

print("── TrainingExample ──")
te = TrainingExample(id="te-1", prompt="check oauth", response="tokens valid", reward=0.8)
test("creation", te.prompt == "check oauth" and te.reward == 0.8)
test("default signal binary", te.reward_signal == RewardSignal.BINARY)

d = te.to_dict()
test("to_dict", d["reward"] == 0.8 and d["reward_signal"] == "binary")

te2 = TrainingExample.from_dict(d)
test("roundtrip", te2.id == te.id and te2.reward == te.reward)

te3 = TrainingExample(id="te-2", prompt="x", response="y", reward=0.67, reward_signal=RewardSignal.DECOMPOSED)
test("decomposed signal", te3.reward_signal == RewardSignal.DECOMPOSED)

# ── PreferencePair ──

print("\n── PreferencePair ──")
pp = PreferencePair(id="pp-1", prompt="check docs", chosen="good answer", rejected="bad answer", reason="more complete")
test("creation", pp.chosen == "good answer")

d = pp.to_dict()
pp2 = PreferencePair.from_dict(d)
test("roundtrip", pp2.chosen == pp.chosen and pp2.reason == pp.reason)

# ── RewardSignal ──

print("\n── RewardSignal ──")
test("BINARY", RewardSignal.BINARY.value == "binary")
test("DECOMPOSED", RewardSignal.DECOMPOSED.value == "decomposed")
test("CONSENSUS", RewardSignal.CONSENSUS.value == "consensus")
test("CALIBRATED", RewardSignal.CALIBRATED.value == "calibrated")

# ── TrainingDataCollector ──

print("\n── TrainingDataCollector ──")
tmpdir = tempfile.mkdtemp()
collector = TrainingDataCollector(output_dir=tmpdir)

# binary verdict
examples = collector.collect_from_run(
    test_id="test-1",
    executor_prompt="check the refund policy",
    executor_response="refund within 30 days",
    judge_verdicts=[{"judge_id": "accuracy", "passed": True, "confidence": 0.9}],
    model="claude-sonnet",
)
test("1 example from binary pass", len(examples) == 1)
test("reward 1.0 for pass", examples[0].reward == 1.0)
test("signal binary", examples[0].reward_signal == RewardSignal.BINARY)

# decomposed verdict
examples = collector.collect_from_run(
    test_id="test-2",
    executor_prompt="check oauth",
    executor_response="uses access tokens",
    judge_verdicts=[{
        "judge_id": "completeness",
        "passed": False,
        "confidence": 0.33,
        "sub_assertions": [
            {"question": "mentions access tokens?", "passed": True, "reason": "yes"},
            {"question": "mentions refresh tokens?", "passed": False, "reason": "missing"},
            {"question": "mentions expiry?", "passed": False, "reason": "not covered"},
        ],
    }],
)
test("decomposed reward 1/3", approx(examples[0].reward, 1/3))
test("signal decomposed", examples[0].reward_signal == RewardSignal.DECOMPOSED)
test("sub_rewards stored", len(examples[0].sub_rewards) == 3)

# stats
stats = collector.stats
test("total examples 2", stats["total_examples"] == 2)
test("decomposed count 1", stats["decomposed_count"] == 1)
test("binary count 1", stats["binary_count"] == 1)

# ── preference pair generation ──

print("\n── preference pair generation ──")
collector2 = TrainingDataCollector(output_dir=tmpdir)

# first run: pass
collector2.collect_from_run(
    test_id="pair-test",
    executor_prompt="evaluate API docs",
    executor_response="good detailed answer with all endpoints",
    judge_verdicts=[{"judge_id": "j1", "passed": True, "confidence": 0.95}],
)
test("no pairs after 1 run", len(collector2.pairs) == 0)

# second run: fail (same test, same judge)
collector2.collect_from_run(
    test_id="pair-test",
    executor_prompt="evaluate API docs",
    executor_response="vague incomplete answer",
    judge_verdicts=[{"judge_id": "j1", "passed": False, "confidence": 0.2}],
)
test("pair generated", len(collector2.pairs) >= 1)
pair = collector2.pairs[0]
test("chosen is good", "good detailed" in pair.chosen)
test("rejected is bad", "vague incomplete" in pair.rejected)

# ── preference pair with decomposed subs ──

print("\n── preference pairs with sub-assertions ──")
collector3 = TrainingDataCollector(output_dir=tmpdir)

subs_good = [
    {"question": "q1", "passed": True, "reason": "yes"},
    {"question": "q2", "passed": True, "reason": "yes"},
]
subs_bad = [
    {"question": "q1", "passed": True, "reason": "yes"},
    {"question": "q2", "passed": False, "reason": "missing"},
]

collector3.collect_from_run("st", "p", "good resp", [{"judge_id": "j", "passed": True, "sub_assertions": subs_good}])
collector3.collect_from_run("st", "p", "bad resp", [{"judge_id": "j", "passed": False, "sub_assertions": subs_bad}])

test("sdpo pair generated", len(collector3.pairs) >= 1)
pair = collector3.pairs[0]
test("has structured reason", len(pair.reason) > 0)
test("chosen subs stored", len(pair.chosen_sub_assertions) == 2)
test("rejected subs stored", len(pair.rejected_sub_assertions) == 2)

# ── GRPO export ──

print("\n── GRPO export ──")
grpo_path = os.path.join(tmpdir, "grpo.jsonl")
count = export_grpo_jsonl(collector.examples, grpo_path)
test("grpo file created", os.path.exists(grpo_path))
test("grpo count 2", count == 2)

with open(grpo_path) as f:
    lines = f.readlines()
test("grpo 2 lines", len(lines) == 2)
first = json.loads(lines[0])
test("grpo has prompt", "prompt" in first)
test("grpo has reward", "reward" in first)
test("grpo has metadata", "metadata" in first)

# ── DPO export ──

print("\n── DPO export ──")
dpo_path = os.path.join(tmpdir, "dpo.jsonl")
count = export_dpo_jsonl(collector2.pairs, dpo_path)
test("dpo file created", os.path.exists(dpo_path))
test("dpo count", count >= 1)

with open(dpo_path) as f:
    first = json.loads(f.readline())
test("dpo has chosen", "chosen" in first)
test("dpo has rejected", "rejected" in first)
test("dpo minimal (no reason)", "reason" not in first)

# ── SDPO export ──

print("\n── SDPO export ──")
sdpo_path = os.path.join(tmpdir, "sdpo.jsonl")
count = export_sdpo_jsonl(collector3.pairs, sdpo_path)
test("sdpo file created", os.path.exists(sdpo_path))

with open(sdpo_path) as f:
    first = json.loads(f.readline())
test("sdpo has reason", "reason" in first)
test("sdpo has chosen_sub_assertions", "chosen_sub_assertions" in first)
test("sdpo has rejected_sub_assertions", "rejected_sub_assertions" in first)
test("sdpo has metadata", "metadata" in first)

# ── empty exports ──

print("\n── edge cases ──")
empty_path = os.path.join(tmpdir, "empty.jsonl")
test("empty grpo", export_grpo_jsonl([], empty_path) == 0)
test("empty dpo", export_dpo_jsonl([], empty_path) == 0)
test("empty sdpo", export_sdpo_jsonl([], empty_path) == 0)

# multiple judges per run
multi = TrainingDataCollector(output_dir=tmpdir)
multi.collect_from_run("t", "p", "r", [
    {"judge_id": "j1", "passed": True, "confidence": 0.9},
    {"judge_id": "j2", "passed": False, "confidence": 0.3},
])
test("multi-judge 2 examples", len(multi.examples) == 2)

print(f"\n{'='*50}")
print(f"distill: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
