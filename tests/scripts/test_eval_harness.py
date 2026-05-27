#!/usr/bin/env python3
"""standalone tests for benchmark eval harness.

run: python3 tests/scripts/test_eval_harness.py
"""
import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dokumen.eval.types import EvalCase, EvalResult, EvalSummary
from dokumen.eval.metrics import accuracy, precision, recall, f1, expected_calibration_error
from dokumen.eval.datasets import load_dataset, save_dataset, list_datasets
from dokumen.eval.harness import EvalHarness

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


# ── EvalCase ──

print("── EvalCase ──")
c = EvalCase(id="c1", executor_response="30 day refund", expected_pass=True, assertion="mentions refund")
test("creation", c.id == "c1" and c.expected_pass is True)
test("defaults", c.tags == [] and c.metadata == {} and c.system_prompt == "")

d = c.to_dict()
test("to_dict keys", "id" in d and "expected_pass" in d and "executor_response" in d)

c2 = EvalCase.from_dict(d)
test("roundtrip id", c2.id == c.id)
test("roundtrip expected_pass", c2.expected_pass == c.expected_pass)
test("roundtrip assertion", c2.assertion == c.assertion)

c3 = EvalCase(id="c3", executor_response="x", expected_pass=False, tags=["policy", "refund"])
test("tags stored", c3.tags == ["policy", "refund"])

# ── EvalResult ──

print("\n── EvalResult ──")
r = EvalResult(case_id="c1", expected_pass=True, actual_pass=True, confidence=0.9)
test("correct (TP)", r.correct is True)
test("is_true_positive", r.is_true_positive is True)
test("not false_positive", r.is_false_positive is False)

r2 = EvalResult(case_id="c2", expected_pass=True, actual_pass=False, confidence=0.3)
test("incorrect (FN)", r2.correct is False)
test("is_false_negative", r2.is_false_negative is True)

r3 = EvalResult(case_id="c3", expected_pass=False, actual_pass=True, confidence=0.8)
test("is_false_positive", r3.is_false_positive is True)

r4 = EvalResult(case_id="c4", expected_pass=False, actual_pass=False, confidence=0.1)
test("is_true_negative", r4.is_true_negative is True)

d = r.to_dict()
test("to_dict has correct", d["correct"] is True)
test("to_dict has confidence", d["confidence"] == 0.9)

# ── metrics ──

print("\n── metrics ──")
test("accuracy 3/4", approx(accuracy(3, 4), 0.75))
test("accuracy 0/0", accuracy(0, 0) == 0.0)
test("accuracy 5/5", accuracy(5, 5) == 1.0)

test("precision 3/(3+1)", approx(precision(3, 1), 0.75))
test("precision 0/0", precision(0, 0) == 0.0)

test("recall 3/(3+2)", approx(recall(3, 2), 0.6))
test("recall 0/0", recall(0, 0) == 0.0)

test("f1 balanced", approx(f1(0.75, 0.6), 2 * 0.75 * 0.6 / (0.75 + 0.6)))
test("f1 zero", f1(0.0, 0.0) == 0.0)
test("f1 perfect", approx(f1(1.0, 1.0), 1.0))

# ── ECE ──

print("\n── expected calibration error ──")
# perfectly calibrated: confidence matches accuracy
test("empty ECE", expected_calibration_error([]) == 0.0)

# all correct with confidence 1.0 → ECE = 0
perfect = [(1.0, True)] * 10
test("perfect calibration", approx(expected_calibration_error(perfect), 0.0))

# all wrong with confidence 1.0 → ECE = 1.0
worst = [(1.0, False)] * 10
test("worst calibration", approx(expected_calibration_error(worst), 1.0))

# confidence 0.5, all correct → ECE = 0.5
half = [(0.5, True)] * 10
test("half confidence all correct", approx(expected_calibration_error(half), 0.5))

# mixed: some calibrated, some not
mixed = [(0.9, True), (0.9, True), (0.9, False), (0.1, False), (0.1, False)]
ece = expected_calibration_error(mixed)
test("mixed ECE > 0", ece > 0)
test("mixed ECE < 1", ece < 1)

# ── datasets ──

print("\n── datasets ──")
tmpdir = tempfile.mkdtemp()

cases = [
    EvalCase(id="d1", executor_response="yes", expected_pass=True, tags=["t1"]),
    EvalCase(id="d2", executor_response="no", expected_pass=False, tags=["t2"]),
]
path = os.path.join(tmpdir, "test_dataset.json")
save_dataset(cases, path, name="test-ds")
test("save creates file", os.path.exists(path))

loaded = load_dataset(path)
test("load count", len(loaded) == 2)
test("load id", loaded[0].id == "d1")
test("load expected_pass", loaded[1].expected_pass is False)
test("load tags", loaded[0].tags == ["t1"])

# list datasets
path2 = os.path.join(tmpdir, "another.json")
save_dataset([], path2)
found = list_datasets(tmpdir)
test("list finds 2", len(found) == 2)
test("list empty dir", list_datasets(os.path.join(tmpdir, "nope")) == [])

# load nonexistent
try:
    load_dataset("/nonexistent/path.json")
    test("nonexistent raises", False)
except FileNotFoundError:
    test("nonexistent raises", True)

# load invalid json
bad_path = os.path.join(tmpdir, "bad.json")
with open(bad_path, "w") as f:
    f.write("not json")
try:
    load_dataset(bad_path)
    test("invalid json raises", False)
except json.JSONDecodeError:
    test("invalid json raises", True)

# load bare list format
bare_path = os.path.join(tmpdir, "bare.json")
with open(bare_path, "w") as f:
    json.dump([{"id": "b1", "executor_response": "x", "expected_pass": True}], f)
bare = load_dataset(bare_path)
test("bare list format", len(bare) == 1 and bare[0].id == "b1")

# ── EvalHarness ──

print("\n── EvalHarness ──")


def mock_judge(executor_response, assertion, system_prompt="", user_prompt=""):
    """simple mock: passes if 'correct' is in the response."""
    is_pass = "correct" in executor_response.lower()
    return {
        "passed": is_pass,
        "confidence": 0.95 if is_pass else 0.1,
        "input_tokens": 100,
        "output_tokens": 50,
    }


harness = EvalHarness(judge_fn=mock_judge, judge_model="mock-v1", dataset_name="test")

cases = [
    EvalCase(id="h1", executor_response="this is correct", expected_pass=True),
    EvalCase(id="h2", executor_response="this is wrong", expected_pass=False),
    EvalCase(id="h3", executor_response="also correct answer", expected_pass=True),
    EvalCase(id="h4", executor_response="correct but expected fail", expected_pass=False),
]

summary = harness.run(cases)
test("total count", summary.total == 4)
test("accuracy", approx(summary.accuracy, 0.75))  # h1=TP, h2=TN, h3=TP, h4=FP
test("true positives", summary.true_positives == 2)
test("true negatives", summary.true_negatives == 1)
test("false positives", summary.false_positives == 1)
test("false negatives", summary.false_negatives == 0)
test("precision 2/3", approx(summary.precision, 2/3))
test("recall 2/2", approx(summary.recall, 1.0))
test("f1", summary.f1 > 0)
test("token tracking", summary.total_input_tokens == 400)
test("duration > 0 or 0", summary.total_duration_ms >= 0)

# ── harness with tags ──

print("\n── harness with tags ──")
tagged_cases = [
    EvalCase(id="t1", executor_response="correct", expected_pass=True, tags=["policy"]),
    EvalCase(id="t2", executor_response="wrong", expected_pass=False, tags=["api"]),
    EvalCase(id="t3", executor_response="correct", expected_pass=True, tags=["policy", "api"]),
]

summary = harness.run_with_cases(tagged_cases)
test("tag metrics computed", "policy" in summary.tag_metrics)
test("policy count", summary.tag_metrics["policy"]["count"] == 2)
test("api count", summary.tag_metrics["api"]["count"] == 2)

# tag filter
summary_filtered = harness.run_with_cases(tagged_cases, tags_filter=["api"])
test("tag filter reduces cases", summary_filtered.total == 2)

# ── harness error handling ──

print("\n── harness error handling ──")


def failing_judge(**kwargs):
    raise RuntimeError("judge crashed")


err_harness = EvalHarness(judge_fn=failing_judge, judge_model="broken")
result = err_harness.run_case(EvalCase(id="err", executor_response="x", expected_pass=True))
test("error captured", result.error is not None)
test("error defaults to fail", result.actual_pass is False)

# ── save results ──

print("\n── save results ──")
out_path = os.path.join(tmpdir, "results", "eval_out.json")
harness.save_results(summary, out_path)
test("results file exists", os.path.exists(out_path))
with open(out_path) as f:
    saved = json.load(f)
test("saved has accuracy", "accuracy" in saved)
test("saved has results", len(saved["results"]) == 3)

# ── EvalSummary to_dict ──

print("\n── EvalSummary to_dict ──")
d = summary.to_dict()
test("to_dict dataset_name", d["dataset_name"] == "test")
test("to_dict judge_model", d["judge_model"] == "mock-v1")
test("to_dict has tag_metrics", "tag_metrics" in d)
test("to_dict has timestamp", "timestamp" in d)

# ── empty dataset ──

print("\n── edge cases ──")
empty_summary = harness.run([])
test("empty dataset total=0", empty_summary.total == 0)
test("empty accuracy=0", empty_summary.accuracy == 0.0)
test("empty f1=0", empty_summary.f1 == 0.0)

# single case
single = harness.run([EvalCase(id="s1", executor_response="correct", expected_pass=True)])
test("single case accuracy=1", single.accuracy == 1.0)

print(f"\n{'='*50}")
print(f"eval harness: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
