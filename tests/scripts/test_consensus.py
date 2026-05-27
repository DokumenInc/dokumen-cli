#!/usr/bin/env python3
"""standalone tests for multi-judge consensus.

run: python3 tests/scripts/test_consensus.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dokumen.eval.consensus import ConsensusResult, run_consensus

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


# ── ConsensusResult ──

print("── ConsensusResult ──")
cr = ConsensusResult(passed=True, confidence=1.0, agreement_ratio=1.0, runs=[{"run": 1, "passed": True}])
test("creation", cr.passed is True and cr.confidence == 1.0)
d = cr.to_dict()
test("to_dict has num_runs", d["num_runs"] == 1)
test("to_dict has agreement_ratio", d["agreement_ratio"] == 1.0)

# ── unanimous pass ──

print("\n── unanimous pass ──")
call_count = 0


def always_pass(**kwargs):
    global call_count
    call_count += 1
    return {"passed": True, "confidence": 0.95, "input_tokens": 100, "output_tokens": 50}


call_count = 0
result = run_consensus(always_pass, n_runs=3)
test("called 3 times", call_count == 3)
test("passed", result.passed is True)
test("confidence 1.0", approx(result.confidence, 1.0))
test("agreement 1.0", approx(result.agreement_ratio, 1.0))
test("3 runs", len(result.runs) == 3)
test("tokens tracked", result.total_input_tokens == 300)

# ── unanimous fail ──

print("\n── unanimous fail ──")


def always_fail(**kwargs):
    return {"passed": False, "confidence": 0.1}


result = run_consensus(always_fail, n_runs=3)
test("failed", result.passed is False)
test("confidence 1.0 (unanimous)", approx(result.confidence, 1.0))
test("agreement 1.0", approx(result.agreement_ratio, 1.0))

# ── 2/3 pass (majority pass) ──

print("\n── 2/3 majority pass ──")
counter = 0


def two_of_three(**kwargs):
    global counter
    counter += 1
    return {"passed": counter % 3 != 0, "confidence": 0.7}


counter = 0
result = run_consensus(two_of_three, n_runs=3)
test("passed (2/3 >= 0.5)", result.passed is True)
test("agreement 2/3", approx(result.agreement_ratio, 2/3))
test("confidence = agreement", approx(result.confidence, 2/3))

# ── 1/3 pass (majority fail) ──

print("\n── 1/3 majority fail ──")
counter2 = 0


def one_of_three(**kwargs):
    global counter2
    counter2 += 1
    return {"passed": counter2 % 3 == 1, "confidence": 0.5}


counter2 = 0
result = run_consensus(one_of_three, n_runs=3)
test("failed (1/3 < 0.5)", result.passed is False)
test("agreement 2/3", approx(result.agreement_ratio, 2/3))

# ── custom threshold ──

print("\n── custom threshold ──")
counter3 = 0


def half_pass(**kwargs):
    global counter3
    counter3 += 1
    return {"passed": counter3 % 2 == 1, "confidence": 0.6}


# 2/4 pass, threshold=0.75 → fail
counter3 = 0
result = run_consensus(half_pass, n_runs=4, threshold=0.75)
test("fails high threshold", result.passed is False)

# 2/4 pass, threshold=0.25 → pass
counter3 = 0
result = run_consensus(half_pass, n_runs=4, threshold=0.25)
test("passes low threshold", result.passed is True)

# ── error handling ──

print("\n── error handling ──")
err_count = 0


def sometimes_crash(**kwargs):
    global err_count
    err_count += 1
    if err_count == 2:
        raise RuntimeError("judge crashed")
    return {"passed": True, "confidence": 0.9}


err_count = 0
result = run_consensus(sometimes_crash, n_runs=3)
test("handles errors gracefully", len(result.runs) == 3)
test("error recorded", result.runs[1]["error"] is not None)
test("non-error runs pass", result.runs[0]["passed"] is True)
test("error run fails", result.runs[1]["passed"] is False)
# 2/3 pass → overall pass
test("overall passes despite error", result.passed is True)

# ── all crash ──

print("\n── all crash ──")


def always_crash(**kwargs):
    raise RuntimeError("boom")


result = run_consensus(always_crash, n_runs=3)
test("all crash → fail", result.passed is False)
test("all errors recorded", all(r["error"] is not None for r in result.runs))

# ── n_runs=1 (degenerate) ──

print("\n── single run ──")
result = run_consensus(always_pass, n_runs=1)
test("single pass", result.passed is True)
test("single confidence 1.0", approx(result.confidence, 1.0))

# ── n_runs=0 ──

print("\n── zero runs ──")
result = run_consensus(always_pass, n_runs=0)
test("zero runs → fail", result.passed is False)
test("zero runs confidence 0", result.confidence == 0.0)
test("zero runs empty", len(result.runs) == 0)

# ── kwargs forwarded ──

print("\n── kwargs forwarding ──")
received_kwargs = {}


def capture_kwargs(**kwargs):
    received_kwargs.update(kwargs)
    return {"passed": True, "confidence": 0.9}


run_consensus(capture_kwargs, n_runs=1, executor_response="test", assertion="check")
test("kwargs forwarded", received_kwargs.get("executor_response") == "test")
test("assertion forwarded", received_kwargs.get("assertion") == "check")

print(f"\n{'='*50}")
print(f"consensus: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
