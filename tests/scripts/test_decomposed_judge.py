#!/usr/bin/env python3
"""standalone tests for decomposed binary judging.

run: python3 tests/scripts/test_decomposed_judge.py
"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dokumen.sdk.types import SubAssertion, JudgeVerdict
from dokumen.sdk.judge import parse_decomposed_verdict, parse_verdict

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


print("── SubAssertion ──")
sa = SubAssertion(question="mentions OAuth?", passed=True, reason="yes")
test("creation", sa.question == "mentions OAuth?" and sa.passed is True)
d = sa.to_dict()
test("to_dict keys", set(d.keys()) == {"question", "passed", "reason"})
test("to_dict values", d["question"] == "mentions OAuth?" and d["passed"] is True)

print("\n── JudgeVerdict with sub_assertions ──")
subs = [
    SubAssertion(question="q1", passed=True, reason="yes"),
    SubAssertion(question="q2", passed=False, reason="no"),
]
v = JudgeVerdict(judge_id="test", passed=False, sub_assertions=subs, confidence=0.5)
test("sub_assertions stored", len(v.sub_assertions) == 2)
test("sub[0] passed", v.sub_assertions[0].passed is True)
test("sub[1] failed", v.sub_assertions[1].passed is False)

d = v.to_dict()
test("to_dict has sub_assertions", "sub_assertions" in d)
test("to_dict sub count", len(d["sub_assertions"]) == 2)
test("to_dict sub[0] is dict", d["sub_assertions"][0]["question"] == "q1")

print("\n── backward compat (no sub_assertions) ──")
v2 = JudgeVerdict(judge_id="old", passed=True, confidence=0.9)
test("empty list default", v2.sub_assertions == [])
test("to_dict empty list", v2.to_dict()["sub_assertions"] == [])

print("\n── parse_decomposed_verdict ──")

# valid decomposed response
resp = json.dumps({
    "sub_assertions": [
        {"question": "mentions OAuth?", "passed": True, "reason": "yes"},
        {"question": "mentions API keys?", "passed": True, "reason": "yes"},
        {"question": "accurate?", "passed": False, "reason": "wrong expiry"},
    ],
})
result = parse_decomposed_verdict(resp)
test("parses valid response", result is not None)
subs, conf = result
test("3 sub-assertions", len(subs) == 3)
test("confidence 2/3", approx(conf, 2/3))
test("sub[0] is SubAssertion", isinstance(subs[0], SubAssertion))
test("sub[2] failed", subs[2].passed is False)

# code fence
resp_fence = (
    "evaluation:\n\n```json\n"
    + json.dumps({"sub_assertions": [
        {"question": "q1", "passed": True, "reason": "ok"},
        {"question": "q2", "passed": True, "reason": "ok"},
    ]})
    + "\n```"
)
result = parse_decomposed_verdict(resp_fence)
test("parses code fence", result is not None)
subs, conf = result
test("fence: 2 subs", len(subs) == 2)
test("fence: confidence 1.0", conf == 1.0)

# all fail
resp_fail = json.dumps({"sub_assertions": [
    {"question": "q1", "passed": False, "reason": "no"},
    {"question": "q2", "passed": False, "reason": "no"},
]})
subs, conf = parse_decomposed_verdict(resp_fail)
test("all fail confidence 0.0", conf == 0.0)

# empty list
subs, conf = parse_decomposed_verdict(json.dumps({"sub_assertions": []}))
test("empty list confidence 0.0", conf == 0.0)
test("empty list subs empty", subs == [])

# garbage
test("garbage returns None", parse_decomposed_verdict("not json") is None)
test("None returns None", parse_decomposed_verdict(None) is None)
test("missing key returns None", parse_decomposed_verdict(json.dumps({"verdict": "PASS"})) is None)

print("\n── fallback to regular verdict ──")
regular = json.dumps({"verdict": "PASS", "confidence": 0.85, "reason": "looks good"})
test("regular verdict still parses", parse_decomposed_verdict(regular) is None)
v = parse_verdict(regular)
test("parse_verdict works", v is not None and v.passed is True and v.confidence == 0.85)

print(f"\n{'='*50}")
print(f"decomposed judging: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
