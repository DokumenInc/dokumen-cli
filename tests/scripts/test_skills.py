#!/usr/bin/env python3
"""standalone tests for self-improving skill framework.

run: python3 tests/scripts/test_skills.py
"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dokumen.skills.types import SkillEntry, SkillCategory
from dokumen.skills.store import SkillStore
from dokumen.skills.extractor import SkillExtractor
from dokumen.skills.injector import SkillInjector

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


# ── SkillEntry ──

print("── SkillEntry ──")
s = SkillEntry(content="always check refresh tokens", category=SkillCategory.DOMAIN, tags=["oauth"])
test("creation", s.content == "always check refresh tokens")
test("id generated", s.id.startswith("sk-"))
test("category", s.category == SkillCategory.DOMAIN)
test("effectiveness 0/0", s.effectiveness == 0.0)

s.times_used = 10
s.times_helpful = 7
test("effectiveness 7/10", approx(s.effectiveness, 0.7))

d = s.to_dict()
test("to_dict has content", d["content"] == "always check refresh tokens")
test("to_dict category str", d["category"] == "domain")

s2 = SkillEntry.from_dict(d)
test("roundtrip content", s2.content == s.content)
test("roundtrip category", s2.category == s.category)
test("roundtrip tags", s2.tags == ["oauth"])

# ── SkillCategory ──

print("\n── SkillCategory ──")
test("EXECUTOR", SkillCategory.EXECUTOR.value == "executor")
test("JUDGE", SkillCategory.JUDGE.value == "judge")
test("TOOL_USE", SkillCategory.TOOL_USE.value == "tool_use")
test("DOMAIN", SkillCategory.DOMAIN.value == "domain")
test("ERROR_RECOVERY", SkillCategory.ERROR_RECOVERY.value == "error_recovery")

# ── SkillStore ──

print("\n── SkillStore ──")
tmpdir = tempfile.mkdtemp()
store = SkillStore(store_path=tmpdir)
test("empty store", len(store.get_all()) == 0)

store.add(SkillEntry(id="s1", content="tip one", category=SkillCategory.EXECUTOR, client_id="c1"))
store.add(SkillEntry(id="s2", content="tip two", category=SkillCategory.JUDGE, client_id="c1"))
store.add(SkillEntry(id="s3", content="tip three", category=SkillCategory.EXECUTOR, client_id="c2"))
test("3 skills", len(store.get_all()) == 3)

# client filter
test("client c1", len(store.get_all("c1")) == 2)
test("client c2", len(store.get_all("c2")) == 1)

# category filter
test("executor category", len(store.get_by_category("executor")) == 2)
test("judge category", len(store.get_by_category("judge")) == 1)
test("category + client", len(store.get_by_category("executor", "c1")) == 1)

# persistence
store2 = SkillStore(store_path=tmpdir)
test("reloaded", len(store2.get_all()) == 3)

# effectiveness tracking
store.record_usage("s1", was_helpful=True)
store.record_usage("s1", was_helpful=True)
store.record_usage("s1", was_helpful=False)
s1 = [s for s in store.get_all() if s.id == "s1"][0]
test("times_used", s1.times_used == 3)
test("times_helpful", s1.times_helpful == 2)
test("effectiveness 2/3", approx(s1.effectiveness, 2/3))

# prune
store.add(SkillEntry(id="bad", content="bad tip", times_used=10, times_helpful=1))
pruned = store.prune(min_uses=5, max_effectiveness=0.2)
test("pruned 1", pruned == 1)
test("bad removed", len([s for s in store.get_all() if s.id == "bad"]) == 0)

# delete
store.delete("s2")
test("deleted", len(store.get_all()) == 2)

# ── SkillStore search ──

print("\n── SkillStore search ──")
tmpdir2 = tempfile.mkdtemp()
ss = SkillStore(store_path=tmpdir2)
ss.add(SkillEntry(id="e1", content="oauth tip", embedding=[1.0, 0.0, 0.0]))
ss.add(SkillEntry(id="e2", content="api tip", embedding=[0.9, 0.1, 0.0]))
ss.add(SkillEntry(id="e3", content="unrelated", embedding=[0.0, 0.0, 1.0]))
ss.add(SkillEntry(id="e4", content="no embedding"))

results = ss.search(query_embedding=[1.0, 0.0, 0.0], top_k=2)
test("search returns 2", len(results) == 2)
test("most similar first", results[0][0].id == "e1")
test("skips no embedding", all(r[0].id != "e4" for r in results))

# ── SkillExtractor ──

print("\n── SkillExtractor ──")
extractor = SkillExtractor(use_llm=False)

# decomposed failure
verdicts = [{
    "judge_id": "accuracy",
    "passed": False,
    "confidence": 0.33,
    "sub_assertions": [
        {"question": "mentions OAuth?", "passed": True, "reason": "yes"},
        {"question": "mentions refresh tokens?", "passed": False, "reason": "missing"},
        {"question": "accurate expiry?", "passed": False, "reason": "wrong date"},
    ],
}]

skills = extractor.extract_from_verdicts(
    test_id="oauth-test",
    executor_prompt="check oauth docs",
    executor_response="oauth uses access tokens",
    judge_verdicts=verdicts,
)
test("extracted skills from decomposed", len(skills) >= 2)
test("skills have content", all(s.content for s in skills))
test("skills tagged", any(s.tags for s in skills))
test("source test set", all(s.source_test == "oauth-test" for s in skills))

# pass verdict → no skills
pass_verdicts = [{"judge_id": "acc", "passed": True, "confidence": 0.9}]
skills = extractor.extract_from_verdicts("t", "", "", pass_verdicts)
test("no skills from pass", len(skills) == 0)

# error verdict
error_verdicts = [{"judge_id": "broken", "passed": False, "error": True}]
skills = extractor.extract_from_verdicts("t", "", "", error_verdicts)
test("skill from error", len(skills) == 1)
test("error category is judge", skills[0].category == SkillCategory.JUDGE)

# low confidence
low_verdicts = [{"judge_id": "unsure", "passed": False, "confidence": 0.1, "failure_reason": "response was vague"}]
skills = extractor.extract_from_verdicts("t", "", "", low_verdicts)
test("skill from low confidence", len(skills) == 1)

# ── SkillInjector ──

print("\n── SkillInjector ──")
tmpdir3 = tempfile.mkdtemp()
inj_store = SkillStore(store_path=tmpdir3)
inj_store.add(SkillEntry(id="i1", content="always verify token expiry", times_used=5, times_helpful=4))
inj_store.add(SkillEntry(id="i2", content="check both access and refresh tokens", times_used=3, times_helpful=3))

injector = SkillInjector(store=inj_store, max_skills=5)

skills = injector.get_relevant_skills()
test("gets all skills", len(skills) == 2)
test("sorted by effectiveness", skills[0].id == "i2")  # 3/3 > 4/5

prompt = "you are an oauth judge"
injected = injector.inject_into_prompt(prompt, skills)
test("injection adds content", len(injected) > len(prompt))
test("contains tip", "token expiry" in injected)
test("original preserved", prompt in injected)

# empty injection
test("empty no change", injector.inject_into_prompt(prompt, []) == prompt)

# record outcome
injector.record_outcome(skills, test_passed=True)
s = [s for s in inj_store.get_all() if s.id == "i1"][0]
test("usage recorded", s.times_used == 6)
test("helpful recorded", s.times_helpful == 5)

# ── corrupted store ──

print("\n── corrupted store ──")
corrupt_dir = tempfile.mkdtemp()
with open(os.path.join(corrupt_dir, "skills.json"), "w") as f:
    f.write("{{bad json")
cs = SkillStore(store_path=corrupt_dir)
test("corrupted starts fresh", len(cs.get_all()) == 0)

print(f"\n{'='*50}")
print(f"skills: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
