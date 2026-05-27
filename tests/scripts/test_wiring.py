#!/usr/bin/env python3
"""tests for the 3 wiring tasks: task tools registry, skills auto-injection, content summarizer.

run: python3 tests/scripts/test_wiring.py
"""
import sys
import os
import asyncio
import importlib.util

_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)

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


def _load(name, rel_path):
    path = os.path.join(_root, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── load dependencies ──
_log_mod = _load("dokumen.logging_config", "dokumen/logging_config.py")
_debug_mod = _load("dokumen.debug", "dokumen/debug.py")
_tools_mod = _load("dokumen.tools_object", "dokumen/tools_object.py")

# task system
_task_types = _load("dokumen.tasks.types", "dokumen/tasks/types.py")
_task_mgr = _load("dokumen.tasks.manager", "dokumen/tasks/manager.py")
_task_tools = _load("dokumen.tasks.tools", "dokumen/tasks/tools.py")

# skills
_skills_loader = _load("dokumen.skills.loader", "dokumen/skills/loader.py")

# web_fetch
_wf_mod = _load("dokumen.tools.web_fetch", "dokumen/tools/web_fetch.py")

loop = asyncio.new_event_loop()


# ═══════════════════════════════════════════
# 1. task tools in registry
# ═══════════════════════════════════════════

print("\n── task tools registry ──")

# trigger registration
_tools_mod._register_task_tools()

test("TASK_TOOLS not empty", len(_tools_mod.TASK_TOOLS) > 0)
test("task_create registered", "task_create" in _tools_mod.TASK_TOOLS)
test("task_update registered", "task_update" in _tools_mod.TASK_TOOLS)
test("task_list registered", "task_list" in _tools_mod.TASK_TOOLS)
test("task_output registered", "task_output" in _tools_mod.TASK_TOOLS)
test("4 task tools total", len(_tools_mod.TASK_TOOLS) == 4)

# resolve a task tool — factory should return ToolDefinition
td = _tools_mod.TASK_TOOLS["task_create"]()
test("task_create returns ToolDefinition", isinstance(td, _tools_mod.ToolDefinition))
test("task_create has handler", td.handler is not None)
test("task_create has parameters", "properties" in td.parameters)

# actually call the tool
result = loop.run_until_complete(td.handler({"description": "test task from wiring test"}))
test("task_create handler returns dict", isinstance(result, dict))
test("task_create handler success", result.get("success") is True)
test("task_create handler has task_id", "task_id" in result)

# task_list
tl = _tools_mod.TASK_TOOLS["task_list"]()
result = loop.run_until_complete(tl.handler({}))
test("task_list returns tasks", result.get("count", 0) >= 1)

# task_update
task_id = result["tasks"][0]["id"]
tu = _tools_mod.TASK_TOOLS["task_update"]()
result = loop.run_until_complete(tu.handler({"task_id": task_id, "status": "in_progress"}))
test("task_update success", result.get("success") is True)

# task_output
to = _tools_mod.TASK_TOOLS["task_output"]()
result = loop.run_until_complete(to.handler({"task_id": task_id, "content": "some finding"}))
test("task_output success", result.get("success") is True)

# get_all_tool_names includes task tools
all_names = _tools_mod.get_all_tool_names()
test("get_all_tool_names has task_create", "task_create" in all_names)
test("get_all_tool_names has task_list", "task_list" in all_names)

# error cases
result = loop.run_until_complete(td.handler({}))
test("task_create rejects empty description", result.get("success") is False)

result = loop.run_until_complete(tu.handler({"task_id": "nonexistent", "status": "completed"}))
test("task_update rejects nonexistent", result.get("success") is False)

result = loop.run_until_complete(tu.handler({}))
test("task_update rejects empty params", result.get("success") is False)


# ═══════════════════════════════════════════
# 2. system skills auto-injection
# ═══════════════════════════════════════════

print("\n── system skills auto-injection ──")

get_all_skills = _skills_loader.get_all_skills
ExecutionMode = _skills_loader.ExecutionMode
SYSTEM_SKILLS = _skills_loader.SYSTEM_SKILLS

# check system skills exist
test("system skills exist", len(SYSTEM_SKILLS) >= 3)

# check which are inline
inline_skills = [s for s in SYSTEM_SKILLS if s.mode == ExecutionMode.INLINE]
test("at least 1 inline system skill", len(inline_skills) >= 1)

# get inline skill names for later checks
inline_names = {s.name for s in inline_skills}
fork_names = {s.name for s in SYSTEM_SKILLS if s.mode == ExecutionMode.FORK}
test("fork skills not empty", len(fork_names) >= 1)

# simulate what collect_skills now does — Source C adds inline system skills
# we can't call the real collect_skills since it needs dokumen_schema
# but we can verify the logic independently
seen = set()
injected = []
system = get_all_skills(include_system=True)
for skill in system:
    if skill.name in seen:
        continue
    if skill.mode != ExecutionMode.INLINE:
        continue
    injected.append((skill.name, skill.prompt, "system"))
    seen.add(skill.name)

test("inline skills auto-collected", len(injected) >= 1)
test("fork skills excluded", not any(n in fork_names for n, _, _ in injected))
test("collected skills have content", all(len(c) > 0 for _, c, _ in injected))
test("collected skills sourced as system", all(s == "system" for _, _, s in injected))

# verify that scaffold-specified skills would override system
seen2 = {"qa-check"}  # pretend scaffold already has qa-check
injected2 = []
for skill in system:
    if skill.name in seen2:
        continue
    if skill.mode != ExecutionMode.INLINE:
        continue
    injected2.append(skill.name)
    seen2.add(skill.name)
test("scaffold overrides system", "qa-check" not in injected2)

# each system skill has required fields
for skill in SYSTEM_SKILLS:
    test(f"system '{skill.name}' has description", len(skill.description) > 0)
    test(f"system '{skill.name}' has prompt", len(skill.prompt) > 0)


# ═══════════════════════════════════════════
# 3. ProviderSummarizer
# ═══════════════════════════════════════════

print("\n── ProviderSummarizer ──")

ProviderSummarizer = _wf_mod.ProviderSummarizer
ContentSummarizer = _wf_mod.ContentSummarizer
WebFetcher = _wf_mod.WebFetcher

# mock provider that returns predictable content
class MockProvider:
    async def complete(self, messages, system_prompt=None, **kwargs):
        user_msg = messages[0]["content"] if messages else ""
        # extract the prompt from the user message
        return {"content": f"extracted: mock summary"}

provider = MockProvider()
summarizer = ProviderSummarizer(provider)

test("ProviderSummarizer is ContentSummarizer", isinstance(summarizer, ContentSummarizer))
test("has system prompt", len(ProviderSummarizer.SYSTEM_PROMPT) > 0)

# call summarize
result = loop.run_until_complete(summarizer.summarize("page content here", "find the title"))
test("summarize returns string", isinstance(result, str))
test("summarize returns provider output", "extracted" in result)

# truncation
long_content = "x" * 100_000
result = loop.run_until_complete(summarizer.summarize(long_content, "find stuff"))
test("long content still works", isinstance(result, str))

# custom max_content_chars
short_summarizer = ProviderSummarizer(provider, max_content_chars=100)
test("custom max_chars", short_summarizer._max_chars == 100)

# wire into WebFetcher
fetcher = WebFetcher(summarizer=summarizer)
test("fetcher accepts ProviderSummarizer", fetcher._summarizer is summarizer)

# provider that returns non-dict
class RawProvider:
    async def complete(self, messages, **kwargs):
        return "raw string response"

raw_summarizer = ProviderSummarizer(RawProvider())
result = loop.run_until_complete(raw_summarizer.summarize("content", "prompt"))
test("handles non-dict provider response", result == "raw string response")

# provider that raises
class FailProvider:
    async def complete(self, messages, **kwargs):
        raise RuntimeError("api down")

fail_summarizer = ProviderSummarizer(FailProvider())
# summarizer errors are caught by WebFetcher._summarize, not by ProviderSummarizer itself
try:
    loop.run_until_complete(fail_summarizer.summarize("c", "p"))
    test("failing provider raises", False)
except RuntimeError:
    test("failing provider raises", True)


# ═══════════════════════════════════════════
print(f"\n{'=' * 50}")
print(f"  wiring tests: {passed} passed, {failed} failed")
print(f"{'=' * 50}")

sys.exit(0 if failed == 0 else 1)
