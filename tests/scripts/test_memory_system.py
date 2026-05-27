#!/usr/bin/env python3
"""standalone tests for three-tier memory system.

run: python3 tests/scripts/test_memory_system.py
"""
import sys
import os
import json
import tempfile
import shutil
import asyncio
import time

_root = os.path.join(os.path.dirname(__file__), "..", "..")

import importlib.util


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_root, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# load in dependency order
schemas_mod = _load("dokumen.memory.schemas", "dokumen/memory/schemas.py")
base_mod = _load("dokumen.memory.base", "dokumen/memory/base.py")
session_mod = _load("dokumen.memory.session_memory", "dokumen/memory/session_memory.py")
extractor_mod = _load("dokumen.memory.extractor", "dokumen/memory/extractor.py")
memdir_mod = _load("dokumen.memory.memdir", "dokumen/memory/memdir.py")

# pull in the classes we need
Memory = schemas_mod.Memory
MemoryOperation = schemas_mod.MemoryOperation
SessionEntry = session_mod.SessionEntry
SessionMemory = session_mod.SessionMemory
DefaultSummarizer = session_mod.DefaultSummarizer
ExtractionResult = extractor_mod.ExtractionResult
MemoryExtractor = extractor_mod.MemoryExtractor
MemoryType = memdir_mod.MemoryType
MemdirEntry = memdir_mod.MemdirEntry
MemdirStore = memdir_mod.MemdirStore

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


# ── SessionEntry ──

print("── SessionEntry ──")

entry = SessionEntry(content="found oauth docs at docs/auth.md")
test("content stored", entry.content == "found oauth docs at docs/auth.md")
test("default category general", entry.category == "general")
test("timestamp auto-set", entry.timestamp > 0)
test("metadata default empty", entry.metadata == {})

entry2 = SessionEntry(content="tool returned 200 lines", category="tool_result", metadata={"tool": "read_file"})
test("custom category", entry2.category == "tool_result")
test("metadata stored", entry2.metadata["tool"] == "read_file")

# to_dict / from_dict roundtrip
d = entry2.to_dict()
test("to_dict has content", d["content"] == "tool returned 200 lines")
test("to_dict has category", d["category"] == "tool_result")
test("to_dict has timestamp", "timestamp" in d)
test("to_dict has metadata", d["metadata"]["tool"] == "read_file")

entry3 = SessionEntry.from_dict(d)
test("from_dict content", entry3.content == entry2.content)
test("from_dict category", entry3.category == entry2.category)
test("from_dict metadata", entry3.metadata == entry2.metadata)
test("from_dict timestamp", entry3.timestamp == entry2.timestamp)

# from_dict with missing optional keys
minimal = SessionEntry.from_dict({"content": "bare"})
test("from_dict minimal content", minimal.content == "bare")
test("from_dict minimal category default", minimal.category == "general")
test("from_dict minimal metadata default", minimal.metadata == {})

# ── SessionMemory ──

print("\n── SessionMemory ──")

sm = SessionMemory()
test("starts empty", sm.entry_count == 0)
test("total_chars zero", sm.total_chars == 0)

sm.add("checked refund policy")
test("entry_count after add", sm.entry_count == 1)
test("total_chars after add", sm.total_chars == len("checked refund policy"))

sm.add("read_file returned 500 lines", category="tool_result")
sm.add("executor found 3 relevant files", category="finding")
test("entry_count 3", sm.entry_count == 3)

# category filtering
findings = sm.get_entries(category="finding")
test("category filter finds 1 finding", len(findings) == 1)
test("finding content correct", findings[0].content == "executor found 3 relevant files")

tool_results = sm.get_entries(category="tool_result")
test("tool_result filter finds 1", len(tool_results) == 1)

all_entries = sm.get_entries()
test("no filter returns all", len(all_entries) == 3)

# unknown category returns empty
test("unknown category empty", sm.get_entries(category="nonexistent") == [])

# needs_compaction — entry count threshold
sm_tight = SessionMemory(max_entries=3)
test("not compaction yet at 0", not sm_tight.needs_compaction)
sm_tight.add("a")
sm_tight.add("b")
test("not compaction at 2 of 3", not sm_tight.needs_compaction)
sm_tight.add("c")
test("needs_compaction at threshold", sm_tight.needs_compaction)

# needs_compaction — char count threshold
sm_chars = SessionMemory(max_chars=20)
sm_chars.add("hello")
test("not over char limit yet", not sm_chars.needs_compaction)
sm_chars.add("this pushes it over twenty chars total")
test("needs_compaction over char limit", sm_chars.needs_compaction)

# compact() — needs more than 5 entries to trigger
print("\n── SessionMemory.compact ──")

sm_compact = SessionMemory()
for i in range(15):
    sm_compact.add(f"entry number {i}", category="general" if i % 2 == 0 else "tool_result")

before_count = sm_compact.entry_count
summary = asyncio.run(sm_compact.compact())
test("compact reduces entries", sm_compact.entry_count < before_count)
test("compact returns summary string", isinstance(summary, str) and len(summary) > 0)
test("compact keeps recent entries", sm_compact.entry_count >= 1)

# the old entries should be summarized — verify summary is retained in context
ctx = sm_compact.get_context()
test("context includes summarized section", "summarized" in ctx or "previous" in ctx)
test("context includes current session", "current" in ctx)

# compact with few entries (<=5) does nothing
sm_small = SessionMemory()
sm_small.add("only one")
result = asyncio.run(sm_small.compact())
test("compact noop when <=5 entries", result == "")
test("entries unchanged after noop compact", sm_small.entry_count == 1)

# get_context with no entries
sm_empty = SessionMemory()
ctx_empty = sm_empty.get_context()
test("empty context is empty string", ctx_empty == "")

# get_context truncates when too long
sm_long = SessionMemory()
for i in range(5):
    sm_long.add("x" * 3000)
ctx_truncated = sm_long.get_context(max_chars=100)
test("context respects max_chars", len(ctx_truncated) <= 100)

# clear()
sm_clear = SessionMemory()
sm_clear.add("first entry")
sm_clear.add("second entry")
sm_clear.clear()
test("clear removes entries", sm_clear.entry_count == 0)
test("clear resets total_chars", sm_clear.total_chars == 0)
test("clear removes summaries", sm_clear.get_context() == "")

# to_dict / from_dict roundtrip
print("\n── SessionMemory serialization ──")

sm_serial = SessionMemory()
sm_serial.add("remember this", category="finding")
sm_serial.add("tool output here", category="tool_result")

sd = sm_serial.to_dict()
test("to_dict has entries", len(sd["entries"]) == 2)
test("to_dict has summaries", "summaries" in sd)
test("to_dict has total_chars", sd["total_chars"] > 0)

sm_loaded = SessionMemory.from_dict(sd)
test("from_dict restores entry count", sm_loaded.entry_count == 2)
test("from_dict restores total_chars", sm_loaded.total_chars == sd["total_chars"])
test("from_dict restores content", sm_loaded.get_entries()[0].content == "remember this")
test("from_dict restores category", sm_loaded.get_entries()[1].category == "tool_result")

# from_dict with summaries
sm_with_summary = SessionMemory()
for i in range(12):
    sm_with_summary.add(f"pre-compact entry {i}")
asyncio.run(sm_with_summary.compact())
sd2 = sm_with_summary.to_dict()
sm_restored = SessionMemory.from_dict(sd2)
test("from_dict restores summaries", sd2["summaries"] == sm_restored._summaries)

# ── ExtractionResult ──

print("\n── ExtractionResult ──")

er = ExtractionResult(
    memories=[Memory(id="m1", content="test content")],
    skipped=2,
    source_test="auth-test",
    extraction_time=0.05,
)
test("memories stored", len(er.memories) == 1)
test("skipped stored", er.skipped == 2)
test("source_test stored", er.source_test == "auth-test")
test("extraction_time stored", er.extraction_time == 0.05)

erd = er.to_dict()
test("to_dict memories", len(erd["memories"]) == 1)
test("to_dict skipped", erd["skipped"] == 2)
test("to_dict source_test", erd["source_test"] == "auth-test")
test("to_dict extraction_time", erd["extraction_time"] == 0.05)

# ── MemoryExtractor ──

print("\n── MemoryExtractor ──")

extractor = MemoryExtractor()

# judge failure with long enough reason
run_fail = {
    "test_id": "refund-policy-test",
    "passed": False,
    "judges": [
        {
            "judge_id": "accuracy",
            "passed": False,
            "reason": "the executor did not mention the 30-day refund window correctly",
            "confidence": 0.8,
        }
    ],
    "tool_calls": [],
}
result = extractor.extract_from_run(run_fail)
test("extracts failure memory", len(result.memories) >= 1)
test("source_test recorded", result.source_test == "refund-policy-test")
test("memory content has test_id", "refund-policy-test" in result.memories[0].content)
test("memory content has judge_id", "accuracy" in result.memories[0].content)
test("metadata type is failure_pattern", result.memories[0].metadata["type"] == "failure_pattern")
test("metadata has confidence", result.memories[0].metadata["confidence"] == 0.8)
test("extraction_time set", result.extraction_time > 0)

# judge failure with short reason — should be skipped
run_short_reason = {
    "test_id": "some-test",
    "passed": False,
    "judges": [
        {
            "judge_id": "clarity",
            "passed": False,
            "reason": "bad",  # too short
        }
    ],
    "tool_calls": [],
}
result_short = extractor.extract_from_run(run_short_reason)
test("short reason skipped", len(result_short.memories) == 0)
test("skipped count incremented", result_short.skipped == 1)

# sub-assertion failures
run_sub = {
    "test_id": "oauth-test",
    "passed": False,
    "judges": [
        {
            "judge_id": "completeness",
            "passed": False,
            "reason": "ok",  # too short for main failure
            "sub_assertions": [
                {
                    "question": "does it mention refresh tokens?",
                    "passed": False,
                    "reason": "the executor never mentioned refresh token rotation in its response",
                },
                {
                    "question": "ok?",
                    "passed": False,
                    "reason": "no",  # too short
                },
                {
                    "question": "access token expiry mentioned?",
                    "passed": True,
                    "reason": "yes, correctly stated",
                },
            ],
        }
    ],
    "tool_calls": [],
}
result_sub = extractor.extract_from_run(run_sub)
# one sub-assertion has long enough reason, one is too short, one passed
test("sub-assertion memory extracted", len(result_sub.memories) >= 1)
test("sub-assertion memory type", result_sub.memories[0].metadata["type"] == "sub_assertion_failure")
test("sub-assertion skipped short reason", result_sub.skipped >= 1)

# successful run with tool calls — extracts tool pattern
run_success = {
    "test_id": "docs-check",
    "passed": True,
    "judges": [],
    "tool_calls": [
        {"tool": "glob", "args": {}, "result": "ok"},
        {"tool": "read_file", "args": {}, "result": "ok"},
        {"tool": "read_file", "args": {}, "result": "ok"},  # duplicate — deduplicated
        {"tool": "search_files", "args": {}, "result": "ok"},
    ],
}
result_tools = extractor.extract_from_run(run_success)
test("tool pattern memory extracted", len(result_tools.memories) == 1)
test("tool pattern type", result_tools.memories[0].metadata["type"] == "tool_pattern")
test("tool_count in metadata", result_tools.memories[0].metadata["tool_count"] == 4)
test("sequence in content", "→" in result_tools.memories[0].content)

# no tool memories on failure
run_fail_no_tools = {
    "test_id": "no-tool-test",
    "passed": False,
    "judges": [],
    "tool_calls": [
        {"tool": "read_file", "args": {}, "result": "ok"},
        {"tool": "read_file", "args": {}, "result": "ok"},
    ],
}
result_fail_no_tools = extractor.extract_from_run(run_fail_no_tools)
test("no tool pattern on failure", len(result_fail_no_tools.memories) == 0)

# fewer than 2 distinct tool calls — no tool memory
run_single_tool = {
    "test_id": "single-tool",
    "passed": True,
    "judges": [],
    "tool_calls": [
        {"tool": "read_file", "args": {}, "result": "ok"},
    ],
}
result_single = extractor.extract_from_run(run_single_tool)
test("single tool call no pattern", len(result_single.memories) == 0)

# all same tool (dedups to 1) — no pattern
run_repeated = {
    "test_id": "repeated-tool",
    "passed": True,
    "judges": [],
    "tool_calls": [
        {"tool": "read_file", "args": {}},
        {"tool": "read_file", "args": {}},
        {"tool": "read_file", "args": {}},
    ],
}
result_repeated = extractor.extract_from_run(run_repeated)
test("all same tool deduped to 1 — no pattern", len(result_repeated.memories) == 0)

# empty run
run_empty = {"test_id": "empty", "passed": True, "judges": [], "tool_calls": []}
result_empty = extractor.extract_from_run(run_empty)
test("empty run no memories", len(result_empty.memories) == 0)

# run with no failures → no failure memories
run_all_pass = {
    "test_id": "all-pass",
    "passed": True,
    "judges": [
        {"judge_id": "accuracy", "passed": True, "reason": "all good here and very clear"},
    ],
    "tool_calls": [],
}
result_pass = extractor.extract_from_run(run_all_pass)
test("passing judges produce no memories", len(result_pass.memories) == 0)

# max_memories_per_run cap
extractor_tight = MemoryExtractor(min_reason_length=5, max_memories_per_run=2)
many_failures = {
    "test_id": "flood-test",
    "passed": False,
    "judges": [
        {"judge_id": f"j{i}", "passed": False, "reason": f"failure reason for judge {i} that is long enough"}
        for i in range(10)
    ],
    "tool_calls": [],
}
result_capped = extractor_tight.extract_from_run(many_failures)
test("capped at max_memories_per_run", len(result_capped.memories) == 2)
test("excess counted as skipped", result_capped.skipped >= 8)

# min_reason_length filtering via constructor
extractor_strict = MemoryExtractor(min_reason_length=100)
run_medium_reason = {
    "test_id": "medium-reason",
    "passed": False,
    "judges": [
        {"judge_id": "judge1", "passed": False, "reason": "this reason is thirty chars long..."},
    ],
    "tool_calls": [],
}
result_strict = extractor_strict.extract_from_run(run_medium_reason)
test("strict min_reason_length filters medium reason", len(result_strict.memories) == 0)

# ── MemoryType ──

print("\n── MemoryType ──")

test("USER value", MemoryType.USER.value == "user")
test("FEEDBACK value", MemoryType.FEEDBACK.value == "feedback")
test("PROJECT value", MemoryType.PROJECT.value == "project")
test("REFERENCE value", MemoryType.REFERENCE.value == "reference")
test("from string user", MemoryType("user") == MemoryType.USER)
test("from string reference", MemoryType("reference") == MemoryType.REFERENCE)

# ── MemdirEntry ──

print("\n── MemdirEntry ──")

mde = MemdirEntry(
    id="user_mabid",
    name="mo's background",
    description="ml research lead at dokumen",
    memory_type=MemoryType.USER,
    content="mo is an ml researcher at eth zurich, currently ml research lead at dokumen.",
    filename="user_mabid.md",
)
test("id stored", mde.id == "user_mabid")
test("name stored", mde.name == "mo's background")
test("memory_type stored", mde.memory_type == MemoryType.USER)
test("content stored", "eth zurich" in mde.content)
test("filename stored", mde.filename == "user_mabid.md")
test("created_at set", mde.created_at > 0)
test("updated_at set", mde.updated_at > 0)

# to_markdown
md_text = mde.to_markdown()
test("markdown has frontmatter delimiter", md_text.startswith("---"))
test("markdown has name field", "name:" in md_text)
test("markdown has type field", "type:" in md_text)
test("markdown has content after frontmatter", "eth zurich" in md_text)
test("markdown type value correct", "user" in md_text)

# from_markdown roundtrip
mde2 = MemdirEntry.from_markdown(md_text, "user_mabid.md")
test("from_markdown not None", mde2 is not None)
test("from_markdown id from filename", mde2.id == "user_mabid")
test("from_markdown name", mde2.name == "mo's background")
test("from_markdown type", mde2.memory_type == MemoryType.USER)
test("from_markdown content", "eth zurich" in mde2.content)
test("from_markdown description", mde2.description == "ml research lead at dokumen")

# from_markdown bad input returns None
test("from_markdown no frontmatter → None", MemdirEntry.from_markdown("just plain text", "f.md") is None)
test("from_markdown empty string → None", MemdirEntry.from_markdown("", "f.md") is None)
test("from_markdown incomplete delimiter → None", MemdirEntry.from_markdown("---\nname: x\n", "f.md") is None)

# from_markdown with unknown type falls back to PROJECT
unknown_type_md = "---\nname: test\ndescription: d\ntype: unknown_type\ncreated_at: 0\nupdated_at: 0\n---\n\ncontent here\n"
mde_fallback = MemdirEntry.from_markdown(unknown_type_md, "test.md")
test("unknown type falls back to PROJECT", mde_fallback is not None and mde_fallback.memory_type == MemoryType.PROJECT)

# to_memory conversion
mem_obj = mde.to_memory()
test("to_memory id", mem_obj.id == mde.id)
test("to_memory content", mem_obj.content == mde.content)
test("to_memory metadata type", mem_obj.metadata["type"] == "user")
test("to_memory metadata name", mem_obj.metadata["name"] == "mo's background")

# ── MemdirStore ──

print("\n── MemdirStore ──")

tmpdir = tempfile.mkdtemp()
store = MemdirStore(tmpdir)
test("directory created", os.path.isdir(tmpdir))
test("directory property", store.directory == tmpdir)

# save and load
e1 = MemdirEntry(
    id="proj_deadline",
    name="deadline info",
    description="source verification deadline end of weekend",
    memory_type=MemoryType.PROJECT,
    content="source verification agent must be done by end of weekend.",
    filename="proj_deadline.md",
)
saved_path = store.save(e1)
test("save returns path", os.path.exists(saved_path))
test("save creates file", os.path.exists(os.path.join(tmpdir, "proj_deadline.md")))

loaded = store.load("proj_deadline.md")
test("load returns entry", loaded is not None)
test("load content correct", "source verification" in loaded.content)
test("load name correct", loaded.name == "deadline info")

# load nonexistent returns None
test("load missing → None", store.load("no_such_file.md") is None)

# load_all
e2 = MemdirEntry(
    id="ref_rlm",
    name="RLM paper",
    description="RLVR reference",
    memory_type=MemoryType.REFERENCE,
    content="RLM paper discusses RLVR and SDPO distillation techniques.",
    filename="ref_rlm.md",
)
store.save(e2)
all_entries = store.load_all()
test("load_all returns 2 entries", len(all_entries) == 2)
test("load_all excludes MEMORY.md index", all(e.filename != "MEMORY.md" for e in all_entries))

# update_index creates MEMORY.md
index_path = os.path.join(tmpdir, "MEMORY.md")
test("MEMORY.md exists after save", os.path.exists(index_path))
with open(index_path) as f:
    index_content = f.read()
test("index has Memory Index header", "# Memory Index" in index_content)
test("index has project section", "project" in index_content)
test("index has reference section", "reference" in index_content)
test("index links to file", "proj_deadline.md" in index_content)

# find_by_type
projects = store.find_by_type(MemoryType.PROJECT)
test("find_by_type project", len(projects) == 1)
test("find_by_type content correct", projects[0].id == "proj_deadline")

refs = store.find_by_type(MemoryType.REFERENCE)
test("find_by_type reference", len(refs) == 1)

users = store.find_by_type(MemoryType.USER)
test("find_by_type user empty", len(users) == 0)

# search_by_content
results = store.search_by_content("RLVR distillation")
test("search finds reference entry", len(results) >= 1)
test("most relevant first", results[0].id == "ref_rlm")

results_partial = store.search_by_content("deadline")
test("search finds project entry", len(results_partial) >= 1)

results_none = store.search_by_content("xyzzy_not_present_anywhere")
test("search no match returns empty", results_none == [])

# search_by_content max_results cap
store2 = MemdirStore(tempfile.mkdtemp())
for i in range(10):
    store2.save(MemdirEntry(
        id=f"m{i}", name=f"memory {i}", description="test memory",
        memory_type=MemoryType.PROJECT, content=f"this is test memory entry {i}",
        filename=f"m{i}.md",
    ))
capped = store2.search_by_content("test memory", max_results=3)
test("search_by_content respects max_results", len(capped) <= 3)

# delete
store.delete("proj_deadline.md")
test("delete removes file", not os.path.exists(os.path.join(tmpdir, "proj_deadline.md")))
after_delete = store.load_all()
test("load_all after delete has 1 entry", len(after_delete) == 1)

# delete nonexistent returns False
test("delete nonexistent returns False", store.delete("ghost.md") is False)

# delete existing returns True
e3 = MemdirEntry(
    id="tmp_e", name="tmp", description="d", memory_type=MemoryType.FEEDBACK,
    content="temporary feedback entry.", filename="tmp_e.md",
)
store.save(e3)
test("delete existing returns True", store.delete("tmp_e.md") is True)

# ── MemoryStore protocol via MemdirStore ──

print("\n── MemoryStore protocol ──")

proto_dir = tempfile.mkdtemp()
pstore = MemdirStore(proto_dir)

# add via protocol
mem = Memory(
    id="proto-1",
    content="protocol-added memory content",
    metadata={
        "type": "user",
        "name": "proto memory",
        "description": "added via protocol",
        "filename": "proto_1.md",
    },
)
pstore.add(mem)
test("protocol add saves file", os.path.exists(os.path.join(proto_dir, "proto_1.md")))

# get_all via protocol
all_mems = pstore.get_all()
test("get_all returns Memory objects", len(all_mems) == 1)
test("get_all Memory has content", all_mems[0].content == "protocol-added memory content")
# id is derived from filename stem (proto_1.md → proto_1), not the original memory id
test("get_all Memory has id", all_mems[0].id == "proto_1")

# add second memory
mem2 = Memory(
    id="proto-2",
    content="second protocol memory",
    metadata={
        "type": "project",
        "name": "second",
        "description": "another memory",
        "filename": "proto_2.md",
    },
)
pstore.add(mem2)
test("get_all returns 2 after second add", len(pstore.get_all()) == 2)

# update via protocol — id used here is the filename stem (proto_1), not the original memory id
pstore.update("proto_1", "updated content for memory")
updated = [m for m in pstore.get_all() if m.id == "proto_1"]
test("update changes content", len(updated) == 1 and updated[0].content == "updated content for memory")

# update nonexistent is a noop — no crash
pstore.update("does-not-exist", "whatever content")
test("update nonexistent is noop", len(pstore.get_all()) == 2)

# auto-creates nested directory
nested_path = os.path.join(tempfile.mkdtemp(), "deep", "nested", "memory")
nested_store = MemdirStore(nested_path)
test("auto-creates nested dir", os.path.isdir(nested_path))

print(f"\n{'='*50}")
print(f"  memory system tests: {passed} passed, {failed} failed")
print(f"{'='*50}")
sys.exit(1 if failed else 0)
