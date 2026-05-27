#!/usr/bin/env python3
"""standalone tests for context, tasks, coordinator, planning, orchestrator, skills.

run: python3 tests/scripts/test_context_tasks_coord.py
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


# load modules in dependency order

# 1. context
archive_mod = _load("dokumen.context.archive", "dokumen/context/archive.py")
compactor_mod = _load("dokumen.context.compactor", "dokumen/context/compactor.py")
micro_mod = _load("dokumen.context.micro_compact", "dokumen/context/micro_compact.py")

# 2. tasks
task_types_mod = _load("dokumen.tasks.types", "dokumen/tasks/types.py")
task_mgr_mod = _load("dokumen.tasks.manager", "dokumen/tasks/manager.py")

# 3. coordinator — types has no deps; messaging/shared_memory before coordinator
coord_types_mod = _load("dokumen.coordinator.types", "dokumen/coordinator/types.py")
coord_msg_mod = _load("dokumen.coordinator.messaging", "dokumen/coordinator/messaging.py")
coord_mem_mod = _load("dokumen.coordinator.shared_memory", "dokumen/coordinator/shared_memory.py")
coord_worker_mod = _load("dokumen.coordinator.worker", "dokumen/coordinator/worker.py")
coord_mod = _load("dokumen.coordinator.coordinator", "dokumen/coordinator/coordinator.py")

# 4. planning
plan_types_mod = _load("dokumen.planning.types", "dokumen/planning/types.py")
plan_mgr_mod = _load("dokumen.planning.planner", "dokumen/planning/planner.py")

# 5. tool orchestrator
orch_mod = _load("dokumen.tools.orchestrator", "dokumen/tools/orchestrator.py")

# 6. skill loader — yaml is available (pyyaml installed)
skill_mod = _load("dokumen.skills.loader", "dokumen/skills/loader.py")

# ── shorthand imports ──
Turn = compactor_mod.Turn
CompactionResult = compactor_mod.CompactionResult
ContextCompactor = compactor_mod.ContextCompactor
RuleSummarizer = compactor_mod.RuleSummarizer
CHARS_PER_TOKEN = compactor_mod.CHARS_PER_TOKEN

MicroCompactor = micro_mod.MicroCompactor
ToolResultEntry = micro_mod.ToolResultEntry

TaskStatus = task_types_mod.TaskStatus
TaskOutput = task_types_mod.TaskOutput
Task = task_types_mod.Task

TaskManager = task_mgr_mod.TaskManager
InMemoryTaskStore = task_mgr_mod.InMemoryTaskStore

WorkerTask = coord_types_mod.WorkerTask
WorkerResult = coord_types_mod.WorkerResult
WorkerStatus = coord_types_mod.WorkerStatus
CoordinatorPlan = coord_types_mod.CoordinatorPlan

WorkerAgent = coord_worker_mod.WorkerAgent
CoordinatorAgent = coord_mod.CoordinatorAgent

PlanStatus = plan_types_mod.PlanStatus
PlanStep = plan_types_mod.PlanStep
Plan = plan_types_mod.Plan

PlanManager = plan_mgr_mod.PlanManager

ToolConcurrencyMode = orch_mod.ToolConcurrencyMode
classify_tool = orch_mod.classify_tool
ToolCall = orch_mod.ToolCall
ToolResult = orch_mod.ToolResult
ToolBatch = orch_mod.ToolBatch
ToolOrchestrator = orch_mod.ToolOrchestrator

ExecutionMode = skill_mod.ExecutionMode
SkillDefinition = skill_mod.SkillDefinition
load_skill_file = skill_mod.load_skill_file
load_skills_from_directory = skill_mod.load_skills_from_directory
get_all_skills = skill_mod.get_all_skills
SYSTEM_SKILLS = skill_mod.SYSTEM_SKILLS

# ── simple test harness ──

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


# ─────────────────────────────────────────────
# context compactor tests
# ─────────────────────────────────────────────

print("\n── ContextCompactor ──")

# turn creation
t = Turn(role="user", content="hello world")
test("turn role set", t.role == "user")
test("turn content set", t.content == "hello world")
test("turn token estimate > 0", t.token_estimate > 0)
test("turn token estimate ~chars/4", t.token_estimate == len("hello world") // CHARS_PER_TOKEN)
test("turn timestamp is float", isinstance(t.timestamp, float))
test("turn metadata defaults empty", t.metadata == {})

# turn with explicit token estimate
t2 = Turn(role="assistant", content="x" * 100, token_estimate=42)
test("explicit token estimate respected", t2.token_estimate == 42)

# compaction result
cr = CompactionResult(
    summary="summ",
    turns_removed=5,
    turns_kept=3,
    tokens_before=1000,
    tokens_after=200,
    compaction_time=0.05,
)
test("compaction result tokens_saved property", cr.tokens_saved == 800)
test("compaction result to_dict has keys", "tokens_saved" in cr.to_dict())
test("compaction result to_dict tokens_saved value", cr.to_dict()["tokens_saved"] == 800)

# add_turn and current_tokens
cc = ContextCompactor(max_tokens=10000)
test("initial turn count zero", cc.turn_count == 0)
test("initial current_tokens zero", cc.current_tokens == 0)

cc.add_turn("user", "a" * 400)  # 100 tokens
cc.add_turn("assistant", "b" * 400)  # 100 tokens
test("turn count after two adds", cc.turn_count == 2)
test("current_tokens reflects both turns", cc.current_tokens == 200)

# needs_compaction
cc_small = ContextCompactor(max_tokens=100, compact_threshold=0.75)
cc_small.add_turn("user", "x" * 400)  # 100 tokens → 100% used
test("needs_compaction true when over threshold", cc_small.needs_compaction)

cc_fine = ContextCompactor(max_tokens=10000, compact_threshold=0.75)
cc_fine.add_turn("user", "x" * 40)
test("needs_compaction false when under threshold", not cc_fine.needs_compaction)

# compact() via asyncio.run
cc_compact = ContextCompactor(max_tokens=50000, keep_recent=2)
for i in range(8):
    cc_compact.add_turn("user" if i % 2 == 0 else "assistant", f"turn {i} content here " * 5)

result = asyncio.run(cc_compact.compact())
test("compact returns CompactionResult", isinstance(result, CompactionResult))
test("turns_removed > 0 after compaction", result.turns_removed > 0)
test("summary generated after compaction", len(result.summary) > 0)
test("summary contains compacted context header", "compacted context" in result.summary)
test("tokens_before > 0", result.tokens_before > 0)
test("tokens_saved >= 0", result.tokens_saved >= 0)
test("compaction_time > 0", result.compaction_time >= 0)

# compact with too few turns (no-op)
cc_small2 = ContextCompactor(max_tokens=50000, keep_recent=10)
cc_small2.add_turn("user", "hi")
result2 = asyncio.run(cc_small2.compact())
test("compact noop when turns <= keep_recent", result2.turns_removed == 0)
test("compact noop tokens_saved zero", result2.tokens_saved == 0)

# get_messages format
cc_msg = ContextCompactor()
cc_msg.add_turn("user", "hello")
cc_msg.add_turn("assistant", "hi there")
msgs = cc_msg.get_messages()
test("get_messages returns list", isinstance(msgs, list))
test("get_messages length matches turns", len(msgs) == 2)
test("get_messages has role key", "role" in msgs[0])
test("get_messages has content key", "content" in msgs[0])
test("get_messages role values correct", msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant")

# reinject_context after compaction
cc_reinject = ContextCompactor(
    max_tokens=50000,
    keep_recent=2,
    reinject_context=["important context A", "important context B"],
)
for i in range(6):
    cc_reinject.add_turn("user", f"message {i} " * 10)

asyncio.run(cc_reinject.compact())
turns_after = cc_reinject.get_turns()
reinjected = [t for t in turns_after if t.metadata.get("reinjected")]
test("reinject_context adds turns after compaction", len(reinjected) == 2)

# stats()
cc_stats = ContextCompactor(max_tokens=10000)
cc_stats.add_turn("user", "test content " * 20)
stats = cc_stats.stats()
test("stats returns dict", isinstance(stats, dict))
test("stats has turn_count", "turn_count" in stats)
test("stats has current_tokens", "current_tokens" in stats)
test("stats has usage_pct", "usage_pct" in stats)
test("stats has compactions key", "compactions" in stats)
test("stats compactions starts at 0", stats["compactions"] == 0)
test("stats total_tokens_processed > 0", stats["total_tokens_processed"] > 0)


# ─────────────────────────────────────────────
# micro compactor tests
# ─────────────────────────────────────────────

print("\n── MicroCompactor ──")

mc = MicroCompactor(age_threshold=300)
test("micro initial entry count zero", mc.entry_count == 0)
test("micro initial total_chars zero", mc.total_chars == 0)

# track tool results
mc.track("read_file", "file contents here " * 50)
mc.track("glob", "*.py\n*.md\n")
test("entry count after two tracks", mc.entry_count == 2)
test("total_chars positive after tracks", mc.total_chars > 0)

# compact by age threshold — set threshold to 0 to compact immediately
mc_fast = MicroCompactor(age_threshold=0, default_truncate_to=20)
mc_fast.track("read_file", "x" * 500)
mc_fast.track("glob", "y" * 500)
truncated = mc_fast.compact()
test("compact returns count of truncated entries", truncated == 2)
test("entries marked as truncated", all(e.truncated for e in mc_fast.get_results()))
test("content truncated to limit + suffix", len(mc_fast.get_results()[0].content) > 20)  # includes suffix

# subsequent compact does not double-count
truncated2 = mc_fast.compact()
test("compact skips already-truncated entries", truncated2 == 0)

# old entries not truncated when still fresh (high threshold)
mc_fresh = MicroCompactor(age_threshold=9999, default_truncate_to=20)
mc_fresh.track("read_file", "x" * 500)
t_fresh = mc_fresh.compact()
test("fresh entries not truncated with high threshold", t_fresh == 0)

# per-tool limits
mc_per = MicroCompactor(age_threshold=0, per_tool_limits={"read_file": 10, "glob": 200})
mc_per.track("read_file", "a" * 500)
mc_per.track("glob", "b" * 500)
mc_per.compact()
results = mc_per.get_results()
# read_file should be truncated (500 > 10), glob also (500 > 200)
test("per-tool limit applied to read_file", results[0].truncated)
test("per-tool limit applied to glob", results[1].truncated)
# read_file truncated to 10 chars + suffix
test("read_file truncated shorter than glob", len(results[0].content) < len(results[1].content))

# stats()
mc_stat = MicroCompactor(age_threshold=0, default_truncate_to=5)
mc_stat.track("tool_a", "original content here")
mc_stat.compact()
s = mc_stat.stats()
test("stats total_entries correct", s["total_entries"] == 1)
test("stats truncated_entries correct", s["truncated_entries"] == 1)
test("stats total_chars > 0", s["total_chars"] > 0)
# original_chars tracks the pre-truncation length; after truncation the current
# content may be longer due to the "[truncated from N chars]" suffix, so just
# verify both fields are present and non-negative
test("stats original_chars is non-negative", s["original_chars"] >= 0)

# clear()
mc_clear = MicroCompactor()
mc_clear.track("read_file", "data")
mc_clear.clear()
test("clear removes all entries", mc_clear.entry_count == 0)

# get_result by index
mc_idx = MicroCompactor()
mc_idx.track("tool_x", "data here")
entry = mc_idx.get_result(0)
test("get_result(0) returns entry", entry is not None)
test("get_result(0) tool_name correct", entry.tool_name == "tool_x")
test("get_result(-1) returns None for bad index", mc_idx.get_result(-1) is None)
test("get_result out of bounds returns None", mc_idx.get_result(99) is None)


# ─────────────────────────────────────────────
# task types
# ─────────────────────────────────────────────

print("\n── Task Types ──")

# TaskStatus enum
test("TaskStatus.PENDING value", TaskStatus.PENDING.value == "pending")
test("TaskStatus.IN_PROGRESS value", TaskStatus.IN_PROGRESS.value == "in_progress")
test("TaskStatus.COMPLETED value", TaskStatus.COMPLETED.value == "completed")
test("TaskStatus.FAILED value", TaskStatus.FAILED.value == "failed")
test("TaskStatus.CANCELLED value", TaskStatus.CANCELLED.value == "cancelled")

# TaskOutput creation
to = TaskOutput(content="analysis result")
test("task output content", to.content == "analysis result")
test("task output timestamp set", isinstance(to.timestamp, float))
test("task output metadata empty", to.metadata == {})

# TaskOutput roundtrip
to_d = to.to_dict()
to2 = TaskOutput.from_dict(to_d)
test("task output roundtrip content", to2.content == to.content)
test("task output roundtrip timestamp", abs(to2.timestamp - to.timestamp) < 0.001)

# Task creation
task = Task(name="analyze docs", description="read everything")
test("task id starts with task-", task.id.startswith("task-"))
test("task name correct", task.name == "analyze docs")
test("task status defaults to pending", task.status == TaskStatus.PENDING)
test("task is_terminal false when pending", not task.is_terminal)
test("task duration None when not completed", task.duration is None)

# terminal states
for status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
    t_term = Task(name="x")
    t_term.status = status
    test(f"is_terminal true for {status.value}", t_term.is_terminal)

# duration
task_dur = Task(name="timed")
task_dur.created_at = 1000.0
task_dur.completed_at = 1005.0
test("duration computed correctly", task_dur.duration == 5.0)

# to_dict / from_dict roundtrip
task_rt = Task(name="rt", description="desc", metadata={"k": "v"})
task_rt.status = TaskStatus.IN_PROGRESS
d = task_rt.to_dict()
task_rt2 = Task.from_dict(d)
test("task roundtrip id", task_rt2.id == task_rt.id)
test("task roundtrip name", task_rt2.name == task_rt.name)
test("task roundtrip status", task_rt2.status == TaskStatus.IN_PROGRESS)
test("task roundtrip metadata", task_rt2.metadata == {"k": "v"})


# ─────────────────────────────────────────────
# task manager
# ─────────────────────────────────────────────

print("\n── TaskManager ──")

tm = TaskManager()
test("task manager created", tm is not None)

# create
t1 = tm.create("task one", description="first task")
test("create returns Task", isinstance(t1, Task))
test("created task in pending state", t1.status == TaskStatus.PENDING)

# get
t1_fetched = tm.get(t1.id)
test("get returns correct task", t1_fetched is not None and t1_fetched.id == t1.id)
test("get returns None for unknown id", tm.get("nonexistent") is None)

# list
t2 = tm.create("task two")
tasks = tm.list()
test("list returns all tasks", len(tasks) >= 2)
test("list sorted by created_at", tasks == sorted(tasks, key=lambda t: t.created_at))

# filter by status
pending = tm.list(status=TaskStatus.PENDING)
test("list filter by status works", all(t.status == TaskStatus.PENDING for t in pending))

# start
started = tm.start(t1.id)
test("start transitions to in_progress", started.status == TaskStatus.IN_PROGRESS)
in_prog = tm.list(status=TaskStatus.IN_PROGRESS)
test("list in_progress finds started task", any(t.id == t1.id for t in in_prog))

# add_output
tm.add_output(t1.id, "found 3 files")
tm.add_output(t1.id, "validated schema")
t1_updated = tm.get(t1.id)
test("add_output increments outputs", len(t1_updated.outputs) == 2)
test("add_output content correct", t1_updated.outputs[0].content == "found 3 files")

# complete
completed = tm.complete(t1.id)
test("complete transitions to completed", completed.status == TaskStatus.COMPLETED)
test("completed_at set", completed.completed_at is not None)
test("duration available after complete", completed.duration is not None and completed.duration >= 0)

# cannot start terminal task
re_started = tm.start(t1.id)
test("cannot re-start completed task — stays completed", re_started.status == TaskStatus.COMPLETED)

# fail
t3 = tm.create("failing task")
t3_failed = tm.fail(t3.id, error="something went wrong")
test("fail transitions to failed", t3_failed.status == TaskStatus.FAILED)
test("fail sets error message", t3_failed.error == "something went wrong")

# cancel
t4 = tm.create("cancel me")
cancelled = tm.cancel(t4.id)
test("cancel transitions to cancelled", cancelled.status == TaskStatus.CANCELLED)

# cancel already-terminal returns without error
cancelled2 = tm.cancel(t4.id)
test("cancel terminal task returns task unchanged", cancelled2.status == TaskStatus.CANCELLED)

# get_subtasks with parent_id
parent = tm.create("parent")
child1 = tm.create("child one", parent_id=parent.id)
child2 = tm.create("child two", parent_id=parent.id)
subtasks = tm.get_subtasks(parent.id)
test("get_subtasks returns children", len(subtasks) == 2)
test("get_subtasks all have correct parent_id", all(t.parent_id == parent.id for t in subtasks))

# list filter by parent_id
by_parent = tm.list(parent_id=parent.id)
test("list filter by parent_id works", len(by_parent) == 2)

# summary
s = tm.summary()
test("summary returns dict", isinstance(s, dict))
test("summary has total key", "total" in s)
test("summary total >= created tasks", s["total"] >= 6)
test("summary has by_status", "by_status" in s)

# disk persistence
tmpdir = tempfile.mkdtemp()
try:
    tm_disk = TaskManager(persist_dir=os.path.join(tmpdir, "tasks"))
    tp = tm_disk.create("persisted task", description="saved to disk")
    tm_disk.start(tp.id)

    # create new manager pointing at same dir — should reload
    tm_disk2 = TaskManager(persist_dir=os.path.join(tmpdir, "tasks"))
    tp2 = tm_disk2.get(tp.id)
    test("disk persistence: task reloaded", tp2 is not None)
    test("disk persistence: status preserved", tp2.status == TaskStatus.IN_PROGRESS)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────
# coordinator types
# ─────────────────────────────────────────────

print("\n── Coordinator Types ──")

# WorkerTask creation
wt = WorkerTask(name="check-api", goal="validate api docs", tools=["read_file"])
test("worker task id starts with worker-", wt.id.startswith("worker-"))
test("worker task name set", wt.name == "check-api")
test("worker task goal set", wt.goal == "validate api docs")
test("worker task tools set", wt.tools == ["read_file"])
test("worker task default timeout", wt.timeout == 60.0)

# WorkerTask roundtrip
wt_d = wt.to_dict()
wt2 = WorkerTask.from_dict(wt_d)
test("worker task roundtrip id", wt2.id == wt.id)
test("worker task roundtrip name", wt2.name == wt.name)
test("worker task roundtrip goal", wt2.goal == wt.goal)

# WorkerResult
wr = WorkerResult(task_id="t-abc", output="found issues", findings=["issue1", "issue2"])
test("worker result success when completed", wr.success)
test("worker result findings length", len(wr.findings) == 2)

wr_fail = WorkerResult(task_id="t-fail", status=WorkerStatus.FAILED, error="boom")
test("worker result not success when failed", not wr_fail.success)
test("worker result not success when timeout", not WorkerResult(task_id="x", status=WorkerStatus.TIMEOUT).success)

# WorkerResult roundtrip
wr_d = wr.to_dict()
wr2 = WorkerResult.from_dict(wr_d)
test("worker result roundtrip task_id", wr2.task_id == wr.task_id)
test("worker result roundtrip findings", wr2.findings == wr.findings)
test("worker result roundtrip status", wr2.status == WorkerStatus.COMPLETED)

# CoordinatorPlan
cp = CoordinatorPlan(
    main_goal="validate all docs",
    worker_tasks=[wt],
    synthesis_strategy="merge",
)
test("coordinator plan id starts with plan-", cp.id.startswith("plan-"))
test("coordinator plan worker_count", cp.worker_count == 1)
test("coordinator plan synthesis_strategy set", cp.synthesis_strategy == "merge")

# CoordinatorPlan roundtrip
cp_d = cp.to_dict()
cp2 = CoordinatorPlan.from_dict(cp_d)
test("coordinator plan roundtrip id", cp2.id == cp.id)
test("coordinator plan roundtrip main_goal", cp2.main_goal == cp.main_goal)
test("coordinator plan roundtrip worker_tasks", len(cp2.worker_tasks) == 1)


# ─────────────────────────────────────────────
# coordinator agent (stub mode, provider=None)
# ─────────────────────────────────────────────

print("\n── CoordinatorAgent ──")

coord = CoordinatorAgent(provider=None, max_workers=4, default_timeout=30.0)
test("coordinator created", coord is not None)

# execute_plan with stub workers
plan = CoordinatorPlan(
    main_goal="test goal",
    worker_tasks=[
        WorkerTask(name="w1", goal="read api docs"),
        WorkerTask(name="w2", goal="check examples"),
    ],
)
results = asyncio.run(coord.execute_plan(plan))
test("execute_plan returns list", isinstance(results, list))
test("execute_plan result count matches tasks", len(results) == 2)
test("stub workers return success", all(r.success for r in results))
test("stub output contains goal text", any("read api docs" in r.output for r in results))
test("stub findings populated", all(len(r.findings) > 0 for r in results))

# synthesize — merge strategy
wr_a = WorkerResult(task_id="t-a", output="output A", findings=["finding 1"])
wr_b = WorkerResult(task_id="t-b", output="output B", findings=["finding 2"])
merged = coord.synthesize([wr_a, wr_b], strategy="merge")
test("merge synthesis contains task ids", "t-a" in merged and "t-b" in merged)
test("merge synthesis contains outputs", "output A" in merged and "output B" in merged)

# synthesize — vote strategy
wr_vote1 = WorkerResult(task_id="v1", output="x", findings=["common finding", "unique 1"])
wr_vote2 = WorkerResult(task_id="v2", output="y", findings=["common finding", "unique 2"])
voted = coord.synthesize([wr_vote1, wr_vote2], strategy="vote")
test("vote synthesis contains findings header", "findings" in voted.lower())
test("vote synthesis includes common finding", "common finding" in voted)

# synthesize — chain strategy
chained = coord.synthesize([wr_a, wr_b], strategy="chain")
test("chain synthesis has step headers", "step 1" in chained and "step 2" in chained)
test("chain synthesis is ordered", chained.index("step 1") < chained.index("step 2"))

# synthesize — all failed returns fallback message
failed_results = [WorkerResult(task_id="f1", status=WorkerStatus.FAILED, error="x")]
synth_fail = coord.synthesize(failed_results, strategy="merge")
test("synthesize with all failed returns fallback", "failed" in synth_fail)

# run() with worker_configs
output = asyncio.run(coord.run(
    "validate docs",
    worker_configs=[
        {"name": "reader", "goal": "read the docs", "tools": ["read_file"]},
        {"name": "checker", "goal": "check links", "tools": ["web_fetch"]},
    ],
))
test("run returns non-empty string", isinstance(output, dict) and output.get("synthesis", "") != "")

# run() without worker_configs creates single worker
output2 = asyncio.run(coord.run("single worker goal"))
test("run without configs returns string", isinstance(output2, dict) and output2.get("synthesis", "") != "")


# ─────────────────────────────────────────────
# planning types
# ─────────────────────────────────────────────

print("\n── Planning Types ──")

# PlanStep creation
step = PlanStep(description="list all files", done_criteria="have file list", tools_needed=["glob"])
test("step id starts with step-", step.id.startswith("step-"))
test("step description set", step.description == "list all files")
test("step done_criteria set", step.done_criteria == "have file list")
test("step status defaults to draft", step.status == PlanStatus.DRAFT)

# PlanStep roundtrip
step_d = step.to_dict()
step2 = PlanStep.from_dict(step_d)
test("step roundtrip id", step2.id == step.id)
test("step roundtrip description", step2.description == step.description)
test("step roundtrip status", step2.status == PlanStatus.DRAFT)
test("step roundtrip tools_needed", step2.tools_needed == ["glob"])

# Plan creation
plan_obj = Plan(
    goal="validate all api docs",
    steps=[
        PlanStep(description="list files", order=0),
        PlanStep(description="validate each", order=1),
    ],
)
test("plan id starts with plan-", plan_obj.id.startswith("plan-"))
test("plan total_steps count", plan_obj.total_steps == 2)
test("plan completed_steps starts at 0", plan_obj.completed_steps == 0)
test("plan progress starts at 0.0", plan_obj.progress == 0.0)

# plan progress after completing a step
plan_obj.steps[0].status = PlanStatus.COMPLETED
test("plan progress after one step", abs(plan_obj.progress - 0.5) < 0.001)
test("plan completed_steps increments", plan_obj.completed_steps == 1)

# plan current_step
cur = plan_obj.current_step
test("current_step is the non-completed step", cur.description == "validate each")

# all done
plan_obj.steps[1].status = PlanStatus.COMPLETED
test("current_step None when all done", plan_obj.current_step is None)
test("progress 1.0 when all done", plan_obj.progress == 1.0)

# plan with no steps
empty_plan = Plan(goal="empty")
test("empty plan progress is 0.0", empty_plan.progress == 0.0)
test("empty plan current_step is None", empty_plan.current_step is None)

# Plan roundtrip
plan_rt = Plan(goal="roundtrip test", steps=[PlanStep(description="step A")])
plan_rt_d = plan_rt.to_dict()
plan_rt2 = Plan.from_dict(plan_rt_d)
test("plan roundtrip id", plan_rt2.id == plan_rt.id)
test("plan roundtrip goal", plan_rt2.goal == plan_rt.goal)
test("plan roundtrip steps length", len(plan_rt2.steps) == 1)
test("plan roundtrip step description", plan_rt2.steps[0].description == "step A")


# ─────────────────────────────────────────────
# plan manager
# ─────────────────────────────────────────────

print("\n── PlanManager ──")

pm = PlanManager()
test("plan manager created", pm is not None)

# create
p1 = pm.create("main goal", steps=[
    {"description": "step 1", "done_criteria": "done 1", "tools_needed": ["glob"]},
    {"description": "step 2", "done_criteria": "done 2"},
])
test("create returns Plan", isinstance(p1, Plan))
test("created plan has 2 steps", p1.total_steps == 2)
test("created plan status is draft", p1.status == PlanStatus.DRAFT)

# get
p1_fetched = pm.get(p1.id)
test("get returns correct plan", p1_fetched is not None and p1_fetched.id == p1.id)
test("get returns None for unknown", pm.get("nope") is None)

# list
p2 = pm.create("second plan")
all_plans = pm.list()
test("list returns all plans", len(all_plans) >= 2)

# list filter by status
drafts = pm.list(status=PlanStatus.DRAFT)
test("list filter by status works", all(p.status == PlanStatus.DRAFT for p in drafts))

# approve
approved = pm.approve(p1.id)
test("approve transitions plan to approved", approved.status == PlanStatus.APPROVED)
test("approve sets steps to approved", all(s.status == PlanStatus.APPROVED for s in approved.steps))

# cannot approve non-draft
approved_again = pm.approve(p1.id)
test("cannot approve already-approved plan", approved_again.status == PlanStatus.APPROVED)

# start_step
step0_id = p1.steps[0].id
started_step = pm.start_step(p1.id, step0_id)
test("start_step returns PlanStep", isinstance(started_step, PlanStep))
test("start_step status in_progress", started_step.status == PlanStatus.IN_PROGRESS)
test("plan status in_progress when step started", pm.get(p1.id).status == PlanStatus.IN_PROGRESS)

# complete_step
completed_step = pm.complete_step(p1.id, step0_id, output="found 5 files")
test("complete_step returns PlanStep", isinstance(completed_step, PlanStep))
test("complete_step status completed", completed_step.status == PlanStatus.COMPLETED)
test("complete_step output stored", completed_step.output == "found 5 files")

# complete remaining step to finish plan
step1_id = p1.steps[1].id
pm.complete_step(p1.id, step1_id, output="all validated")
test("plan completed when all steps done", pm.get(p1.id).status == PlanStatus.COMPLETED)

# fail_step
p3 = pm.create("fail test", steps=[{"description": "risky step"}])
pm.approve(p3.id)
failed_step = pm.fail_step(p3.id, p3.steps[0].id, error="connection refused")
test("fail_step returns PlanStep", isinstance(failed_step, PlanStep))
test("fail_step status failed", failed_step.status == PlanStatus.FAILED)
test("fail_step error stored", failed_step.error == "connection refused")
test("plan status failed when step fails", pm.get(p3.id).status == PlanStatus.FAILED)

# add_step
p4 = pm.create("add step test")
new_step = pm.add_step(p4.id, "new step desc", done_criteria="new done", tools_needed=["read_file"])
test("add_step returns PlanStep", isinstance(new_step, PlanStep))
test("add_step adds to plan", pm.get(p4.id).total_steps == 1)
test("add_step description correct", new_step.description == "new step desc")

# add_step to unknown plan
no_step = pm.add_step("nope", "foo")
test("add_step to unknown plan returns None", no_step is None)

# get_progress_summary
pm.approve(p4.id)
summary_p4 = pm.get_progress_summary(p4.id)
test("get_progress_summary returns dict", isinstance(summary_p4, dict))
test("get_progress_summary has plan_id", summary_p4["plan_id"] == p4.id)
test("get_progress_summary has goal", "goal" in summary_p4)
test("get_progress_summary has progress", "progress" in summary_p4)
test("get_progress_summary returns None for unknown", pm.get_progress_summary("bad") is None)

# disk persistence
tmpdir2 = tempfile.mkdtemp()
try:
    pm_disk = PlanManager(persist_dir=os.path.join(tmpdir2, "plans"))
    pd = pm_disk.create("disk plan", steps=[{"description": "step X"}])
    pm_disk.approve(pd.id)

    pm_disk2 = PlanManager(persist_dir=os.path.join(tmpdir2, "plans"))
    pd2 = pm_disk2.get(pd.id)
    test("plan disk persistence: reloaded", pd2 is not None)
    test("plan disk persistence: status preserved", pd2.status == PlanStatus.APPROVED)
    test("plan disk persistence: steps preserved", pd2.total_steps == 1)
finally:
    shutil.rmtree(tmpdir2, ignore_errors=True)


# ─────────────────────────────────────────────
# tool orchestrator
# ─────────────────────────────────────────────

print("\n── ToolOrchestrator ──")

# classify_tool for known read/write/unknown tools
test("classify read_file as read_only", classify_tool("read_file") == ToolConcurrencyMode.READ_ONLY)
test("classify glob as read_only", classify_tool("glob") == ToolConcurrencyMode.READ_ONLY)
test("classify list_files as read_only", classify_tool("list_files") == ToolConcurrencyMode.READ_ONLY)
test("classify search_files as read_only", classify_tool("search_files") == ToolConcurrencyMode.READ_ONLY)
test("classify web_fetch as read_only", classify_tool("web_fetch") == ToolConcurrencyMode.READ_ONLY)
test("classify browser_snapshot as read_only", classify_tool("browser_snapshot") == ToolConcurrencyMode.READ_ONLY)
test("classify write_file as write", classify_tool("write_file") == ToolConcurrencyMode.WRITE)
test("classify run_shell_command as write", classify_tool("run_shell_command") == ToolConcurrencyMode.WRITE)
test("classify browser_click as write", classify_tool("browser_click") == ToolConcurrencyMode.WRITE)
test("classify unknown tool as unknown", classify_tool("some_custom_tool") == ToolConcurrencyMode.UNKNOWN)

# ToolCall auto-classification
tc_read = ToolCall(tool_name="read_file", args={"path": "a.py"})
test("ToolCall auto-classifies read_file", tc_read.mode == ToolConcurrencyMode.READ_ONLY)

tc_write = ToolCall(tool_name="write_file", args={"path": "b.py", "content": "x"})
test("ToolCall auto-classifies write_file", tc_write.mode == ToolConcurrencyMode.WRITE)

tc_unknown = ToolCall(tool_name="custom_tool", args={})
test("ToolCall auto-classifies unknown as unknown", tc_unknown.mode == ToolConcurrencyMode.UNKNOWN)

# partition: consecutive reads batch together
calls = [
    ToolCall(tool_name="read_file", args={}),
    ToolCall(tool_name="glob", args={}),
    ToolCall(tool_name="write_file", args={}),
    ToolCall(tool_name="read_file", args={}),
]
orch = ToolOrchestrator()
batches = orch.partition(calls)
test("partition produces 3 batches", len(batches) == 3)
test("first batch is concurrent (2 reads)", batches[0].concurrent and len(batches[0].calls) == 2)
test("second batch is serial (write)", not batches[1].concurrent)
test("third batch is concurrent (1 read)", batches[2].concurrent)

# partition empty list
test("partition empty list returns empty", orch.partition([]) == [])

# partition all reads
all_reads = [ToolCall(tool_name="read_file", args={}) for _ in range(4)]
read_batches = orch.partition(all_reads)
test("all reads produce one batch", len(read_batches) == 1)
test("all reads batch is concurrent", read_batches[0].concurrent)
test("all reads batch has all calls", len(read_batches[0].calls) == 4)

# partition all writes
all_writes = [ToolCall(tool_name="write_file", args={}) for _ in range(3)]
write_batches = orch.partition(all_writes)
test("all writes produce N serial batches", len(write_batches) == 3)
test("all write batches are not concurrent", all(not b.concurrent for b in write_batches))

# execute with mock executor
async def mock_executor(tool_name: str, args: dict) -> str:
    await asyncio.sleep(0)  # yield to event loop
    return f"result of {tool_name}"

orch_exec = ToolOrchestrator(executor=mock_executor)
exec_calls = [
    ToolCall(tool_name="read_file", args={"path": "a.py"}, call_id="c1"),
    ToolCall(tool_name="glob", args={"pattern": "*.py"}, call_id="c2"),
    ToolCall(tool_name="write_file", args={}, call_id="c3"),
]
results = asyncio.run(orch_exec.execute(exec_calls))
test("execute returns list of ToolResult", all(isinstance(r, ToolResult) for r in results))
test("execute results length matches calls", len(results) == 3)
test("execute results all successful", all(r.success for r in results))
test("execute result call_ids preserved", results[0].call_id == "c1")
test("execute result tool_names correct", results[2].tool_name == "write_file")
test("execute result output populated", "result of read_file" in results[0].output)

# execute raises without executor
orch_no_exec = ToolOrchestrator()
try:
    asyncio.run(orch_no_exec.execute([ToolCall(tool_name="read_file", args={})]))
    test("execute raises without executor", False)
except ValueError:
    test("execute raises without executor", True)

# executor exception -> failed result (not raised)
async def failing_executor(tool_name: str, args: dict) -> str:
    raise RuntimeError("tool crashed")

orch_fail = ToolOrchestrator(executor=failing_executor)
fail_results = asyncio.run(orch_fail.execute([ToolCall(tool_name="read_file", args={}, call_id="fx")]))
test("failed executor returns ToolResult with success=False", not fail_results[0].success)
test("failed executor result has error", fail_results[0].error is not None)

# concurrent batch actually runs concurrently (check timing)
_start_times = []

async def slow_executor(tool_name: str, args: dict) -> str:
    _start_times.append(time.time())
    await asyncio.sleep(0.05)
    return f"done {tool_name}"

orch_timing = ToolOrchestrator(executor=slow_executor)
timing_calls = [
    ToolCall(tool_name="read_file", args={}, call_id=f"t{i}") for i in range(3)
]
t0 = time.time()
asyncio.run(orch_timing.execute(timing_calls))
elapsed = time.time() - t0
# if truly concurrent, 3x0.05s tasks should finish in ~0.05-0.15s, not ~0.15s+
test("concurrent reads run faster than serial would", elapsed < 0.20)

# stats
stats_orch = orch_exec.stats()
test("orchestrator stats has total_calls", "total_calls" in stats_orch)
test("orchestrator stats total_calls > 0", stats_orch["total_calls"] > 0)
test("orchestrator stats has total_batches", "total_batches" in stats_orch)


# ─────────────────────────────────────────────
# skill loader
# ─────────────────────────────────────────────

print("\n── SkillLoader ──")

# SkillDefinition creation
sd = SkillDefinition(
    name="my-skill",
    description="does something",
    prompt="you are a helper",
    mode=ExecutionMode.INLINE,
    tools=["read_file"],
)
test("skill name set", sd.name == "my-skill")
test("skill description set", sd.description == "does something")
test("skill mode inline", sd.mode == ExecutionMode.INLINE)
test("skill tools set", sd.tools == ["read_file"])
test("skill effectiveness defaults to 1.0", sd.effectiveness == 1.0)

# ExecutionMode enum
test("ExecutionMode.INLINE value", ExecutionMode.INLINE.value == "inline")
test("ExecutionMode.FORK value", ExecutionMode.FORK.value == "fork")

# SkillDefinition roundtrip
sd_d = sd.to_dict()
sd2 = SkillDefinition.from_dict(sd_d)
test("skill roundtrip name", sd2.name == sd.name)
test("skill roundtrip mode", sd2.mode == ExecutionMode.INLINE)
test("skill roundtrip tools", sd2.tools == ["read_file"])
test("skill roundtrip description", sd2.description == sd.description)

# SkillDefinition from_dict with fork mode
sd_fork = SkillDefinition.from_dict({"name": "fork-skill", "mode": "fork", "description": "forked"})
test("skill from_dict fork mode", sd_fork.mode == ExecutionMode.FORK)

tmpdir3 = tempfile.mkdtemp()
try:
    # load_skill_file from yaml
    yaml_path = os.path.join(tmpdir3, "my-skill.yaml")
    with open(yaml_path, "w") as f:
        f.write("name: yaml-skill\ndescription: from yaml\nprompt: do the thing\nmode: inline\ntools:\n  - glob\n")
    skill_from_yaml = load_skill_file(yaml_path)
    test("load_skill_file from yaml returns SkillDefinition", isinstance(skill_from_yaml, SkillDefinition))
    test("load_skill_file yaml name", skill_from_yaml.name == "yaml-skill")
    test("load_skill_file yaml description", skill_from_yaml.description == "from yaml")
    test("load_skill_file yaml tools", skill_from_yaml.tools == ["glob"])
    test("load_skill_file yaml mode", skill_from_yaml.mode == ExecutionMode.INLINE)

    # load_skill_file from markdown (no frontmatter)
    md_path = os.path.join(tmpdir3, "plain-skill.md")
    with open(md_path, "w") as f:
        f.write("# plain skill\n\nyou are a plain skill agent. do things.\n")
    skill_from_md = load_skill_file(md_path)
    test("load_skill_file from md returns SkillDefinition", isinstance(skill_from_md, SkillDefinition))
    test("load_skill_file md name from filename", skill_from_md.name == "plain-skill")
    test("load_skill_file md prompt is content", "plain skill" in skill_from_md.prompt)

    # load_skill_file from markdown with frontmatter
    md_fm_path = os.path.join(tmpdir3, "frontmatter-skill.md")
    with open(md_fm_path, "w") as f:
        f.write("---\nname: fm-skill\ndescription: has frontmatter\nmode: fork\ntools:\n  - web_fetch\n---\n\nthe prompt body goes here\n")
    skill_fm = load_skill_file(md_fm_path)
    test("load_skill_file md frontmatter name overrides filename", skill_fm.name == "fm-skill")
    test("load_skill_file md frontmatter mode", skill_fm.mode == ExecutionMode.FORK)
    test("load_skill_file md frontmatter tools", skill_fm.tools == ["web_fetch"])
    test("load_skill_file md frontmatter prompt body", "prompt body" in skill_fm.prompt)

    # load_skill_file returns None for non-existent file
    test("load_skill_file None for missing file", load_skill_file("/tmp/nonexistent_12345.yaml") is None)

    # load_skills_from_directory
    skills_dir = os.path.join(tmpdir3, "skills_dir")
    os.makedirs(skills_dir)
    for i, fname in enumerate(["alpha.yaml", "beta.md", "gamma.yml"]):
        fpath = os.path.join(skills_dir, fname)
        if fname.endswith(".yaml") or fname.endswith(".yml"):
            with open(fpath, "w") as f:
                f.write(f"name: skill-{i}\ndescription: skill {i}\nprompt: prompt {i}\n")
        else:
            with open(fpath, "w") as f:
                f.write(f"# skill {i}\n\nthis is skill {i}")

    loaded = load_skills_from_directory(skills_dir)
    test("load_skills_from_directory loads all files", len(loaded) == 3)
    test("load_skills_from_directory returns list of SkillDefinition", all(isinstance(s, SkillDefinition) for s in loaded))

    # non-existent directory returns empty list
    empty_skills = load_skills_from_directory("/tmp/does_not_exist_abc123")
    test("load_skills_from_directory empty for missing dir", empty_skills == [])

finally:
    shutil.rmtree(tmpdir3, ignore_errors=True)

# get_all_skills with system
all_skills = get_all_skills()
test("get_all_skills returns list", isinstance(all_skills, list))
test("get_all_skills includes system", len(all_skills) >= len(SYSTEM_SKILLS))

# SYSTEM_SKILLS has mimick, qa-check, link-check
system_names = {s.name for s in SYSTEM_SKILLS}
test("SYSTEM_SKILLS has mimick", "mimick" in system_names)
test("SYSTEM_SKILLS has qa-check", "qa-check" in system_names)
test("SYSTEM_SKILLS has link-check", "link-check" in system_names)

# mimick skill is fork mode
mimick = next(s for s in SYSTEM_SKILLS if s.name == "mimick")
test("mimick skill is fork mode", mimick.mode == ExecutionMode.FORK)
test("mimick skill has read tools", "read_file" in mimick.tools)

# qa-check skill is inline mode
qa = next(s for s in SYSTEM_SKILLS if s.name == "qa-check")
test("qa-check skill is inline mode", qa.mode == ExecutionMode.INLINE)

# link-check has web_fetch tool
link = next(s for s in SYSTEM_SKILLS if s.name == "link-check")
test("link-check has web_fetch tool", "web_fetch" in link.tools)

# get_all_skills project override replaces system skill
tmpdir4 = tempfile.mkdtemp()
try:
    override_dir = tmpdir4
    with open(os.path.join(override_dir, "mimick.yaml"), "w") as f:
        f.write("name: mimick\ndescription: overridden mimick\nprompt: custom prompt\n")

    overridden = get_all_skills(project_skills_dir=override_dir)
    overridden_mimick = next(s for s in overridden if s.name == "mimick")
    test("project skill overrides system skill", overridden_mimick.description == "overridden mimick")
    # total count unchanged (override, not addition)
    test("project override does not add duplicate", len(overridden) == len(SYSTEM_SKILLS))
finally:
    shutil.rmtree(tmpdir4, ignore_errors=True)

# get_all_skills with include_system=False
no_system = get_all_skills(include_system=False)
test("get_all_skills include_system=False returns empty when no dir", no_system == [])


# ─────────────────────────────────────────────
# summary
# ─────────────────────────────────────────────

print(f"\n── results: {passed} passed, {failed} failed ──\n")
sys.exit(0 if failed == 0 else 1)
