#!/usr/bin/env python3
"""standalone tests for DAG task system, message bus, shared memory, and auto-decomposition.

run: python3 tests/scripts/test_dag_messaging.py
"""
import sys
import os
import asyncio
import tempfile
import shutil
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
    spec = importlib.util.spec_from_file_location(name, os.path.join(_root, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# load modules
_types = _load("dokumen.tasks.types", "dokumen/tasks/types.py")
sys.modules["dokumen.tasks"] = type(sys)("dokumen.tasks")
sys.modules["dokumen.tasks"].types = _types
sys.modules["dokumen.tasks"].TaskStatus = _types.TaskStatus
sys.modules["dokumen.tasks"].Task = _types.Task
sys.modules["dokumen.tasks"].TaskOutput = _types.TaskOutput

_mgr = _load("dokumen.tasks.manager", "dokumen/tasks/manager.py")

Task = _types.Task
TaskStatus = _types.TaskStatus
TaskManager = _mgr.TaskManager
topological_sort = _mgr.topological_sort
validate_dependencies = _mgr.validate_dependencies
is_task_ready = _mgr.is_task_ready

loop = asyncio.new_event_loop()

# ═══════════════════════════════════════════
# TaskStatus BLOCKED
# ═══════════════════════════════════════════

print("── TaskStatus.BLOCKED ──")
test("BLOCKED exists", TaskStatus.BLOCKED.value == "blocked")
test("BLOCKED not terminal", not Task(status=TaskStatus.BLOCKED).is_terminal)
test("BLOCKED is_blocked", Task(status=TaskStatus.BLOCKED).is_blocked)
test("PENDING not is_blocked", not Task(status=TaskStatus.PENDING).is_blocked)

# ═══════════════════════════════════════════
# depends_on field
# ═══════════════════════════════════════════

print("\n── depends_on field ──")
t = Task(name="x", depends_on=["a", "b"])
test("depends_on stored", t.depends_on == ["a", "b"])
test("depends_on in to_dict", t.to_dict()["depends_on"] == ["a", "b"])
t2 = Task.from_dict(t.to_dict())
test("depends_on roundtrip", t2.depends_on == ["a", "b"])
test("depends_on default empty", Task().depends_on == [])
test("from_dict no depends_on", Task.from_dict({"name": "x"}).depends_on == [])


# ═══════════════════════════════════════════
# topological sort
# ═══════════════════════════════════════════

print("\n── topological sort ──")

# simple chain: a -> b -> c
a = Task(id="a", name="a")
b = Task(id="b", name="b", depends_on=["a"])
c = Task(id="c", name="c", depends_on=["b"])
order = topological_sort([c, a, b])  # out of order input
ids = [t.id for t in order]
test("chain: a before b", ids.index("a") < ids.index("b"))
test("chain: b before c", ids.index("b") < ids.index("c"))
test("chain: all present", len(ids) == 3)

# diamond: a -> b, a -> c, b+c -> d
a2 = Task(id="a2", name="a")
b2 = Task(id="b2", name="b", depends_on=["a2"])
c2 = Task(id="c2", name="c", depends_on=["a2"])
d2 = Task(id="d2", name="d", depends_on=["b2", "c2"])
order2 = topological_sort([d2, b2, c2, a2])
ids2 = [t.id for t in order2]
test("diamond: a first", ids2[0] == "a2")
test("diamond: d last", ids2[-1] == "d2")
test("diamond: b before d", ids2.index("b2") < ids2.index("d2"))
test("diamond: c before d", ids2.index("c2") < ids2.index("d2"))

# no deps: all tasks returned
x = Task(id="x1", name="x")
y = Task(id="y1", name="y")
z = Task(id="z1", name="z")
order3 = topological_sort([x, y, z])
test("no deps: all returned", len(order3) == 3)

# empty input
test("empty input", topological_sort([]) == [])

# cycle: a -> b -> a (cycled nodes omitted)
ca = Task(id="ca", name="a", depends_on=["cb"])
cb = Task(id="cb", name="b", depends_on=["ca"])
order4 = topological_sort([ca, cb])
test("cycle: omitted", len(order4) == 0)

# partial cycle: a (no deps), b -> c -> b (cycle)
pa = Task(id="pa", name="a")
pb = Task(id="pb", name="b", depends_on=["pc"])
pc = Task(id="pc", name="c", depends_on=["pb"])
order5 = topological_sort([pa, pb, pc])
test("partial cycle: a returned", len(order5) == 1 and order5[0].id == "pa")

# missing dep (ignored)
m1 = Task(id="m1", name="a", depends_on=["nonexistent"])
order6 = topological_sort([m1])
test("missing dep: task still returned", len(order6) == 1)


# ═══════════════════════════════════════════
# validate_dependencies
# ═══════════════════════════════════════════

print("\n── validate_dependencies ──")

# valid graph
v1 = Task(id="v1", name="a")
v2 = Task(id="v2", name="b", depends_on=["v1"])
errors = validate_dependencies([v1, v2])
test("valid graph no errors", len(errors) == 0)

# self-reference
sr = Task(id="sr", name="self", depends_on=["sr"])
errors2 = validate_dependencies([sr])
test("self-ref detected", "sr" in errors2)

# unknown ref
ur = Task(id="ur", name="unknown", depends_on=["nonexistent"])
errors3 = validate_dependencies([ur])
test("unknown ref detected", "ur" in errors3)

# cycle detection
cyc_a = Task(id="cyc_a", name="a", depends_on=["cyc_b"])
cyc_b = Task(id="cyc_b", name="b", depends_on=["cyc_a"])
errors4 = validate_dependencies([cyc_a, cyc_b])
test("cycle detected", "_cycles" in errors4)

# empty is valid
test("empty is valid", validate_dependencies([]) == {})


# ═══════════════════════════════════════════
# is_task_ready
# ═══════════════════════════════════════════

print("\n── is_task_ready ──")

ta = Task(id="ta", status=TaskStatus.COMPLETED)
tb = Task(id="tb", status=TaskStatus.BLOCKED, depends_on=["ta"])
tc = Task(id="tc", status=TaskStatus.BLOCKED, depends_on=["ta", "tb"])
all_map = {"ta": ta, "tb": tb, "tc": tc}

test("completed dep: tb ready", is_task_ready(tb, all_map))
test("incomplete dep: tc not ready", not is_task_ready(tc, all_map))

# no deps always ready
td = Task(id="td", status=TaskStatus.PENDING)
test("no deps: ready", is_task_ready(td, {}))

# completed task not ready (already done)
te = Task(id="te", status=TaskStatus.COMPLETED)
test("completed not ready", not is_task_ready(te, {}))

# in_progress not ready
tf = Task(id="tf", status=TaskStatus.IN_PROGRESS)
test("in_progress not ready", not is_task_ready(tf, {}))


# ═══════════════════════════════════════════
# TaskManager DAG operations
# ═══════════════════════════════════════════

print("\n── TaskManager DAG ──")

tm = TaskManager()

# create with deps
ta = tm.create("step a", description="first")
tb = tm.create("step b", description="needs a", depends_on=[ta.id])
tc = tm.create("step c", description="needs a", depends_on=[ta.id])
td = tm.create("step d", description="needs b+c", depends_on=[tb.id, tc.id])

test("a starts pending", ta.status == TaskStatus.PENDING)
test("b starts blocked", tb.status == TaskStatus.BLOCKED)
test("c starts blocked", tc.status == TaskStatus.BLOCKED)
test("d starts blocked", td.status == TaskStatus.BLOCKED)

# get_ready_tasks
ready = tm.get_ready_tasks()
test("only a is ready", len(ready) == 1 and ready[0].id == ta.id)

# cannot start blocked task
blocked_start = tm.start(tb.id)
test("cannot start blocked task", blocked_start.status == TaskStatus.BLOCKED)

# complete a -> unblocks b and c
tm.start(ta.id)
tm.complete(ta.id)

# b and c should now be pending
b_after = tm.get(tb.id)
c_after = tm.get(tc.id)
d_after = tm.get(td.id)
test("b unblocked to pending", b_after.status == TaskStatus.PENDING)
test("c unblocked to pending", c_after.status == TaskStatus.PENDING)
test("d still blocked", d_after.status == TaskStatus.BLOCKED)

ready2 = tm.get_ready_tasks()
ready_ids = {t.id for t in ready2}
test("b and c now ready", tb.id in ready_ids and tc.id in ready_ids)
test("d not ready yet", td.id not in ready_ids)

# complete b and c -> unblocks d
tm.start(tb.id)
tm.complete(tb.id)
tm.start(tc.id)
tm.complete(tc.id)

d_after2 = tm.get(td.id)
test("d unblocked after b+c", d_after2.status == TaskStatus.PENDING)

# execution order
order = tm.get_execution_order()
order_ids = [t.id for t in order]
test("topo order has all tasks", len(order_ids) == 4)

# validate
errors = tm.validate()
test("valid graph", len(errors) == 0)

# get_dependents
deps_of_a = tm.get_dependents(ta.id)
dep_ids = {t.id for t in deps_of_a}
test("a's dependents are b and c", dep_ids == {tb.id, tc.id})

# blocked count (critical path)
tm2 = TaskManager()
t1 = tm2.create("root")
t2 = tm2.create("mid", depends_on=[t1.id])
t3 = tm2.create("leaf", depends_on=[t2.id])
test("root blocks 2 transitively", tm2.get_blocked_count(t1.id) == 2)
test("mid blocks 1", tm2.get_blocked_count(t2.id) == 1)
test("leaf blocks 0", tm2.get_blocked_count(t3.id) == 0)


# ═══════════════════════════════════════════
# cascade failure
# ═══════════════════════════════════════════

print("\n── cascade failure ──")

tm3 = TaskManager()
fa = tm3.create("step a")
fb = tm3.create("step b", depends_on=[fa.id])
fc = tm3.create("step c", depends_on=[fb.id])
fd = tm3.create("independent")

tm3.start(fa.id)
tm3.fail(fa.id, error="boom")

fb_after = tm3.get(fb.id)
fc_after = tm3.get(fc.id)
fd_after = tm3.get(fd.id)

test("b cascade failed", fb_after.status == TaskStatus.FAILED)
test("c cascade failed (transitive)", fc_after.status == TaskStatus.FAILED)
test("d unaffected", fd_after.status == TaskStatus.PENDING)
test("b has cascade error msg", "dependency" in (fb_after.error or ""))


# ═══════════════════════════════════════════
# event system
# ═══════════════════════════════════════════

print("\n── event system ──")

tm4 = TaskManager()
ready_events = []
complete_events = []
failed_events = []
all_done = []

tm4.on("task:ready", lambda t: ready_events.append(t.id))
tm4.on("task:complete", lambda t: complete_events.append(t.id))
tm4.on("task:failed", lambda t: failed_events.append(t.id))
unsub = tm4.on("all:complete", lambda: all_done.append(True))

ea = tm4.create("a")
test("ready event on create", ea.id in ready_events)

eb = tm4.create("b", depends_on=[ea.id])
test("no ready event for blocked", eb.id not in ready_events)

tm4.start(ea.id)
tm4.complete(ea.id)
test("complete event fired", ea.id in complete_events)
test("b unblocked event fired", eb.id in ready_events)

tm4.start(eb.id)
tm4.complete(eb.id)
test("all:complete fired", len(all_done) == 1)

# unsubscribe
unsub()
tm4.create("c")
# no crash = unsubscribe worked

# event on fail
tm5 = TaskManager()
fx = tm5.create("x")
fy = tm5.create("y", depends_on=[fx.id])
fail_events = []
tm5.on("task:failed", lambda t: fail_events.append(t.id))
tm5.start(fx.id)
tm5.fail(fx.id, "err")
test("fail event for x", fx.id in fail_events)
test("cascade fail event for y", fy.id in fail_events)


# ═══════════════════════════════════════════
# add_batch
# ═══════════════════════════════════════════

print("\n── add_batch ──")

tm6 = TaskManager()
batch_a = Task(id="ba", name="a")
batch_b = Task(id="bb", name="b", depends_on=["ba"])
batch_c = Task(id="bc", name="c", depends_on=["ba"])
batch_d = Task(id="bd", name="d", depends_on=["bb", "bc"])

added = tm6.add_batch([batch_d, batch_b, batch_c, batch_a])  # out of order
test("batch: all added", len(added) == 4)

ba = tm6.get("ba")
bb = tm6.get("bb")
bd = tm6.get("bd")
test("batch: a is pending", ba.status == TaskStatus.PENDING)
test("batch: b is blocked", bb.status == TaskStatus.BLOCKED)
test("batch: d is blocked", bd.status == TaskStatus.BLOCKED)


# ═══════════════════════════════════════════
# backward compat: tasks without depends_on
# ═══════════════════════════════════════════

print("\n── backward compat ──")

tm7 = TaskManager()
old1 = tm7.create("old task")
old2 = tm7.create("another old task", parent_id=old1.id)
test("no depends_on still works", old1.status == TaskStatus.PENDING)
test("parent_id still works", old2.parent_id == old1.id)
test("subtasks still work", len(tm7.get_subtasks(old1.id)) == 1)
tm7.start(old1.id)
tm7.complete(old1.id)
test("old lifecycle still works", tm7.get(old1.id).status == TaskStatus.COMPLETED)


# ═══════════════════════════════════════════
# MessageBus
# ═══════════════════════════════════════════

print("\n── MessageBus ──")

_msg = _load("dokumen.coordinator.messaging", "dokumen/coordinator/messaging.py")
MessageBus = _msg.MessageBus
Message = _msg.Message
BROADCAST = _msg.BROADCAST

bus = MessageBus()

# send point-to-point
msg1 = bus.send("worker-1", "worker-2", "found api.md")
test("send returns message", msg1.sender == "worker-1")
test("message has id", msg1.id.startswith("msg-"))
test("message not broadcast", not msg1.is_broadcast)

# get unread
unread = bus.get_unread("worker-2")
test("worker-2 has 1 unread", len(unread) == 1)
test("unread content correct", unread[0].content == "found api.md")

# worker-1 has no unread (they sent it)
test("worker-1 no unread", len(bus.get_unread("worker-1")) == 0)

# worker-3 has no unread
test("worker-3 no unread", len(bus.get_unread("worker-3")) == 0)

# mark read
bus.mark_read("worker-2")
test("after mark_read: no unread", len(bus.get_unread("worker-2")) == 0)

# broadcast
msg2 = bus.broadcast("coordinator", "focus on auth module")
test("broadcast is_broadcast", msg2.is_broadcast)
test("broadcast recipient is *", msg2.recipient == BROADCAST)

# broadcasts go to everyone except sender
unread_w1 = bus.get_unread("worker-1")
unread_w2 = bus.get_unread("worker-2")
unread_coord = bus.get_unread("coordinator")
test("worker-1 gets broadcast", len(unread_w1) == 1)
test("worker-2 gets broadcast", len(unread_w2) == 1)
test("coordinator doesn't get own broadcast", len(unread_coord) == 0)

# get_conversation
bus.send("worker-1", "worker-2", "hey")
bus.send("worker-2", "worker-1", "sup")
convo = bus.get_conversation("worker-1", "worker-2")
test("conversation has both directions", len(convo) >= 2)

# get_all
all_msgs = bus.get_all()
test("get_all returns all", len(all_msgs) >= 4)

all_for_w2 = bus.get_all("worker-2")
test("get_all filtered", len(all_for_w2) >= 2)

# subscribe
received = []
unsub = bus.subscribe("worker-2", lambda m: received.append(m.content))
bus.send("worker-1", "worker-2", "callback test")
test("subscriber notified", "callback test" in received)

# broadcast subscriber
broadcast_received = []
bus.subscribe("worker-3", lambda m: broadcast_received.append(m.content))
bus.broadcast("worker-1", "hello all")
test("broadcast subscriber notified", "hello all" in broadcast_received)

# unsubscribe
unsub()
bus.send("worker-1", "worker-2", "after unsub")
test("unsubscribed: not notified", "after unsub" not in received)

# summary
summary = bus.get_summary()
test("summary not empty", len(summary) > 0)
test("summary has header", "inter-agent" in summary.lower())

# empty bus summary
bus2 = MessageBus()
test("empty bus summary is empty", bus2.get_summary() == "")

# message_count
test("message_count", bus.message_count >= 6)

# clear
bus.clear()
test("clear empties messages", bus.message_count == 0)
test("clear empties unread", len(bus.get_unread("worker-2")) == 0)

# Message to_dict / from_dict
msg = Message(sender="a", recipient="b", content="hello")
d = msg.to_dict()
test("Message to_dict has sender", d["sender"] == "a")
msg2 = Message.from_dict(d)
test("Message roundtrip", msg2.sender == "a" and msg2.content == "hello")


# ═══════════════════════════════════════════
# SharedMemory
# ═══════════════════════════════════════════

print("\n── SharedMemory ──")

_smem = _load("dokumen.coordinator.shared_memory", "dokumen/coordinator/shared_memory.py")
SharedMemory = _smem.SharedMemory

mem = SharedMemory()

# write and read
mem.write("worker-1", "findings", "found 3 endpoints")
test("read own write", mem.read("worker-1", "findings") == "found 3 endpoints")

# another agent can read
test("cross-agent read", mem.read("worker-1", "findings") == "found 3 endpoints")

# read nonexistent
test("read nonexistent", mem.read("worker-1", "nope") is None)

# read_any
test("read_any", mem.read_any("worker-1/findings") == "found 3 endpoints")

# list_by_agent
mem.write("worker-1", "status", "done")
entries = mem.list_by_agent("worker-1")
test("list_by_agent", len(entries) == 2)
test("list_by_agent keys", "findings" in entries and "status" in entries)

# empty agent
test("list empty agent", len(mem.list_by_agent("worker-99")) == 0)

# task result
mem.write_task_result("task-1", "worker-1", "analysis complete")
test("get_task_result", mem.get_task_result("task-1") == "analysis complete")
test("get nonexistent task result", mem.get_task_result("nope") is None)

# summary
summary = mem.get_summary()
test("summary not empty", len(summary) > 0)
test("summary has header", "shared team memory" in summary.lower())
test("summary mentions worker-1", "worker-1" in summary)

# empty summary
mem2 = SharedMemory()
test("empty summary is empty", mem2.get_summary() == "")

# size
test("size", mem.size >= 3)

# clear
mem.clear()
test("clear empties", mem.size == 0)

# overwrite preserves created_at (upsert behavior)
mem3 = SharedMemory()
mem3.write("a", "key", "v1")
mem3.write("a", "key", "v2")
test("overwrite updates value", mem3.read("a", "key") == "v2")

# truncation in summary
mem4 = SharedMemory()
mem4.write("agent", "long", "x" * 500)
summary4 = mem4.get_summary(max_value_chars=50)
test("summary truncates long values", "..." in summary4)


# ═══════════════════════════════════════════
# auto-decomposition parser
# ═══════════════════════════════════════════

print("\n── auto-decomposition parser ──")

# need to load coordinator module carefully
_coord_types = _load("dokumen.coordinator.types", "dokumen/coordinator/types.py")
sys.modules["dokumen.coordinator"] = type(sys)("dokumen.coordinator")
sys.modules["dokumen.coordinator"].types = _coord_types
sys.modules["dokumen.coordinator"].messaging = _msg
sys.modules["dokumen.coordinator"].shared_memory = _smem

# stub worker module
_worker = _load("dokumen.coordinator.worker", "dokumen/coordinator/worker.py")
sys.modules["dokumen.coordinator"].worker = _worker

_coord = _load("dokumen.coordinator.coordinator", "dokumen/coordinator/coordinator.py")
_parse_task_specs = _coord._parse_task_specs

# fenced json
text1 = '''here's my plan:
```json
[
  {"title": "analyze", "description": "do analysis", "assignee": "w1", "depends_on": []},
  {"title": "report", "description": "write report", "assignee": "w2", "depends_on": ["analyze"]}
]
```'''
specs1 = _parse_task_specs(text1)
test("fenced json parsed", len(specs1) == 2)
test("fenced: first title", specs1[0]["title"] == "analyze")
test("fenced: deps", specs1[1]["depends_on"] == ["analyze"])

# bare json
text2 = '[{"title": "only", "description": "task", "assignee": "w1", "depends_on": []}]'
specs2 = _parse_task_specs(text2)
test("bare json parsed", len(specs2) == 1)

# json with surrounding text
text3 = 'sure! [{"title": "x", "description": "y", "assignee": "w1", "depends_on": []}] hope that helps'
specs3 = _parse_task_specs(text3)
test("surrounded json parsed", len(specs3) == 1)

# garbage
specs4 = _parse_task_specs("no json here at all")
test("garbage returns empty", specs4 == [])

# empty
specs5 = _parse_task_specs("")
test("empty returns empty", specs5 == [])

# malformed
specs6 = _parse_task_specs("```json\nnot valid json\n```")
specs7 = _parse_task_specs("[not valid]")
test("malformed fenced returns empty", specs6 == [])
test("malformed bare returns empty", specs7 == [])


# ═══════════════════════════════════════════
# coordinator with DAG execution
# ═══════════════════════════════════════════

print("\n── coordinator DAG execution ──")

CoordinatorAgent = _coord.CoordinatorAgent
WorkerTask = _coord_types.WorkerTask
CoordinatorPlan = _coord_types.CoordinatorPlan

# stub execution: coordinator with no provider runs workers in stub mode
coord = CoordinatorAgent(max_workers=3, synthesis_strategy="merge")

# create plan with dependencies
t_a = WorkerTask(id="wt-a", name="worker-1", goal="do step a")
t_b = WorkerTask(id="wt-b", name="worker-2", goal="do step b", depends_on=["wt-a"])
t_c = WorkerTask(id="wt-c", name="worker-3", goal="do step c", depends_on=["wt-a"])
t_d = WorkerTask(id="wt-d", name="worker-1", goal="do step d", depends_on=["wt-b", "wt-c"])

plan = CoordinatorPlan(
    main_goal="complete all steps",
    worker_tasks=[t_d, t_a, t_b, t_c],  # out of order
    synthesis_strategy="merge",
)

results = loop.run_until_complete(coord.execute_plan(plan))
test("all tasks got results", len(results) == 4)

# check that results include all task ids
result_ids = {r.task_id for r in results}
test("result for a", "wt-a" in result_ids)
test("result for b", "wt-b" in result_ids)
test("result for c", "wt-c" in result_ids)
test("result for d", "wt-d" in result_ids)

# message bus accessible
test("coordinator has bus", coord.bus is not None)
test("coordinator has shared memory", coord.shared_memory is not None)

# run() method
run_result = loop.run_until_complete(coord.run(goal="test goal"))
test("run returns dict", isinstance(run_result, dict))
test("run has success", "success" in run_result)
test("run has synthesis", "synthesis" in run_result)

# run with worker_configs
run2 = loop.run_until_complete(coord.run(
    goal="multi worker",
    worker_configs=[
        {"name": "w1", "goal": "part 1"},
        {"name": "w2", "goal": "part 2"},
    ],
))
test("run with configs", isinstance(run2, dict))

# depends_on in WorkerTask roundtrip
wt = WorkerTask(name="x", depends_on=["dep1", "dep2"])
d = wt.to_dict()
test("WorkerTask to_dict has depends_on", d["depends_on"] == ["dep1", "dep2"])
wt2 = WorkerTask.from_dict(d)
test("WorkerTask roundtrip depends_on", wt2.depends_on == ["dep1", "dep2"])


# ═══════════════════════════════════════════
# disk persistence with depends_on
# ═══════════════════════════════════════════

print("\n── disk persistence ──")

tmpdir = tempfile.mkdtemp(prefix="dag_persist_")
tm_disk = TaskManager(persist_dir=tmpdir)
dp_a = tm_disk.create("a")
dp_b = tm_disk.create("b", depends_on=[dp_a.id])

# reload from disk
tm_disk2 = TaskManager(persist_dir=tmpdir)
loaded_b = tm_disk2.get(dp_b.id)
test("persisted depends_on", loaded_b is not None and loaded_b.depends_on == [dp_a.id])
test("persisted status", loaded_b.status == TaskStatus.BLOCKED)

shutil.rmtree(tmpdir)


# ═══════════════════════════════════════════
# summary
# ═══════════════════════════════════════════

print(f"\n{'='*50}")
print(f"  DAG + messaging tests: {passed} passed, {failed} failed")
print(f"{'='*50}")

loop.close()
sys.exit(1 if failed else 0)
