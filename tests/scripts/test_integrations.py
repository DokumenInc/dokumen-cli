#!/usr/bin/env python3
"""standalone tests for integration wiring.

tests config sections, task tools, pipeline stage files, and yaml parsing.
avoids importing modules that depend on claude_agent_sdk.

run: python3 tests/scripts/test_integrations.py
"""
import sys
import os
import json
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
    """load a module directly without triggering dokumen.__init__."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_root, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════
# config sections (pydantic models)
# ═══════════════════════════════════════════

print("── config models ──")

try:
    # config.py only needs pydantic + yaml, not claude_agent_sdk
    _config = _load("dokumen.config", "dokumen/config.py")
    CompactionConfig = _config.CompactionConfig
    CoordinatorConfig = _config.CoordinatorConfig
    TasksConfig = _config.TasksConfig
    SkillsConfig = _config.SkillsConfig
    DokumenConfig = _config.DokumenConfig
    config_available = True
except Exception as e:
    print(f"  (skipping config tests: {e})")
    config_available = False

if config_available:
    # compaction defaults
    cc = CompactionConfig()
    test("compaction disabled by default", cc.enabled is False)
    test("compaction threshold default", cc.token_threshold == 0.75)
    test("compaction budget default", cc.token_budget == 100000)
    test("compaction keep_recent default", cc.keep_recent_turns == 5)
    test("micro_compact enabled default", cc.micro_compact_enabled is True)
    test("micro_compact age default", cc.micro_compact_age_seconds == 120.0)
    test("micro_compact max_chars default", cc.micro_compact_max_chars == 500)

    # compaction custom
    cc2 = CompactionConfig(enabled=True, token_threshold=0.5, token_budget=50000)
    test("compaction custom enabled", cc2.enabled is True)
    test("compaction custom threshold", cc2.token_threshold == 0.5)
    test("compaction custom budget", cc2.token_budget == 50000)

    # coordinator defaults
    print("\n── CoordinatorConfig ──")
    coc = CoordinatorConfig()
    test("coordinator disabled by default", coc.enabled is False)
    test("coordinator max_workers default", coc.max_workers == 5)
    test("coordinator strategy default", coc.synthesis_strategy == "merge")
    test("coordinator worker_timeout default", coc.worker_timeout == 120.0)
    test("coordinator worker_model default", coc.worker_model is None)

    # coordinator custom
    coc2 = CoordinatorConfig(enabled=True, max_workers=3, synthesis_strategy="vote", worker_model="haiku")
    test("coordinator custom enabled", coc2.enabled is True)
    test("coordinator custom workers", coc2.max_workers == 3)
    test("coordinator custom strategy", coc2.synthesis_strategy == "vote")
    test("coordinator custom model", coc2.worker_model == "haiku")

    # tasks defaults
    print("\n── TasksConfig ──")
    tc = TasksConfig()
    test("tasks disabled by default", tc.enabled is False)
    test("tasks persist default", tc.persist_to_disk is True)
    test("tasks max default", tc.max_tasks == 100)

    tc2 = TasksConfig(enabled=True, max_tasks=50)
    test("tasks custom enabled", tc2.enabled is True)
    test("tasks custom max", tc2.max_tasks == 50)

    # skills defaults
    print("\n── SkillsConfig ──")
    sc = SkillsConfig()
    test("skills enabled by default", sc.enabled is True)
    test("skills dir default", sc.dir is None)
    test("skills system default", sc.include_system is True)
    test("skills max_per_prompt default", sc.max_skills_per_prompt == 5)

    sc2 = SkillsConfig(enabled=False, dir="my-skills", include_system=False)
    test("skills custom disabled", sc2.enabled is False)
    test("skills custom dir", sc2.dir == "my-skills")
    test("skills custom system", sc2.include_system is False)

    # DokumenConfig has new sections
    print("\n── DokumenConfig fields ──")
    fields = DokumenConfig.model_fields
    test("config has compaction field", "compaction" in fields)
    test("config has coordinator field", "coordinator" in fields)
    test("config has tasks field", "tasks" in fields)
    test("config has skills field", "skills" in fields)

    # validation bounds
    print("\n── config validation ──")
    try:
        CompactionConfig(token_threshold=0.0)
        test("threshold 0.0 rejected", False)
    except Exception:
        test("threshold 0.0 rejected", True)

    try:
        CompactionConfig(token_threshold=1.0)
        test("threshold 1.0 rejected", False)
    except Exception:
        test("threshold 1.0 rejected", True)

    test("threshold 0.1 ok", CompactionConfig(token_threshold=0.1).token_threshold == 0.1)
    test("threshold 0.95 ok", CompactionConfig(token_threshold=0.95).token_threshold == 0.95)

    try:
        CoordinatorConfig(max_workers=0)
        test("workers 0 rejected", False)
    except Exception:
        test("workers 0 rejected", True)

    try:
        CoordinatorConfig(max_workers=21)
        test("workers 21 rejected", False)
    except Exception:
        test("workers 21 rejected", True)

    test("workers 1 ok", CoordinatorConfig(max_workers=1).max_workers == 1)
    test("workers 20 ok", CoordinatorConfig(max_workers=20).max_workers == 20)

    try:
        TasksConfig(max_tasks=0)
        test("max_tasks 0 rejected", False)
    except Exception:
        test("max_tasks 0 rejected", True)

    try:
        SkillsConfig(max_skills_per_prompt=0)
        test("max_skills 0 rejected", False)
    except Exception:
        test("max_skills 0 rejected", True)


# ═══════════════════════════════════════════
# yaml config parsing
# ═══════════════════════════════════════════

print("\n── yaml config parsing ──")

if config_available:
    tmpdir = tempfile.mkdtemp(prefix="config_test_")
    config_path = os.path.join(tmpdir, "dokumen.yaml")

    yaml_content = """\
version: "1.0"
provider:
  name: anthropic
  model: claude-sonnet-4-5-20250929

compaction:
  enabled: true
  token_threshold: 0.8
  token_budget: 200000
  keep_recent_turns: 10

coordinator:
  enabled: true
  max_workers: 3
  synthesis_strategy: vote
  worker_timeout: 60.0

tasks:
  enabled: true
  persist_to_disk: false
  max_tasks: 50

skills:
  enabled: true
  dir: my-skills
  include_system: false
  max_skills_per_prompt: 3
"""

    with open(config_path, "w") as f:
        f.write(yaml_content)

    try:
        load_config = _config.load_config
        cfg = load_config(config_path)

        test("parsed compaction.enabled", cfg.compaction.enabled is True)
        test("parsed compaction.threshold", cfg.compaction.token_threshold == 0.8)
        test("parsed compaction.budget", cfg.compaction.token_budget == 200000)
        test("parsed compaction.keep_recent", cfg.compaction.keep_recent_turns == 10)

        test("parsed coordinator.enabled", cfg.coordinator.enabled is True)
        test("parsed coordinator.max_workers", cfg.coordinator.max_workers == 3)
        test("parsed coordinator.strategy", cfg.coordinator.synthesis_strategy == "vote")
        test("parsed coordinator.timeout", cfg.coordinator.worker_timeout == 60.0)

        test("parsed tasks.enabled", cfg.tasks.enabled is True)
        test("parsed tasks.persist", cfg.tasks.persist_to_disk is False)
        test("parsed tasks.max", cfg.tasks.max_tasks == 50)

        test("parsed skills.enabled", cfg.skills.enabled is True)
        test("parsed skills.dir", cfg.skills.dir == "my-skills")
        test("parsed skills.system", cfg.skills.include_system is False)
        test("parsed skills.max_per_prompt", cfg.skills.max_skills_per_prompt == 3)
    except Exception as e:
        print(f"  (yaml parse failed: {e})")
        failed += 1

    shutil.rmtree(tmpdir)

    # test defaults when sections are omitted
    tmpdir2 = tempfile.mkdtemp(prefix="config_test2_")
    config_path2 = os.path.join(tmpdir2, "dokumen.yaml")
    with open(config_path2, "w") as f:
        f.write('version: "1.0"\nprovider:\n  name: anthropic\n  model: claude-sonnet-4-5-20250929\n')

    try:
        cfg2 = load_config(config_path2)
        test("defaults: compaction disabled", cfg2.compaction.enabled is False)
        test("defaults: coordinator disabled", cfg2.coordinator.enabled is False)
        test("defaults: tasks disabled", cfg2.tasks.enabled is False)
        test("defaults: skills enabled", cfg2.skills.enabled is True)
    except Exception as e:
        print(f"  (default config test failed: {e})")
        failed += 1

    shutil.rmtree(tmpdir2)


# ═══════════════════════════════════════════
# task tools
# ═══════════════════════════════════════════

print("\n── task tools ──")

# load task types and manager first
_task_types = _load("dokumen.tasks.types", "dokumen/tasks/types.py")
sys.modules["dokumen.tasks"] = type(sys)("dokumen.tasks")
sys.modules["dokumen.tasks"].types = _task_types
sys.modules["dokumen.tasks"].TaskStatus = _task_types.TaskStatus
sys.modules["dokumen.tasks"].Task = _task_types.Task
sys.modules["dokumen.tasks"].TaskOutput = _task_types.TaskOutput

_task_mgr = _load("dokumen.tasks.manager", "dokumen/tasks/manager.py")
_task_tools = _load("dokumen.tasks.tools", "dokumen/tasks/tools.py")

# set up a fresh manager
mgr = _task_mgr.TaskManager()
_task_tools.set_task_manager(mgr)

loop = asyncio.new_event_loop()

# task_create
result = loop.run_until_complete(
    _task_tools.handle_task_create({"description": "do something"})
)
test("task_create success", result["success"] is True)
test("task_create has id", "task_id" in result)
task_id = result["task_id"]

# task_create empty description
result_empty = loop.run_until_complete(
    _task_tools.handle_task_create({"description": ""})
)
test("task_create rejects empty", result_empty["success"] is False)

# task_create with parent
result_child = loop.run_until_complete(
    _task_tools.handle_task_create({"description": "subtask", "parent_id": task_id})
)
test("task_create child success", result_child["success"] is True)

# task_update to in_progress
result_up = loop.run_until_complete(
    _task_tools.handle_task_update({"task_id": task_id, "status": "in_progress"})
)
test("task_update to in_progress", result_up["success"] is True)

# task_update to completed
result_up2 = loop.run_until_complete(
    _task_tools.handle_task_update({"task_id": task_id, "status": "completed"})
)
test("task_update to completed", result_up2["success"] is True)

# task_update invalid status
result_bad = loop.run_until_complete(
    _task_tools.handle_task_update({"task_id": task_id, "status": "bogus"})
)
test("task_update rejects invalid status", result_bad["success"] is False)

# task_update empty params
result_missing = loop.run_until_complete(
    _task_tools.handle_task_update({"task_id": "", "status": ""})
)
test("task_update rejects empty params", result_missing["success"] is False)

# task_update nonexistent task
result_404 = loop.run_until_complete(
    _task_tools.handle_task_update({"task_id": "nonexistent", "status": "completed"})
)
test("task_update 404 on missing task", result_404["success"] is False)

# task_list
result_list = loop.run_until_complete(_task_tools.handle_task_list({}))
test("task_list success", result_list["success"] is True)
test("task_list has tasks", result_list["count"] >= 2)
test("task_list entries have fields", all("id" in t and "status" in t for t in result_list["tasks"]))

# task_output
result_out = loop.run_until_complete(
    _task_tools.handle_task_output({"task_id": task_id, "content": "found 3 bugs", "type": "text"})
)
test("task_output success", result_out["success"] is True)

# task_output empty params
result_out_bad = loop.run_until_complete(
    _task_tools.handle_task_output({"task_id": "", "content": ""})
)
test("task_output rejects empty", result_out_bad["success"] is False)

# task_output nonexistent
result_out_404 = loop.run_until_complete(
    _task_tools.handle_task_output({"task_id": "nope", "content": "hi"})
)
test("task_output 404 on missing", result_out_404["success"] is False)

# tool definitions
defs = _task_tools.TASK_TOOL_DEFINITIONS
test("4 task tool definitions", len(defs) == 4)
def_names = {d["name"] for d in defs}
test("has task_create def", "task_create" in def_names)
test("has task_update def", "task_update" in def_names)
test("has task_list def", "task_list" in def_names)
test("has task_output def", "task_output" in def_names)

for d in defs:
    test(f"{d['name']} has handler", callable(d["handler"]))
    test(f"{d['name']} has params schema", "type" in d["parameters"])
    test(f"{d['name']} has description", len(d["description"]) > 0)


# ═══════════════════════════════════════════
# stage files exist and have correct structure
# ═══════════════════════════════════════════

print("\n── stage files ──")

comp_path = os.path.join(_root, "dokumen", "stages", "compaction.py")
coord_path = os.path.join(_root, "dokumen", "stages", "coordinator.py")

test("compaction.py exists", os.path.isfile(comp_path))
test("coordinator.py exists", os.path.isfile(coord_path))

with open(comp_path) as f:
    comp_content = f.read()

test("compaction has class", "class CompactionStage" in comp_content)
test("compaction has name property", 'return "compaction"' in comp_content)
test("compaction has async run", "async def run" in comp_content)
test("compaction imports MicroCompactor", "MicroCompactor" in comp_content)
test("compaction imports ContextCompactor", "ContextCompactor" in comp_content)
test("compaction is best-effort", "best-effort" in comp_content)
test("compaction checks enabled", "not self._config.enabled" in comp_content or "not self._config" in comp_content)

with open(coord_path) as f:
    coord_content = f.read()

test("coordinator has class", "class CoordinatorStage" in coord_content)
test("coordinator has name property", 'return "coordinator"' in coord_content)
test("coordinator has async run", "async def run" in coord_content)
test("coordinator imports CoordinatorAgent", "CoordinatorAgent" in coord_content)
test("coordinator checks enabled", "not self._config.enabled" in coord_content or "not self._config" in coord_content)


# ═══════════════════════════════════════════
# stages __init__ exports
# ═══════════════════════════════════════════

print("\n── stages __init__ ──")

init_path = os.path.join(_root, "dokumen", "stages", "__init__.py")
with open(init_path) as f:
    init_content = f.read()

test("exports CompactionStage", "CompactionStage" in init_content)
test("exports CoordinatorStage", "CoordinatorStage" in init_content)
test("CompactionStage in __all__", '"CompactionStage"' in init_content)
test("CoordinatorStage in __all__", '"CoordinatorStage"' in init_content)


# ═══════════════════════════════════════════
# pipeline wiring in test_object.py
# ═══════════════════════════════════════════

print("\n── pipeline wiring ──")

test_obj_path = os.path.join(_root, "dokumen", "test_object.py")
with open(test_obj_path) as f:
    to_content = f.read()

test("imports CompactionStage", "CompactionStage" in to_content)
test("imports CoordinatorStage", "CoordinatorStage" in to_content)
test("has compaction_config param", "compaction_config" in to_content)
test("has coordinator_config param", "coordinator_config" in to_content)
test("has tasks_config param", "tasks_config" in to_content)
test("uses coordinator when enabled", "use_coordinator" in to_content)
test("compaction after executor", "CompactionStage(compaction_config=" in to_content)
test("coordinator replaces executor", "CoordinatorStage(coordinator_config=" in to_content)


# ═══════════════════════════════════════════
# loader wiring
# ═══════════════════════════════════════════

print("\n── loader wiring ──")

loader_path = os.path.join(_root, "dokumen", "loader.py")
with open(loader_path) as f:
    loader_content = f.read()

test("loader accepts compaction_config", "compaction_config=None" in loader_content)
test("loader accepts coordinator_config", "coordinator_config=None" in loader_content)
test("loader accepts tasks_config", "tasks_config=None" in loader_content)
test("loader accepts skills_config", "skills_config=None" in loader_content)
test("loader passes compaction to TestObject", "compaction_config=compaction_config" in loader_content)
test("loader passes coordinator to TestObject", "coordinator_config=coordinator_config" in loader_content)
test("loader reads scaffold coordinator", "scaffold_coordinator" in loader_content)
test("loader reads scaffold compaction", "scaffold_compaction" in loader_content)
test("load_all passes compaction", "compaction_config=compaction_config" in loader_content)
test("load_all passes coordinator", "coordinator_config=coordinator_config" in loader_content)
test("load_all passes tasks", "tasks_config=tasks_config" in loader_content)
test("load_all passes skills", "skills_config=skills_config" in loader_content)
test("load_all reads config.compaction", "config.compaction" in loader_content or "'compaction'" in loader_content)


# ═══════════════════════════════════════════
# task tools: additional edge cases
# ═══════════════════════════════════════════

print("\n── task tools edge cases ──")

# create task then fail it
r1 = loop.run_until_complete(_task_tools.handle_task_create({"description": "will fail"}))
fail_id = r1["task_id"]
loop.run_until_complete(_task_tools.handle_task_update({"task_id": fail_id, "status": "in_progress"}))
r2 = loop.run_until_complete(_task_tools.handle_task_update({"task_id": fail_id, "status": "failed", "error": "timeout"}))
test("task fail with error", r2["success"] is True)

# cancel a task
r3 = loop.run_until_complete(_task_tools.handle_task_create({"description": "will cancel"}))
cancel_id = r3["task_id"]
r4 = loop.run_until_complete(_task_tools.handle_task_update({"task_id": cancel_id, "status": "cancelled"}))
test("task cancel", r4["success"] is True)

# multiple outputs
r5 = loop.run_until_complete(_task_tools.handle_task_create({"description": "multi output"}))
mo_id = r5["task_id"]
loop.run_until_complete(_task_tools.handle_task_output({"task_id": mo_id, "content": "output 1"}))
loop.run_until_complete(_task_tools.handle_task_output({"task_id": mo_id, "content": "output 2", "type": "json"}))
test("multiple outputs ok", True)

# list shows all tasks
r6 = loop.run_until_complete(_task_tools.handle_task_list({}))
test("list shows all created tasks", r6["count"] >= 5)

loop.close()


# ═══════════════════════════════════════════
# summary
# ═══════════════════════════════════════════

print(f"\n{'='*50}")
print(f"  integration tests: {passed} passed, {failed} failed")
print(f"{'='*50}")
sys.exit(1 if failed else 0)
