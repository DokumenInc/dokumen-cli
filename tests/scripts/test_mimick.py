#!/usr/bin/env python3
"""standalone tests for mimick skill-based architecture analysis.

tests the prompt file, cli command wiring, and skill loader integration.

run: python3 tests/scripts/test_mimick.py
"""
import sys
import os
import tempfile
import shutil
import importlib.util

_root = os.path.join(os.path.dirname(__file__), "..", "..")

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


# ═══════════════════════════════════════════
# prompt file
# ═══════════════════════════════════════════

print("── mimick prompt file ──")

prompt_path = os.path.join(_root, "dokumen", "prompts", "executors", "mimick.txt")
test("prompt file exists", os.path.isfile(prompt_path))

with open(prompt_path) as f:
    prompt_content = f.read()

test("prompt not empty", len(prompt_content) > 100)
test("prompt mentions architecture", "architecture" in prompt_content.lower())
test("prompt mentions blueprint", "blueprint" in prompt_content.lower())
test("prompt mentions yaml", "yaml" in prompt_content.lower() or "YAML" in prompt_content)
test("prompt has phases", "phase" in prompt_content.lower())
test("prompt mentions modules", "module" in prompt_content.lower())
test("prompt mentions patterns", "pattern" in prompt_content.lower())
test("prompt mentions dependencies", "dependenc" in prompt_content.lower())
test("prompt has survey phase", "survey" in prompt_content.lower())
test("prompt not copying code disclaimer", "not copying" in prompt_content.lower() or "NOT copying" in prompt_content)


# ═══════════════════════════════════════════
# cli command module
# ═══════════════════════════════════════════

print("\n── cli command module ──")

cmd_path = os.path.join(_root, "dokumen", "cli", "commands", "mimick.py")
test("command file exists", os.path.isfile(cmd_path))

with open(cmd_path) as f:
    cmd_content = f.read()

test("uses click", "import click" in cmd_content)
test("has click.command decorator", '@click.command("mimick")' in cmd_content)
test("has source_path argument", "source_path" in cmd_content)
test("has name option", '"--name"' in cmd_content or "'--name'" in cmd_content)
test("has output option", '"--output"' in cmd_content or "'--output'" in cmd_content)
test("has timeout option", '"--timeout"' in cmd_content or "'--timeout'" in cmd_content)
test("uses executor agent", "ExecutorAgent" in cmd_content)
test("uses sdk tools", "resolve_sdk_tools" in cmd_content)
test("reads mimick prompt", "mimick.txt" in cmd_content)
test("has read-only tools", "read_file" in cmd_content and "list_directory" in cmd_content)
test("has glob tool", "glob" in cmd_content)
test("has search tool", "search_file_content" in cmd_content)
test("async run function", "async def _run_mimick" in cmd_content)
test("writes output to file", 'open(output, "w")' in cmd_content)
test("has fallback prompt", "fallback" in cmd_content.lower())


# ═══════════════════════════════════════════
# skill loader integration
# ═══════════════════════════════════════════

print("\n── skill loader ──")

import yaml
sys.modules["yaml"] = yaml

_loader = _load("dokumen.skills.loader", "dokumen/skills/loader.py")

ExecutionMode = _loader.ExecutionMode
SkillDefinition = _loader.SkillDefinition
SYSTEM_SKILLS = _loader.SYSTEM_SKILLS

test("ExecutionMode.INLINE exists", ExecutionMode.INLINE.value == "inline")
test("ExecutionMode.FORK exists", ExecutionMode.FORK.value == "fork")

# skill definition basics
sd = SkillDefinition(name="test-skill", description="a test", prompt="do something")
test("skill creation", sd.name == "test-skill")
test("skill default mode", sd.mode == ExecutionMode.INLINE)
test("skill default tools", sd.tools == [])
test("skill default effectiveness", sd.effectiveness == 1.0)

# to_dict / from_dict roundtrip
d = sd.to_dict()
test("to_dict has name", d["name"] == "test-skill")
test("to_dict has mode", d["mode"] == "inline")

sd2 = SkillDefinition.from_dict(d)
test("from_dict roundtrip name", sd2.name == "test-skill")
test("from_dict roundtrip mode", sd2.mode == ExecutionMode.INLINE)

# from_dict with defaults
sd3 = SkillDefinition.from_dict({})
test("from_dict default name", sd3.name == "unnamed")
test("from_dict default mode", sd3.mode == ExecutionMode.INLINE)

# fork mode skill
sd_fork = SkillDefinition(name="forked", mode=ExecutionMode.FORK, tools=["read_file"])
test("fork mode", sd_fork.mode == ExecutionMode.FORK)
test("fork tools", sd_fork.tools == ["read_file"])

# with model override
sd_model = SkillDefinition(name="custom", model="haiku")
test("model override", sd_model.model == "haiku")

# when_to_use
sd_when = SkillDefinition(name="x", when_to_use="when testing")
test("when_to_use", sd_when.when_to_use == "when testing")

# metadata
sd_meta = SkillDefinition(name="x", metadata={"version": "1.0"})
test("metadata", sd_meta.metadata == {"version": "1.0"})

# system skills
print("\n── system skills ──")

test("has system skills", len(SYSTEM_SKILLS) >= 3)

skill_names = {s.name for s in SYSTEM_SKILLS}
test("mimick is system", "mimick" in skill_names)
test("qa-check is system", "qa-check" in skill_names)
test("link-check is system", "link-check" in skill_names)

mimick_skill = next(s for s in SYSTEM_SKILLS if s.name == "mimick")
test("mimick is fork mode", mimick_skill.mode == ExecutionMode.FORK)
test("mimick has tools", len(mimick_skill.tools) >= 4)
test("mimick has read_file tool", "read_file" in mimick_skill.tools)
test("mimick has description", len(mimick_skill.description) > 0)
test("mimick has when_to_use", len(mimick_skill.when_to_use) > 0)
test("mimick prompt references file", "mimick" in mimick_skill.prompt.lower())

qa_skill = next(s for s in SYSTEM_SKILLS if s.name == "qa-check")
test("qa-check is inline", qa_skill.mode == ExecutionMode.INLINE)
test("qa-check has tools", len(qa_skill.tools) >= 2)

link_skill = next(s for s in SYSTEM_SKILLS if s.name == "link-check")
test("link-check is fork", link_skill.mode == ExecutionMode.FORK)
test("link-check has web_fetch", "web_fetch" in link_skill.tools)


# ═══════════════════════════════════════════
# skill file loading
# ═══════════════════════════════════════════

print("\n── skill file loading ──")

# load nonexistent file
result = _loader.load_skill_file("/nonexistent/skill.yaml")
test("nonexistent file returns None", result is None)

# load markdown skill file
tmpdir = tempfile.mkdtemp(prefix="skill_test_")
md_path = os.path.join(tmpdir, "my-skill.md")
with open(md_path, "w") as f:
    f.write("you are a helpful assistant. do the thing.\n")

md_skill = _loader.load_skill_file(md_path)
test("md skill loaded", md_skill is not None)
test("md skill name from filename", md_skill.name == "my-skill")
test("md skill prompt is content", "helpful assistant" in md_skill.prompt)

# load from directory
dir_skills = _loader.load_skills_from_directory(tmpdir)
test("directory loading finds skill", len(dir_skills) >= 1)

# empty directory
empty_dir = tempfile.mkdtemp(prefix="empty_skill_")
empty_skills = _loader.load_skills_from_directory(empty_dir)
test("empty dir returns empty", empty_skills == [])

# nonexistent directory
none_skills = _loader.load_skills_from_directory("/nonexistent/skills/dir")
test("nonexistent dir returns empty", none_skills == [])

shutil.rmtree(tmpdir)
shutil.rmtree(empty_dir)

# get_all_skills
print("\n── get_all_skills ──")

all_skills = _loader.get_all_skills(include_system=True)
test("get_all includes system", len(all_skills) >= 3)

no_system = _loader.get_all_skills(include_system=False)
test("no system returns empty", len(no_system) == 0)

# project skills override system
tmpdir2 = tempfile.mkdtemp(prefix="proj_skills_")
override_path = os.path.join(tmpdir2, "mimick.md")
with open(override_path, "w") as f:
    f.write("custom mimick prompt override\n")

with_override = _loader.get_all_skills(project_skills_dir=tmpdir2, include_system=True)
mimick_override = next((s for s in with_override if s.name == "mimick"), None)
test("project skill overrides system", mimick_override is not None)
test("override has custom prompt", "custom mimick" in mimick_override.prompt)

shutil.rmtree(tmpdir2)


# ═══════════════════════════════════════════
# mimick package
# ═══════════════════════════════════════════

print("\n── mimick package ──")

init_path = os.path.join(_root, "dokumen", "mimick", "__init__.py")
test("mimick package exists", os.path.isfile(init_path))

with open(init_path) as f:
    init_content = f.read()

test("init has docstring", '"""' in init_content)
test("init mentions architecture", "architecture" in init_content.lower())

# deleted files should not exist
test("types.py deleted", not os.path.isfile(os.path.join(_root, "dokumen", "mimick", "types.py")))
test("explorer.py deleted", not os.path.isfile(os.path.join(_root, "dokumen", "mimick", "explorer.py")))
test("pattern_detector.py deleted", not os.path.isfile(os.path.join(_root, "dokumen", "mimick", "pattern_detector.py")))
test("blueprint.py deleted", not os.path.isfile(os.path.join(_root, "dokumen", "mimick", "blueprint.py")))


# ═══════════════════════════════════════════
# edge cases
# ═══════════════════════════════════════════

print("\n── edge cases ──")

# skill with all fields
full_skill = SkillDefinition(
    name="full",
    description="a full skill",
    prompt="do everything",
    mode=ExecutionMode.FORK,
    tools=["read_file", "glob"],
    model="opus",
    when_to_use="always",
    effectiveness=0.5,
    metadata={"author": "test"},
)
fd = full_skill.to_dict()
full_rt = SkillDefinition.from_dict(fd)
test("full roundtrip name", full_rt.name == "full")
test("full roundtrip mode", full_rt.mode == ExecutionMode.FORK)
test("full roundtrip tools", full_rt.tools == ["read_file", "glob"])
test("full roundtrip model", full_rt.model == "opus")
test("full roundtrip effectiveness", full_rt.effectiveness == 0.5)
test("full roundtrip metadata", full_rt.metadata == {"author": "test"})

# effectiveness decay
test("effectiveness range low", SkillDefinition(name="x", effectiveness=0.0).effectiveness == 0.0)
test("effectiveness range high", SkillDefinition(name="x", effectiveness=1.0).effectiveness == 1.0)

# empty prompt
test("empty prompt allowed", SkillDefinition(name="x", prompt="").prompt == "")


# ═══════════════════════════════════════════
# summary
# ═══════════════════════════════════════════

print(f"\n{'='*50}")
print(f"  mimick tests: {passed} passed, {failed} failed")
print(f"{'='*50}")
sys.exit(1 if failed else 0)
