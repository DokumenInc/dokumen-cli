from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from dokumen.cli import cli
from dokumen.config import (
    CompactionConfig,
    CoordinatorConfig,
    ExploreConfig,
    TasksConfig,
    load_config,
)
from dokumen.coordinator.types import WorkerStatus, WorkerTask
from dokumen.coordinator.worker import WorkerAgent
from dokumen.loader import load_scaffold
from dokumen.output_schemas import AssertionResult
from dokumen.pipeline import PipelineContext
from dokumen.playwright_tools import get_browser_tool_names
from dokumen.sdk.judge import parse_verdict
from dokumen.sdk.tools import resolve_sdk_tools
from dokumen.sdk.types import ExecutorResult, JudgeVerdict
from dokumen.stages.coordinator import CoordinatorStage
from dokumen.tools.types import ToolDefinition, ToolResult
from dokumen.tools_object import get_all_tool_names
from dokumen_schema.constants import BROWSER_TOOLS, VALID_EXECUTOR_TOOLS


async def _unused_handler(params):
    return ToolResult(success=True, output="")


def test_removed_tools_are_not_public():
    public_tools = set(get_all_tool_names()) | set(VALID_EXECUTOR_TOOLS)

    assert "code_graph_find" not in public_tools
    assert "code_graph_relationships" not in public_tools
    assert "code_graph_dead_code" not in public_tools
    assert "code_graph_complexity" not in public_tools
    assert "read_pdf_section" not in public_tools
    assert "ask" not in public_tools
    assert "code_read_file" not in public_tools
    assert "code_glob" not in public_tools
    assert "code_search" not in public_tools
    assert "code_list_directory" not in public_tools


def test_sdk_resolver_keeps_core_tools_sdk_native():
    result = resolve_sdk_tools(["read_file", "glob", "search_file_content", "run_shell_command"])

    assert result.sdk_tool_names == ["Read", "Glob", "Grep", "Bash"]
    assert result.dokumen_mcp_tools == []


def test_browser_tools_are_sdk_managed_playwright_mcp():
    result = resolve_sdk_tools(["browser_evaluate"], test_name="browser-contract")

    assert set(get_browser_tool_names()) == BROWSER_TOOLS
    assert result.sdk_tool_names == ["Read"]
    assert result.playwright_tool_names == ["mcp__playwright__browser_evaluate"]
    assert result.playwright_mcp_config["type"] == "stdio"


def test_sdk_resolver_exposes_explicit_dokumen_tools_as_mcp():
    read_many_files = ToolDefinition(
        name="read_many_files",
        description="Read multiple files",
        parameters={"type": "object", "properties": {}},
        handler=_unused_handler,
    )

    result = resolve_sdk_tools(
        ["read_file", "read_many_files"],
        dokumen_tool_definitions=[read_many_files],
    )

    assert result.sdk_tool_names == ["Read"]
    assert [tool.name for tool in result.dokumen_mcp_tools] == ["read_many_files"]


def test_sdk_resolver_rejects_unresolved_dokumen_tools():
    with pytest.raises(ValueError, match="Unknown Dokumen tool"):
        resolve_sdk_tools(["read_many_files"])


def test_advanced_runtime_features_default_off():
    assert ExploreConfig().enabled is False
    assert CompactionConfig().enabled is False
    assert CoordinatorConfig().enabled is False
    assert CoordinatorConfig().executor_mode == "sdk"
    assert TasksConfig().enabled is False


def test_checked_in_config_keeps_coordinator_off_by_default():
    config = load_config("dokumen.yaml")

    assert config.coordinator.enabled is False
    assert config.coordinator.executor_mode == "sdk"
    assert config.tasks.enabled is False
    assert config.explore.enabled is False
    assert config.compaction.enabled is False


def test_cli_help_groups_commands_and_keeps_create_removed():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Core Commands:" in result.output
    assert "Supporting Commands:" in result.output
    assert "Experimental Commands:" in result.output
    assert "  run " in result.output
    assert "  validate " in result.output
    assert "  list " in result.output
    assert "  help " in result.output
    assert "  create " not in result.output

    help_result = runner.invoke(cli, ["help"])
    assert help_result.exit_code == 0
    assert "Core Commands:" in help_result.output

    run_help = runner.invoke(cli, ["help", "run"])
    assert run_help.exit_code == 0
    assert "Run skill tests." in run_help.output

    missing = runner.invoke(cli, ["create", "--help"])
    assert missing.exit_code != 0
    assert "No such command 'create'" in missing.output


def test_packaged_authoring_skill_has_valid_frontmatter():
    skill_path = Path(".claude/skills/dokumen-test-author/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")
    _leading, frontmatter, body = content.split("---", 2)
    metadata = yaml.safe_load(frontmatter)

    assert metadata["name"] == "dokumen-test-author"
    assert "Dokumen CLI test scaffolds" in metadata["description"]
    assert "executor.skills" in body
    assert "dokumen validate" in body


async def test_coordinator_stage_returns_canonical_executor_result(tmp_path):
    executor = SimpleNamespace(
        system_prompt="You are testing a skill.",
        user_prompt="Use the release-note-review skill and report the key findings.",
        tools=[SimpleNamespace(name="read_file")],
        provider=None,
    )
    judge = SimpleNamespace(id="success-criteria", system_prompt="Judge the result.")
    ctx = PipelineContext(
        test_id="coordinator-contract",
        reason="Validate coordinator result contract",
        executor=executor,
        judges=[judge],
        files=[],
        timeout=60.0,
        retries=0,
        output_dir=str(tmp_path),
    )

    stage = CoordinatorStage(
        CoordinatorConfig(
            enabled=True,
            max_workers=1,
            worker_timeout=10.0,
            executor_mode="api",
        )
    )

    result_ctx = await stage.run(ctx)

    assert result_ctx.failed is False
    assert isinstance(result_ctx.executor_output, ExecutorResult)
    assert result_ctx.executor_output.success is True
    assert "[stub] would execute:" in result_ctx.executor_output.final_response
    assert result_ctx.executor_output.original_user_prompt == (
        "Use the release-note-review skill and report the key findings."
    )
    assert f"OUTPUT FOLDER: {tmp_path}" in result_ctx.executor_output.user_prompt
    assert f"OUTPUT FOLDER: {tmp_path}" in judge.system_prompt


async def test_coordinator_sdk_worker_mode_maps_executor_result(monkeypatch):
    from dokumen.sdk.agent_wrapper import SdkExecutorWrapper

    async def fake_run(self):
        return ExecutorResult(
            success=True,
            final_response="sdk worker completed",
            tool_calls=[{"tool_name": "Read", "parameters": {"file_path": "README.md"}}],
            input_tokens=7,
            output_tokens=11,
        )

    monkeypatch.setattr(SdkExecutorWrapper, "run", fake_run)

    worker = WorkerAgent(
        provider=SimpleNamespace(model="claude-haiku-4-5-20251001"),
        executor_mode="sdk",
    )
    result = await worker.run(
        WorkerTask(
            id="sdk-worker-contract",
            name="sdk-worker",
            goal="Read the README and report the purpose.",
            tools=["read_file"],
            timeout=10.0,
        )
    )

    assert result.status is WorkerStatus.COMPLETED
    assert result.output == "sdk worker completed"
    assert result.tool_calls == [{"tool_name": "Read", "parameters": {"file_path": "README.md"}}]
    assert result.input_tokens == 7
    assert result.output_tokens == 11


def test_executor_is_normally_prompted_to_use_a_named_skill(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "release-note-review.md").write_text(
        "# Release Note Review\n\n"
        "When reviewing release notes, check that the audience, changed behavior, "
        "and required user action are all explicit.\n",
        encoding="utf-8",
    )

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "release-notes.md").write_text(
        "Feature flags now require an owner field before rollout.\n",
        encoding="utf-8",
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    scaffold_path = tests_dir / "release-note-skill.test.yaml"
    scaffold_path.write_text(
        yaml.safe_dump(
            {
                "name": "release-note-skill",
                "reason": "Verify the executor applies a named skill before the judge evaluates it.",
                "files": [{"path": "docs/release-notes.md"}],
                "executor": {
                    "skills": ["release-note-review"],
                    "tools": ["read_file"],
                    "user_prompt": (
                        "Use the release-note-review skill to inspect the referenced "
                        "release notes file. Report the audience, changed behavior, "
                        "and required user action."
                    ),
                },
                "judges": [
                    {
                        "name": "skill-success-criteria",
                        "include_executor_output": True,
                        "system_prompt": (
                            "Pass only if the executor explicitly applied the "
                            "release-note-review skill and reported audience, "
                            "changed behavior, and required user action. Return JSON: "
                            '{"verdict": "PASS" or "FAIL", "reason": "..."}'
                        ),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    test = load_scaffold(str(scaffold_path), project_root=str(tmp_path))

    assert "Use the release-note-review skill" in test.executor.user_prompt
    assert "## Available Skills" in test.executor.system_prompt
    assert "Release Note Review" in test.executor.system_prompt
    assert "release-note-review" in test.resolved_skills
    assert test.coordinator_config is None or test.coordinator_config.enabled is False


def test_judge_results_do_not_expose_unreliable_score():
    metric = "confi" + "dence"
    parsed = parse_verdict(f'{{"verdict": "PASS", "{metric}": 0.99, "reason": "ok"}}')
    result = JudgeVerdict(judge_id="groundedness", passed=True, reason=parsed.reason)
    assertion = AssertionResult(assertion="groundedness", passed=True, reasoning="ok")

    assert not hasattr(parsed, metric)
    assert metric not in result.to_dict()
    assert metric not in assertion.model_dump()
