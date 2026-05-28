import tomllib
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
from dokumen.logging_config import get_logger
from dokumen.output_schemas import AssertionResult
from dokumen.pipeline import PipelineContext
from dokumen.playwright_tools import get_browser_tool_names
from dokumen.sdk.judge import parse_verdict
from dokumen.sdk.tools import resolve_sdk_tools
from dokumen.sdk.types import ExecutorResult, JudgeVerdict
from dokumen.stages.coordinator import CoordinatorStage
from dokumen.stages.prompting import (
    ensure_final_response_from_conversation,
    prepare_agent_prompts,
    prompt_hash,
)
from dokumen.tools.types import ToolDefinition, ToolResult
from dokumen.tools_object import get_all_tool_names
from dokumen_schema.constants import BROWSER_TOOLS, VALID_EXECUTOR_TOOLS


async def _unused_handler(params):
    return ToolResult(success=True, output="")


def _write_minimal_sop_project(root: Path) -> None:
    (root / "docs").mkdir()
    (root / "sops").mkdir()
    (root / "tests").mkdir()

    (root / "dokumen.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "provider": {
                    "name": "anthropic",
                    "model": "claude-haiku-4-5-20251001",
                },
                "execution": {"timeout": 600},
                "explore": {"enabled": False},
                "compaction": {"enabled": False},
                "coordinator": {"enabled": False},
                "tasks": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "docs" / "customer-ticket.md").write_text(
        "Northstar Logistics is on an Enterprise plan and requests a $1,200 refund.\n",
        encoding="utf-8",
    )
    (root / "sops" / "refund-escalation-sop.md").write_text(
        "# Refund Escalation SOP\n\n"
        "Escalate to Finance when the amount is over $500 or the customer is on "
        "an enterprise plan.\n",
        encoding="utf-8",
    )
    (root / "tests" / "refund-escalation.test.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "refund-escalation",
                "reason": "Verify that the executor follows the refund escalation SOP.",
                "files": [{"path": "docs/customer-ticket.md"}],
                "executor": {
                    "sops": ["refund-escalation-sop"],
                    "tools": ["read_file"],
                    "user_prompt": "Follow the refund-escalation-sop for the ticket.",
                },
                "judges": [
                    {
                        "name": "sop-success-criteria",
                        "include_executor_output": True,
                        "system_prompt": (
                            "Pass only if the executor followed the SOP. Return JSON: "
                            '{"verdict": "PASS" or "FAIL", "reason": "..."}'
                        ),
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


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
    assert "Business SOP Agent Test CLI" in result.output
    assert "Test whether agents follow business SOPs" in result.output
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
    assert "Run agent SOP tests." in run_help.output

    nested_help = runner.invoke(cli, ["help", "list", "tests"])
    assert nested_help.exit_code == 0
    assert "List all test scaffolds." in nested_help.output

    unknown_help = runner.invoke(cli, ["help", "missing"])
    assert unknown_help.exit_code != 0
    assert "No such command: missing" in unknown_help.output

    missing = runner.invoke(cli, ["create", "--help"])
    assert missing.exit_code != 0
    assert "No such command 'create'" in missing.output


def test_default_cli_output_suppresses_internal_info_logs(tmp_path, monkeypatch):
    _write_minimal_sop_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    validate_result = runner.invoke(cli, ["validate"])
    assert validate_result.exit_code == 0
    assert "Validation passed" in validate_result.output
    assert "[INFO" not in validate_result.output
    assert "cli.start" not in validate_result.output
    assert "CI compat check" not in validate_result.output

    dry_run_result = runner.invoke(cli, ["run", "refund-escalation", "--dry-run"])
    assert dry_run_result.exit_code == 0
    assert "Would run 1 test(s):" in dry_run_result.output
    assert "refund-escalation" in dry_run_result.output
    assert "[INFO" not in dry_run_result.output
    assert "loader.instruction" not in dry_run_result.output


def test_distribution_metadata_shares_only_the_public_cli_surface():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert (
        project["description"]
        == "CLI for testing whether agents follow business SOPs with LLM judges"
    )
    assert project["urls"]["Repository"] == "https://github.com/DokumenInc/dokumen-cli"
    assert project["urls"]["Documentation"].endswith("#readme")
    assert "business-sops" in project["keywords"]
    assert "agent-evals" in project["keywords"]
    assert "Topic :: Software Development :: Testing" in project["classifiers"]
    assert project["scripts"] == {"dokumen": "dokumen.cli:cli"}

    dependencies = "\n".join(project["dependencies"])
    assert "onepassword-sdk" not in dependencies
    assert "sentry-sdk" not in dependencies
    assert "onepassword-sdk>=0.3.0" in project["optional-dependencies"]["integrations"]
    assert "sentry-sdk>=1.40.0" in project["optional-dependencies"]["integrations"]


def test_manifest_includes_docs_examples_and_packaged_authoring_skill():
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

    assert "include README.md" in manifest
    assert "include LICENSE" in manifest
    assert "include .claude/skills/dokumen-test-author/SKILL.md" in manifest
    assert "recursive-include docs *.md" in manifest
    assert "recursive-include examples/business-sop *.md *.yaml *.txt" in manifest
    assert "recursive-include tests *.py *.yaml *.txt *.html" in manifest


def test_packaged_authoring_skill_has_valid_frontmatter():
    skill_path = Path(".claude/skills/dokumen-test-author/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")
    _leading, frontmatter, body = content.split("---", 2)
    metadata = yaml.safe_load(frontmatter)

    assert metadata["name"] == "dokumen-test-author"
    assert "business SOP adherence" in metadata["description"]
    assert "executor.sops" in body
    assert "dokumen validate" in body


async def test_coordinator_stage_returns_canonical_executor_result(tmp_path):
    executor = SimpleNamespace(
        system_prompt="You are testing an SOP.",
        user_prompt="Follow the refund escalation SOP and report the key findings.",
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
        "Follow the refund escalation SOP and report the key findings."
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


def test_shared_prompt_preparation_updates_executor_and_judges(tmp_path):
    executor = SimpleNamespace(
        system_prompt="You are testing an SOP.",
        user_prompt="Follow the refund escalation SOP.",
    )
    judge = SimpleNamespace(id="success-criteria", system_prompt="Judge the executor output.")
    ctx = PipelineContext(
        test_id="prompt-contract",
        reason="Validate shared prompt preparation.",
        executor=executor,
        judges=[judge],
        files=[],
        timeout=60.0,
        retries=0,
        output_dir=str(tmp_path),
    )

    prepare_agent_prompts(ctx, "executor", get_logger("tests.prompting"))

    assert ctx.original_user_prompt == "Follow the refund escalation SOP."
    assert f"OUTPUT FOLDER: {tmp_path}" in ctx.executor.user_prompt
    assert f"OUTPUT FOLDER: {tmp_path}" in judge.system_prompt
    assert ctx.original_judge_prompts == {"success-criteria": "Judge the executor output."}
    assert len(prompt_hash(ctx.executor.user_prompt)) == 12
    assert prompt_hash(ctx.executor.user_prompt) == prompt_hash(ctx.executor.user_prompt)
    assert prompt_hash(None) == "none"


def test_final_response_reconstruction_uses_assistant_conversation_chunks():
    result = ExecutorResult(
        success=False,
        final_response="",
        conversation_log=[
            {"role": "user", "content": "ignore the user's prompt"},
            {"role": "assistant", "content": "First assistant chunk."},
            {"role": "assistant", "content": "Second assistant chunk."},
            {"role": "assistant", "content": "   "},
            {"role": "tool", "content": "ignore tool output"},
        ],
    )

    changed = ensure_final_response_from_conversation(
        result,
        get_logger("tests.prompting"),
        "executor-output-contract",
    )

    assert changed is True
    assert result.final_response == "First assistant chunk.\n\nSecond assistant chunk."
    assert (
        ensure_final_response_from_conversation(
            result,
            get_logger("tests.prompting"),
            "executor-output-contract",
        )
        is False
    )


def test_executor_is_normally_prompted_to_follow_a_named_sop(tmp_path):
    sop_dir = tmp_path / "sops"
    sop_dir.mkdir()
    (sop_dir / "refund-escalation-sop.md").write_text(
        "# Refund Escalation SOP\n\n"
        "When reviewing refund requests, identify the plan, amount, refund-window "
        "status, escalation requirement, and next action.\n",
        encoding="utf-8",
    )

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "customer-ticket.md").write_text(
        "Northstar Logistics is on an Enterprise plan and requests a $1,200 refund "
        "18 days after payment.\n",
        encoding="utf-8",
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    scaffold_path = tests_dir / "refund-escalation.test.yaml"
    scaffold_path.write_text(
        yaml.safe_dump(
            {
                "name": "refund-escalation",
                "reason": "Verify the executor follows a named SOP before judging.",
                "files": [{"path": "docs/customer-ticket.md"}],
                "executor": {
                    "sops": ["refund-escalation-sop"],
                    "tools": ["read_file"],
                    "user_prompt": (
                        "Follow the refund-escalation-sop while reviewing the referenced "
                        "customer ticket. Report the plan, amount, refund-window status, "
                        "escalation requirement, and next action."
                    ),
                },
                "judges": [
                    {
                        "name": "sop-success-criteria",
                        "include_executor_output": True,
                        "system_prompt": (
                            "Pass only if the executor explicitly followed the "
                            "refund escalation SOP and reported plan, amount, "
                            "refund-window status, escalation requirement, and next "
                            "action. Return JSON: "
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

    assert "Follow the refund-escalation-sop" in test.executor.user_prompt
    assert "## Available Instructions and SOPs" in test.executor.system_prompt
    assert "Refund Escalation SOP" in test.executor.system_prompt
    assert "refund-escalation-sop" in test.resolved_skills
    assert test.coordinator_config is None or test.coordinator_config.enabled is False


def test_judge_results_do_not_expose_unreliable_score():
    metric = "confi" + "dence"
    parsed = parse_verdict(f'{{"verdict": "PASS", "{metric}": 0.99, "reason": "ok"}}')
    result = JudgeVerdict(judge_id="groundedness", passed=True, reason=parsed.reason)
    assertion = AssertionResult(assertion="groundedness", passed=True, reasoning="ok")

    assert not hasattr(parsed, metric)
    assert metric not in result.to_dict()
    assert metric not in assertion.model_dump()
