import pytest

from dokumen.config import CoordinatorConfig, TasksConfig
from dokumen.output_schemas import AssertionResult
from dokumen.playwright_tools import get_browser_tool_names
from dokumen.sdk.judge import parse_verdict
from dokumen.sdk.tools import resolve_sdk_tools
from dokumen.sdk.types import JudgeVerdict
from dokumen.tools_object import ToolDefinition, ToolResult, get_all_tool_names
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
    assert CoordinatorConfig().enabled is False
    assert TasksConfig().enabled is False


def test_judge_results_do_not_expose_unreliable_score():
    metric = "confi" + "dence"
    parsed = parse_verdict(f'{{"verdict": "PASS", "{metric}": 0.99, "reason": "ok"}}')
    result = JudgeVerdict(judge_id="groundedness", passed=True, reason=parsed.reason)
    assertion = AssertionResult(assertion="groundedness", passed=True, reasoning="ok")

    assert not hasattr(parsed, metric)
    assert metric not in result.to_dict()
    assert metric not in assertion.model_dump()
