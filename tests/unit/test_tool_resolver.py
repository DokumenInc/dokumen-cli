"""Tests for tool_resolver module — tool resolution and auto-injection."""
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from dokumen.tool_resolver import (
    ToolProvenance,
    determine_executor_tool_names,
    enforce_allowed_list,
    auto_inject_tools,
    filter_tools_with_overrides,
    filter_judge_tools,
    resolve_tools,
    _create_placeholder_tool,
)


class TestToolProvenance:
    """Tests for ToolProvenance dataclass."""

    def test_default_empty(self):
        p = ToolProvenance()
        assert p.executor_tools == {}
        assert p.judge_tools == {}
        assert p.explore_tools == {}
        assert p.overrides_active is False
        assert p.removed_tools == []

    def test_to_dict_returns_copies(self):
        p = ToolProvenance(
            executor_tools={"read_file": "scaffold"},
            judge_tools={"accuracy": {"run_shell_command": "auto:standard"}},
            overrides_active=True,
            removed_tools=["web_fetch"],
        )
        d = p.to_dict()
        assert d["executor_tools"] == {"read_file": "scaffold"}
        assert d["overrides_active"] is True
        assert d["removed_tools"] == ["web_fetch"]
        # Verify it's a copy, not reference
        d["executor_tools"]["new_tool"] = "test"
        assert "new_tool" not in p.executor_tools

    def test_to_dict_deep_copies_judge_tools(self):
        p = ToolProvenance(
            judge_tools={"j1": {"t1": "scaffold"}},
        )
        d = p.to_dict()
        d["judge_tools"]["j1"]["t2"] = "new"
        assert "t2" not in p.judge_tools["j1"]


class TestDetermineExecutorToolNames:
    """Tests for determine_executor_tool_names."""

    def test_scaffold_tools_override_defaults(self):
        prov = ToolProvenance()
        result = determine_executor_tool_names(
            scaffold_tools=["read_file", "glob"],
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        assert result == ["read_file", "glob"]
        assert prov.executor_tools["read_file"] == "scaffold"
        assert prov.executor_tools["glob"] == "scaffold"

    def test_global_defaults_when_no_scaffold_tools(self):
        prov = ToolProvenance()
        tools_config = SimpleNamespace(defaults=["read_file", "run_shell_command"], allowed=None, blocked=None)
        result = determine_executor_tool_names(
            scaffold_tools=None,
            tools_config=tools_config,
            scaffold_name="test",
            provenance=prov,
        )
        assert result == ["read_file", "run_shell_command"]
        assert prov.executor_tools["read_file"] == "defaults"

    def test_empty_when_no_scaffold_or_defaults(self):
        prov = ToolProvenance()
        result = determine_executor_tool_names(
            scaffold_tools=None,
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        assert result == []


class TestEnforceAllowedList:
    """Tests for enforce_allowed_list."""

    def test_passes_when_all_allowed(self):
        tools_config = SimpleNamespace(allowed=["read_file", "glob"], defaults=None, blocked=None)
        # Should not raise
        enforce_allowed_list(["read_file", "glob"], tools_config, "test")

    def test_raises_when_disallowed(self):
        tools_config = SimpleNamespace(allowed=["read_file"], defaults=None, blocked=None)
        with pytest.raises(ValueError, match="not in allowed list"):
            enforce_allowed_list(["read_file", "web_fetch"], tools_config, "test")

    def test_no_allowed_list_passes_everything(self):
        tools_config = SimpleNamespace(allowed=None, defaults=None, blocked=None)
        enforce_allowed_list(["anything", "goes"], tools_config, "test")

    def test_no_tools_config_passes(self):
        enforce_allowed_list(["read_file"], None, "test")


class TestAutoInjectTools:
    """Tests for auto_inject_tools."""

    @patch("dokumen.agent_loader.get_agent_tools", return_value=[])
    def test_auto_injects_shell_command_for_standard(self, mock_agent_tools):
        prov = ToolProvenance()
        result, auto = auto_inject_tools(
            executor_tool_names=["read_file"],
            is_browser_agent=False,
            is_research_agent=False,
            is_code_agent=False,
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        assert "run_shell_command" in result
        assert "run_shell_command" in auto
        assert prov.executor_tools["run_shell_command"] == "auto:standard"

    @patch("dokumen.agent_loader.get_agent_tools", return_value=[])
    def test_no_shell_for_browser_agent(self, mock_agent_tools):
        prov = ToolProvenance()
        result, auto = auto_inject_tools(
            executor_tool_names=["read_file"],
            is_browser_agent=True,
            is_research_agent=False,
            is_code_agent=False,
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        assert "run_shell_command" not in auto

    @patch("dokumen.agent_loader.get_agent_tools", return_value=[])
    def test_web_search_for_research_agent(self, mock_agent_tools):
        prov = ToolProvenance()
        result, auto = auto_inject_tools(
            executor_tool_names=["read_file"],
            is_browser_agent=False,
            is_research_agent=True,
            is_code_agent=False,
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        assert "web_search" in result
        assert "web_search" in auto
        assert prov.executor_tools["web_search"] == "auto:research"

    @patch("dokumen.agent_loader.get_agent_tools", return_value=[])
    def test_code_tools_for_code_agent(self, mock_agent_tools):
        prov = ToolProvenance()
        result, auto = auto_inject_tools(
            executor_tool_names=[],
            is_browser_agent=False,
            is_research_agent=False,
            is_code_agent=True,
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        assert "code_read_file" in result
        assert "code_search" in result
        assert "code_glob" in result

    @patch("dokumen.agent_loader.get_agent_tools", return_value=["custom_tool"])
    def test_agent_db_tools_merged(self, mock_agent_tools):
        prov = ToolProvenance()
        result, auto = auto_inject_tools(
            executor_tool_names=["read_file"],
            is_browser_agent=False,
            is_research_agent=False,
            is_code_agent=False,
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        assert "custom_tool" in result
        assert prov.executor_tools["custom_tool"] == "agent:db"


class TestFilterToolsWithOverrides:
    """Tests for filter_tools_with_overrides."""

    def test_override_removes_disabled_tool(self):
        prov = ToolProvenance(executor_tools={"read_file": "scaffold", "web_fetch": "scaffold"})
        overrides = MagicMock()

        with patch("dokumen.tool_resolver.is_tool_enabled_for_test") as mock_enabled:
            mock_enabled.side_effect = lambda name, ov: name != "web_fetch"
            result = filter_tools_with_overrides(
                ["read_file", "web_fetch"],
                auto_injected_tools=set(),
                overrides=overrides,
                tools_config=None,
                scaffold_name="test",
                scaffold_agent=None,
                provenance=prov,
            )
        assert "web_fetch" not in result
        assert "read_file" in result

    def test_override_raises_for_auto_injected_disabled(self):
        prov = ToolProvenance()
        overrides = MagicMock()

        with patch("dokumen.tool_resolver.is_tool_enabled_for_test", return_value=False):
            with pytest.raises(ValueError, match="required for agent"):
                filter_tools_with_overrides(
                    ["run_shell_command"],
                    auto_injected_tools={"run_shell_command"},
                    overrides=overrides,
                    tools_config=None,
                    scaffold_name="test",
                    scaffold_agent="standard",
                    provenance=prov,
                )

    def test_blocked_list_removes_tools(self):
        prov = ToolProvenance(executor_tools={"read_file": "scaffold", "web_fetch": "scaffold"})
        tools_config = SimpleNamespace(
            blocked=["web_fetch"],
            allowed=None,
            defaults=None,
            config=None,
        )
        result = filter_tools_with_overrides(
            ["read_file", "web_fetch"],
            auto_injected_tools=set(),
            overrides=None,
            tools_config=tools_config,
            scaffold_name="test",
            scaffold_agent=None,
            provenance=prov,
        )
        assert "web_fetch" not in result
        assert "read_file" in result
        assert "web_fetch" in prov.removed_tools


class TestFilterJudgeTools:
    """Tests for filter_judge_tools."""

    def test_blocked_list_removes_judge_tools(self):
        tools_config = SimpleNamespace(blocked=["web_fetch"], allowed=None, defaults=None, config=None)
        judge_prov = {"read_file": "scaffold", "web_fetch": "scaffold"}
        result = filter_judge_tools(
            ["read_file", "web_fetch"],
            auto_added_judge_tools=set(),
            overrides=None,
            tools_config=tools_config,
            scaffold_name="test",
            judge_name="accuracy",
            scaffold_agent=None,
            judge_prov=judge_prov,
        )
        assert "web_fetch" not in result
        assert "web_fetch" not in judge_prov


class TestResolveTools:
    """Tests for resolve_tools."""

    def test_resolve_builtin_read_file(self, tmp_path):
        result = resolve_tools(["read_file"], base_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "read_file"

    def test_resolve_builtin_glob(self, tmp_path):
        result = resolve_tools(["glob"], base_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "glob"

    def test_resolve_shell_command(self, tmp_path):
        result = resolve_tools(["run_shell_command"], base_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "run_shell_command"

    def test_unknown_tool_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown tool"):
            resolve_tools(["nonexistent_tool_xyz"], base_dir=str(tmp_path))

    def test_code_tools_without_config_raises(self, tmp_path):
        with pytest.raises(ValueError, match="requires code_repos_config"):
            resolve_tools(["code_read_file"], base_dir=str(tmp_path))


class TestResolveToolsAdditional:
    """Additional resolve_tools tests for coverage."""

    def test_resolve_search_file_content(self, tmp_path):
        result = resolve_tools(["search_file_content"], base_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "search_file_content"

    def test_resolve_web_fetch(self, tmp_path):
        result = resolve_tools(["web_fetch"], base_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "web_fetch"

    def test_resolve_list_directory(self, tmp_path):
        result = resolve_tools(["list_directory"], base_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "list_directory"

    def test_resolve_multiple_tools(self, tmp_path):
        result = resolve_tools(["read_file", "glob", "run_shell_command"], base_dir=str(tmp_path))
        assert len(result) == 3
        names = [t.name for t in result]
        assert "read_file" in names
        assert "glob" in names
        assert "run_shell_command" in names

    def test_code_graph_tools_without_config_raises(self, tmp_path):
        with pytest.raises(ValueError, match="requires code_repos_config"):
            resolve_tools(["code_graph_find"], base_dir=str(tmp_path))

    def test_resolve_delegate_without_registry_creates_placeholder(self, tmp_path):
        result = resolve_tools(["delegate_to_agent"], base_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "delegate_to_agent"
        assert "Placeholder" in result[0].description

    def test_resolve_context_tool_creates_placeholder(self, tmp_path):
        """Context tools should get placeholders."""
        # Context tools are in CONTEXT_TOOLS dict
        from dokumen.tools_object import CONTEXT_TOOLS
        if CONTEXT_TOOLS:
            ctx_name = list(CONTEXT_TOOLS.keys())[0]
            result = resolve_tools([ctx_name], base_dir=str(tmp_path))
            assert len(result) == 1
            assert "Placeholder" in result[0].description

    def test_resolve_load_skill_tool(self, tmp_path):
        result = resolve_tools(["load_skill"], base_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "load_skill"

    def test_resolve_agent_tool_explore(self, tmp_path):
        """Explore tool requires config — mock the config loader."""
        config = MagicMock()
        with patch("dokumen.tool_resolver._get_agent_tool_config") as mock_config:
            mock_config.return_value = (config, str(tmp_path))
            result = resolve_tools(["explore"], base_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "explore"

    def test_resolve_code_tools_with_config(self, tmp_path):
        code_repos = [{"name": "test-repo", "base_dir": str(tmp_path), "include_patterns": [], "exclude_patterns": []}]
        result = resolve_tools(["code_read_file"], base_dir=str(tmp_path), code_repos_config=code_repos)
        assert len(result) == 1
        assert result[0].name == "code_read_file"

    def test_resolve_code_graph_tools_with_config(self, tmp_path):
        code_repos = [{"name": "test-repo", "base_dir": str(tmp_path), "include_patterns": [], "exclude_patterns": []}]
        result = resolve_tools(["code_graph_find"], base_dir=str(tmp_path), code_repos_config=code_repos)
        assert len(result) == 1
        assert result[0].name == "code_graph_find"

    def test_shell_command_with_tools_config(self, tmp_path):
        """Shell command uses per-tool timeout from config."""
        tools_config = SimpleNamespace(
            config=SimpleNamespace(
                run_shell_command=SimpleNamespace(timeout=45.0),
                web_fetch=SimpleNamespace(timeout=30.0),
            ),
            allowed=None, defaults=None, blocked=None,
        )
        result = resolve_tools(["run_shell_command"], base_dir=str(tmp_path), tools_config=tools_config)
        assert len(result) == 1
        assert result[0].name == "run_shell_command"

    def test_web_fetch_with_tools_config(self, tmp_path):
        """Web fetch uses per-tool timeout from config."""
        tools_config = SimpleNamespace(
            config=SimpleNamespace(
                web_fetch=SimpleNamespace(timeout=60.0),
                run_shell_command=SimpleNamespace(timeout=30.0),
            ),
            allowed=None, defaults=None, blocked=None,
        )
        result = resolve_tools(["web_fetch"], base_dir=str(tmp_path), tools_config=tools_config)
        assert len(result) == 1
        assert result[0].name == "web_fetch"


class TestResolveToolsWebSearch:
    """Tests for web_search and anthropic_web_search resolution."""

    @patch("dokumen.tools_object.create_perplexity_web_search_tool")
    def test_resolve_web_search_default_config(self, mock_create, tmp_path):
        mock_create.return_value = MagicMock(name="web_search")
        result = resolve_tools(["web_search"], base_dir=str(tmp_path))
        assert len(result) == 1
        mock_create.assert_called_once_with(api_key=None, model="sonar", max_searches=5)

    @patch("dokumen.tools_object.create_perplexity_web_search_tool")
    def test_resolve_web_search_with_tools_config(self, mock_create, tmp_path):
        mock_create.return_value = MagicMock(name="web_search")
        tools_config = SimpleNamespace(
            config=SimpleNamespace(
                web_search=SimpleNamespace(model="sonar-pro", max_searches=10),
                run_shell_command=SimpleNamespace(timeout=30.0),
                web_fetch=SimpleNamespace(timeout=30.0),
            ),
            allowed=None, defaults=None, blocked=None,
        )
        result = resolve_tools(["web_search"], base_dir=str(tmp_path), tools_config=tools_config)
        assert len(result) == 1
        mock_create.assert_called_once_with(api_key=None, model="sonar-pro", max_searches=10)

    @patch("dokumen.tools_object.create_anthropic_web_search_tool")
    def test_resolve_anthropic_web_search(self, mock_create, tmp_path):
        mock_create.return_value = MagicMock(name="anthropic_web_search")
        result = resolve_tools(["anthropic_web_search"], base_dir=str(tmp_path))
        assert len(result) == 1
        mock_create.assert_called_once_with(max_uses=None, allowed_domains=None, blocked_domains=None)

    @patch("dokumen.tools_object.create_anthropic_web_search_tool")
    def test_resolve_anthropic_web_search_with_config(self, mock_create, tmp_path):
        mock_create.return_value = MagicMock(name="anthropic_web_search")
        tools_config = SimpleNamespace(
            config=SimpleNamespace(
                anthropic_web_search=SimpleNamespace(
                    max_uses=5,
                    allowed_domains=["example.com"],
                    blocked_domains=["spam.com"],
                ),
                run_shell_command=SimpleNamespace(timeout=30.0),
                web_fetch=SimpleNamespace(timeout=30.0),
            ),
            allowed=None, defaults=None, blocked=None,
        )
        result = resolve_tools(["anthropic_web_search"], base_dir=str(tmp_path), tools_config=tools_config)
        assert len(result) == 1
        mock_create.assert_called_once_with(
            max_uses=5, allowed_domains=["example.com"], blocked_domains=["spam.com"]
        )


class TestResolveToolsDelegate:
    """Tests for delegate_to_agent resolution."""

    def test_resolve_delegate_with_registry(self, tmp_path):
        registry = MagicMock()
        provider = MagicMock()
        with patch("dokumen.tools_object.create_delegate_to_agent_tool") as mock_create:
            mock_create.return_value = MagicMock(name="delegate_to_agent")
            result = resolve_tools(
                ["delegate_to_agent"],
                base_dir=str(tmp_path),
                agent_registry=registry,
                agent_provider=provider,
            )
        assert len(result) == 1
        mock_create.assert_called_once()


class TestFilterToolsProvenance:
    """Test provenance tracking in filter functions."""

    def test_override_tracks_removed_tools(self):
        prov = ToolProvenance(executor_tools={"a": "scaffold", "b": "scaffold", "c": "scaffold"})
        overrides = MagicMock()

        with patch("dokumen.tool_resolver.is_tool_enabled_for_test") as mock_enabled:
            mock_enabled.side_effect = lambda name, ov: name != "b"
            result = filter_tools_with_overrides(
                ["a", "b", "c"],
                auto_injected_tools=set(),
                overrides=overrides,
                tools_config=None,
                scaffold_name="test",
                scaffold_agent=None,
                provenance=prov,
            )
        assert result == ["a", "c"]
        assert "b" in prov.removed_tools
        assert "b" not in prov.executor_tools


class TestAutoInjectToolsAdditional:
    """Additional auto_inject_tools tests for edge cases."""

    @patch("dokumen.agent_loader.get_agent_tools", return_value=[])
    def test_shell_command_already_present_not_duplicated(self, mock_agent_tools):
        prov = ToolProvenance()
        result, auto = auto_inject_tools(
            executor_tool_names=["run_shell_command", "read_file"],
            is_browser_agent=False,
            is_research_agent=False,
            is_code_agent=False,
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        assert result.count("run_shell_command") == 1

    @patch("dokumen.agent_loader.get_agent_tools", return_value=[])
    def test_web_search_already_present_not_duplicated(self, mock_agent_tools):
        prov = ToolProvenance()
        result, auto = auto_inject_tools(
            executor_tool_names=["web_search"],
            is_browser_agent=False,
            is_research_agent=True,
            is_code_agent=False,
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        assert result.count("web_search") == 1

    @patch("dokumen.agent_loader.get_agent_tools", return_value=[])
    def test_shell_skipped_if_not_in_allowed_list(self, mock_agent_tools):
        prov = ToolProvenance()
        tools_config = SimpleNamespace(allowed=["read_file"], defaults=None, blocked=None)
        result, auto = auto_inject_tools(
            executor_tool_names=["read_file"],
            is_browser_agent=False,
            is_research_agent=False,
            is_code_agent=False,
            tools_config=tools_config,
            scaffold_name="test",
            provenance=prov,
        )
        assert "run_shell_command" not in result

    @patch("dokumen.agent_loader.get_agent_tools", return_value=[])
    def test_web_search_skipped_if_not_in_allowed_list(self, mock_agent_tools):
        prov = ToolProvenance()
        tools_config = SimpleNamespace(allowed=["read_file"], defaults=None, blocked=None)
        result, auto = auto_inject_tools(
            executor_tool_names=["read_file"],
            is_browser_agent=False,
            is_research_agent=True,
            is_code_agent=False,
            tools_config=tools_config,
            scaffold_name="test",
            provenance=prov,
        )
        assert "web_search" not in result

    @patch("dokumen.agent_loader.get_agent_tools", return_value=[])
    def test_browser_tools_injected(self, mock_agent_tools):
        prov = ToolProvenance()
        result, auto = auto_inject_tools(
            executor_tool_names=["read_file"],
            is_browser_agent=True,
            is_research_agent=False,
            is_code_agent=False,
            tools_config=None,
            scaffold_name="test",
            provenance=prov,
        )
        # Browser tools should be injected
        assert any("browser" in t or t == "read_file" for t in result)
        # Verify provenance tracking
        browser_prov = {k: v for k, v in prov.executor_tools.items() if v == "auto:browser"}
        assert len(browser_prov) > 0


class TestFilterJudgeToolsAdditional:
    """Additional filter_judge_tools tests."""

    def test_override_removes_judge_tools(self):
        overrides = MagicMock()
        judge_prov = {"read_file": "scaffold", "web_fetch": "scaffold"}

        with patch("dokumen.tool_resolver.is_tool_enabled_for_test") as mock_enabled:
            mock_enabled.side_effect = lambda name, ov: name != "web_fetch"
            result = filter_judge_tools(
                ["read_file", "web_fetch"],
                auto_added_judge_tools=set(),
                overrides=overrides,
                tools_config=None,
                scaffold_name="test",
                judge_name="accuracy",
                scaffold_agent=None,
                judge_prov=judge_prov,
            )
        assert "web_fetch" not in result
        assert "web_fetch" not in judge_prov

    def test_override_raises_for_auto_added_disabled(self):
        overrides = MagicMock()
        judge_prov = {"run_shell_command": "auto:standard"}

        with patch("dokumen.tool_resolver.is_tool_enabled_for_test", return_value=False):
            with pytest.raises(ValueError, match="auto-required"):
                filter_judge_tools(
                    ["run_shell_command"],
                    auto_added_judge_tools={"run_shell_command"},
                    overrides=overrides,
                    tools_config=None,
                    scaffold_name="test",
                    judge_name="accuracy",
                    scaffold_agent="standard",
                    judge_prov=judge_prov,
                )

    def test_no_overrides_no_blocked_passes_through(self):
        judge_prov = {"read_file": "scaffold"}
        result = filter_judge_tools(
            ["read_file"],
            auto_added_judge_tools=set(),
            overrides=None,
            tools_config=None,
            scaffold_name="test",
            judge_name="accuracy",
            scaffold_agent=None,
            judge_prov=judge_prov,
        )
        assert result == ["read_file"]


class TestCreatePlaceholderTool:
    """Tests for _create_placeholder_tool."""

    def test_creates_tool_with_name(self):
        tool = _create_placeholder_tool("test_tool")
        assert tool.name == "test_tool"
        assert "Placeholder" in tool.description

    @pytest.mark.asyncio
    async def test_handler_returns_error(self):
        tool = _create_placeholder_tool("test_tool")
        result = await tool.handler({})
        assert result.success is False
        assert "not available" in result.error
