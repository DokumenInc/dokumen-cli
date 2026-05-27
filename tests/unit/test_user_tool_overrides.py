"""
Unit tests for CLI user_tool_overrides module.

TDD: Tests written first, before implementation.
Covers: YAML loading, validation, canonical-to-CLI mapping,
explore safety, and integration with loader blocked/override logic.
"""
import os
import textwrap
from pathlib import Path

import pytest
import yaml


class TestCanonicalToCliMapping:
    """Tests for canonical <-> CLI name mapping."""

    def test_canonical_to_cli_mapping_glob_files(self):
        """glob_files canonical name maps to CLI 'glob'."""
        from dokumen.user_tool_overrides import CANONICAL_TO_CLI
        assert CANONICAL_TO_CLI["glob_files"] == "glob"

    def test_canonical_to_cli_mapping_list_files(self):
        """list_files canonical name maps to CLI 'list_directory'."""
        from dokumen.user_tool_overrides import CANONICAL_TO_CLI
        assert CANONICAL_TO_CLI["list_files"] == "list_directory"

    def test_canonical_to_cli_mapping_search_files(self):
        """search_files canonical name maps to CLI 'search_file_content'."""
        from dokumen.user_tool_overrides import CANONICAL_TO_CLI
        assert CANONICAL_TO_CLI["search_files"] == "search_file_content"

    def test_canonical_to_cli_mapping_code_list(self):
        """code_list canonical name maps to CLI 'code_list_directory'."""
        from dokumen.user_tool_overrides import CANONICAL_TO_CLI
        assert CANONICAL_TO_CLI["code_list"] == "code_list_directory"

    def test_cli_to_canonical_reverse_mapping(self):
        """CLI names reverse-map correctly to canonical names."""
        from dokumen.user_tool_overrides import CLI_TO_CANONICAL
        assert CLI_TO_CANONICAL["glob"] == "glob_files"
        assert CLI_TO_CANONICAL["list_directory"] == "list_files"
        assert CLI_TO_CANONICAL["search_file_content"] == "search_files"
        assert CLI_TO_CANONICAL["code_list_directory"] == "code_list"

    def test_mappings_are_inverses(self):
        """CANONICAL_TO_CLI and CLI_TO_CANONICAL are exact inverses."""
        from dokumen.user_tool_overrides import CANONICAL_TO_CLI, CLI_TO_CANONICAL
        for canonical, cli in CANONICAL_TO_CLI.items():
            assert CLI_TO_CANONICAL[cli] == canonical
        for cli, canonical in CLI_TO_CANONICAL.items():
            assert CANONICAL_TO_CLI[canonical] == cli


class TestValidateToolOverrides:
    """Tests for validate_tool_overrides()."""

    def test_empty_input_returns_empty(self):
        """Empty raw dict returns empty overrides and no errors."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        result = validate_tool_overrides({})
        assert result.overrides == {}
        assert result.errors == []

    def test_valid_tool_with_valid_systems(self):
        """Valid tool with valid systems produces correct override."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        result = validate_tool_overrides({"read_file": ["chat", "test", "explore"]})
        assert "read_file" in result.overrides
        assert result.overrides["read_file"] == frozenset({"chat", "test", "explore"})
        assert result.errors == []

    def test_empty_available_in_disables_tool(self):
        """Empty available_in list disables the tool (frozenset())."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        result = validate_tool_overrides({"read_file": []})
        assert "read_file" in result.overrides
        assert result.overrides["read_file"] == frozenset()
        assert result.errors == []

    def test_unknown_tool_produces_error(self):
        """Unknown tool name produces an error but doesn't crash."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        result = validate_tool_overrides({"totally_fake_tool": ["chat"]})
        assert "totally_fake_tool" not in result.overrides
        assert any("Unknown tool" in e for e in result.errors)

    def test_invalid_system_produces_error(self):
        """Invalid system value produces an error."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        result = validate_tool_overrides({"read_file": ["chat", "mcp"]})
        # 'chat' is valid, 'mcp' is not
        assert "read_file" in result.overrides
        assert "chat" in result.overrides["read_file"]
        assert "mcp" not in result.overrides["read_file"]
        assert any("mcp" in e for e in result.errors)

    def test_system_not_implementable_produces_error(self):
        """System without runtime implementation produces an error."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        # edit_file is only implementable in chat, not test
        result = validate_tool_overrides({"edit_file": ["test"]})
        assert "edit_file" in result.overrides
        assert "test" not in result.overrides["edit_file"]
        assert any("No test implementation" in e for e in result.errors)

    def test_explore_unsafe_tool_produces_error(self):
        """Non-read-only tool in explore produces an error."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        # run_shell_command is implementable in chat+test but NOT explore,
        # so the runtime-compatibility check fires before explore safety
        result = validate_tool_overrides({"run_shell_command": ["explore"]})
        assert "run_shell_command" in result.overrides
        assert "explore" not in result.overrides["run_shell_command"]
        assert any("explore" in e and "run_shell_command" in e for e in result.errors)

    def test_multiple_tools_mixed_validity(self):
        """Multiple tools: valid ones kept, invalid ones skipped with errors."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        result = validate_tool_overrides({
            "read_file": ["chat", "test"],
            "fake_tool": ["chat"],
            "glob_files": ["explore"],
        })
        assert "read_file" in result.overrides
        assert "glob_files" in result.overrides
        assert "fake_tool" not in result.overrides
        assert len(result.errors) == 1  # Only fake_tool error


class TestLoadOverridesFromDir:
    """Tests for load_overrides_from_dir()."""

    def test_missing_dir_returns_none(self, tmp_path):
        """No .dokumen/tool-definitions dir returns None (legacy mode)."""
        from dokumen.user_tool_overrides import load_overrides_from_dir
        result = load_overrides_from_dir(str(tmp_path))
        assert result is None

    def test_empty_dir_returns_empty_result(self, tmp_path):
        """Empty .dokumen/tool-definitions dir returns empty result."""
        from dokumen.user_tool_overrides import load_overrides_from_dir
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)
        result = load_overrides_from_dir(str(tmp_path))
        assert result is not None
        assert result.overrides == {}
        assert result.errors == []

    def test_load_valid_yaml_files(self, tmp_path):
        """Valid YAML files are parsed correctly."""
        from dokumen.user_tool_overrides import load_overrides_from_dir
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)

        # Create a valid override file
        (overrides_dir / "read_file.yaml").write_text(yaml.dump({
            "name": "read_file",
            "available_in": ["chat", "test"],
        }))
        (overrides_dir / "glob_files.yaml").write_text(yaml.dump({
            "name": "glob_files",
            "available_in": ["chat", "test", "explore"],
        }))

        result = load_overrides_from_dir(str(tmp_path))
        assert result is not None
        assert "read_file" in result.overrides
        assert result.overrides["read_file"] == frozenset({"chat", "test"})
        assert "glob_files" in result.overrides
        assert result.overrides["glob_files"] == frozenset({"chat", "test", "explore"})

    def test_invalid_yaml_produces_parse_error(self, tmp_path):
        """Invalid YAML file produces a parse error but doesn't crash."""
        from dokumen.user_tool_overrides import load_overrides_from_dir
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)

        # Create invalid YAML
        (overrides_dir / "bad.yaml").write_text(":::not valid yaml [[[")
        # Create valid YAML alongside
        (overrides_dir / "read_file.yaml").write_text(yaml.dump({
            "name": "read_file",
            "available_in": ["chat"],
        }))

        result = load_overrides_from_dir(str(tmp_path))
        assert result is not None
        assert "read_file" in result.overrides
        # bad.yaml should have produced an error
        assert any("bad.yaml" in e for e in result.errors)

    def test_yaml_not_a_mapping_produces_error(self, tmp_path):
        """YAML that parses to non-dict produces an error."""
        from dokumen.user_tool_overrides import load_overrides_from_dir
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)

        (overrides_dir / "list.yaml").write_text("- item1\n- item2\n")

        result = load_overrides_from_dir(str(tmp_path))
        assert result is not None
        assert any("not a mapping" in e for e in result.errors)

    def test_available_in_not_a_list_produces_error(self, tmp_path):
        """available_in as non-list produces an error."""
        from dokumen.user_tool_overrides import load_overrides_from_dir
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)

        (overrides_dir / "read_file.yaml").write_text(yaml.dump({
            "name": "read_file",
            "available_in": "chat",  # string, not list
        }))

        result = load_overrides_from_dir(str(tmp_path))
        assert result is not None
        assert any("available_in" in e for e in result.errors)

    def test_name_fallback_to_stem(self, tmp_path):
        """If 'name' key missing in YAML, falls back to file stem."""
        from dokumen.user_tool_overrides import load_overrides_from_dir
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)

        (overrides_dir / "read_file.yaml").write_text(yaml.dump({
            "available_in": ["chat", "test"],
        }))

        result = load_overrides_from_dir(str(tmp_path))
        assert result is not None
        assert "read_file" in result.overrides

    def test_non_yaml_files_ignored(self, tmp_path):
        """Non-.yaml files in the directory are ignored."""
        from dokumen.user_tool_overrides import load_overrides_from_dir
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)

        (overrides_dir / "readme.md").write_text("# Tool definitions")
        (overrides_dir / "read_file.yaml").write_text(yaml.dump({
            "name": "read_file",
            "available_in": ["chat"],
        }))

        result = load_overrides_from_dir(str(tmp_path))
        assert result is not None
        assert len(result.overrides) == 1
        assert "read_file" in result.overrides


class TestIsToolEnabledForTest:
    """Tests for is_tool_enabled_for_test()."""

    def test_legacy_mode_all_enabled(self):
        """When overrides is None (legacy), all tools are enabled."""
        from dokumen.user_tool_overrides import is_tool_enabled_for_test
        assert is_tool_enabled_for_test("read_file", None) is True
        assert is_tool_enabled_for_test("glob", None) is True
        assert is_tool_enabled_for_test("run_shell_command", None) is True

    def test_tool_enabled_for_test_in_overrides(self):
        """Tool with 'test' in overrides is enabled."""
        from dokumen.user_tool_overrides import (
            is_tool_enabled_for_test, ToolOverridesResult,
        )
        overrides = ToolOverridesResult(
            overrides={"read_file": frozenset({"chat", "test"})},
        )
        assert is_tool_enabled_for_test("read_file", overrides) is True

    def test_tool_disabled_for_test_in_overrides(self):
        """Tool without 'test' in overrides is disabled."""
        from dokumen.user_tool_overrides import (
            is_tool_enabled_for_test, ToolOverridesResult,
        )
        overrides = ToolOverridesResult(
            overrides={"read_file": frozenset({"chat"})},  # no 'test'
        )
        assert is_tool_enabled_for_test("read_file", overrides) is False

    def test_tool_completely_disabled(self):
        """Tool with empty frozenset is disabled for all systems."""
        from dokumen.user_tool_overrides import (
            is_tool_enabled_for_test, ToolOverridesResult,
        )
        overrides = ToolOverridesResult(
            overrides={"read_file": frozenset()},
        )
        assert is_tool_enabled_for_test("read_file", overrides) is False

    def test_cli_name_maps_to_canonical(self):
        """CLI name 'glob' maps to canonical 'glob_files' for lookup."""
        from dokumen.user_tool_overrides import (
            is_tool_enabled_for_test, ToolOverridesResult,
        )
        overrides = ToolOverridesResult(
            overrides={"glob_files": frozenset({"test"})},
        )
        assert is_tool_enabled_for_test("glob", overrides) is True

    def test_cli_name_list_directory_maps_to_list_files(self):
        """CLI name 'list_directory' maps to canonical 'list_files'."""
        from dokumen.user_tool_overrides import (
            is_tool_enabled_for_test, ToolOverridesResult,
        )
        overrides = ToolOverridesResult(
            overrides={"list_files": frozenset({"chat"})},  # no test
        )
        assert is_tool_enabled_for_test("list_directory", overrides) is False

    def test_tool_not_in_overrides_enabled_by_default(self):
        """Tool not present in overrides map is enabled by default."""
        from dokumen.user_tool_overrides import (
            is_tool_enabled_for_test, ToolOverridesResult,
        )
        overrides = ToolOverridesResult(
            overrides={"read_file": frozenset({"test"})},
        )
        # run_shell_command not in overrides -> enabled
        assert is_tool_enabled_for_test("run_shell_command", overrides) is True


class TestGetEffectiveToolsForSystem:
    """Tests for get_effective_tools_for_system()."""

    def test_returns_tools_for_system(self):
        """Returns only tools whose overrides include the requested system."""
        from dokumen.user_tool_overrides import (
            get_effective_tools_for_system, ToolOverridesResult,
        )
        overrides = ToolOverridesResult(
            overrides={
                "read_file": frozenset({"chat", "test", "explore"}),
                "write_file": frozenset({"chat"}),
                "glob_files": frozenset({"test"}),
            },
        )
        test_tools = get_effective_tools_for_system("test", overrides)
        assert "read_file" in test_tools
        assert "glob_files" in test_tools
        assert "write_file" not in test_tools

    def test_explore_system_returns_explore_tools(self):
        """Explore system returns only tools with 'explore' in overrides."""
        from dokumen.user_tool_overrides import (
            get_effective_tools_for_system, ToolOverridesResult,
        )
        overrides = ToolOverridesResult(
            overrides={
                "read_file": frozenset({"explore"}),
                "glob_files": frozenset({"explore"}),
                "run_shell_command": frozenset({"test"}),
            },
        )
        explore_tools = get_effective_tools_for_system("explore", overrides)
        assert "read_file" in explore_tools
        assert "glob_files" in explore_tools
        assert "run_shell_command" not in explore_tools

    def test_empty_overrides_returns_defaults(self):
        """Empty overrides result falls back to IMPLEMENTABLE_IN defaults."""
        from dokumen.user_tool_overrides import (
            get_effective_tools_for_system, ToolOverridesResult, IMPLEMENTABLE_IN,
        )
        overrides = ToolOverridesResult(overrides={})
        result = get_effective_tools_for_system("test", overrides)
        # Should return all tools with "test" in their IMPLEMENTABLE_IN default
        expected = {name for name, systems in IMPLEMENTABLE_IN.items() if "test" in systems}
        assert result == expected
        assert len(result) > 0  # Sanity: should include at least read_file etc.


class TestExploreSafety:
    """Tests for explore safety enforcement."""

    def test_explore_safe_tools_contains_read_only(self):
        """EXPLORE_SAFE_TOOLS only contains read-only canonical tools."""
        from dokumen.user_tool_overrides import EXPLORE_SAFE_TOOLS
        # These should be in EXPLORE_SAFE_TOOLS
        assert "read_file" in EXPLORE_SAFE_TOOLS
        assert "list_files" in EXPLORE_SAFE_TOOLS
        assert "search_files" in EXPLORE_SAFE_TOOLS
        assert "glob_files" in EXPLORE_SAFE_TOOLS
        assert "code_read_file" in EXPLORE_SAFE_TOOLS

    def test_explore_safe_tools_excludes_write_tools(self):
        """EXPLORE_SAFE_TOOLS does not contain write/execute tools."""
        from dokumen.user_tool_overrides import EXPLORE_SAFE_TOOLS
        assert "write_file" not in EXPLORE_SAFE_TOOLS
        assert "run_shell_command" not in EXPLORE_SAFE_TOOLS
        assert "delete_file" not in EXPLORE_SAFE_TOOLS

    def test_validate_rejects_unsafe_tool_for_explore(self):
        """Validation rejects non-read-only tool for explore system."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        # run_shell_command is in IMPLEMENTABLE_IN for chat+test but NOT explore,
        # so the runtime-compatibility check fires
        result = validate_tool_overrides({"run_shell_command": ["explore"]})
        assert "explore" not in result.overrides.get("run_shell_command", frozenset())
        assert any("explore" in e and "run_shell_command" in e for e in result.errors)

    def test_explore_safety_check_for_implementable_tool(self):
        """Tool implementable in explore but not in EXPLORE_SAFE_TOOLS is rejected."""
        from dokumen.user_tool_overrides import (
            validate_tool_overrides, IMPLEMENTABLE_IN, EXPLORE_SAFE_TOOLS,
        )
        # Find a tool that IS implementable in explore but NOT in EXPLORE_SAFE_TOOLS
        # (If no such tool exists, the safety check is redundant but still valid)
        unsafe_explore_tools = [
            t for t, systems in IMPLEMENTABLE_IN.items()
            if "explore" in systems and t not in EXPLORE_SAFE_TOOLS
        ]
        if unsafe_explore_tools:
            tool = unsafe_explore_tools[0]
            result = validate_tool_overrides({tool: ["explore"]})
            assert "explore" not in result.overrides.get(tool, frozenset())
            assert any("not safe for explore" in e for e in result.errors)


class TestImplementableIn:
    """Tests for the IMPLEMENTABLE_IN mapping."""

    def test_read_file_in_all_systems(self):
        """read_file is implementable in chat, test, and explore."""
        from dokumen.user_tool_overrides import IMPLEMENTABLE_IN
        assert IMPLEMENTABLE_IN["read_file"] == frozenset({"chat", "test", "explore"})

    def test_write_file_chat_and_test(self):
        """write_file is implementable in chat and test."""
        from dokumen.user_tool_overrides import IMPLEMENTABLE_IN
        assert IMPLEMENTABLE_IN["write_file"] == frozenset({"chat", "test"})

    def test_write_file_enabled_for_test_via_override(self):
        """write_file override with 'test' is accepted (not rejected as unimplementable)."""
        from dokumen.user_tool_overrides import validate_tool_overrides
        result = validate_tool_overrides({"write_file": ["chat", "test"]})
        assert "write_file" in result.overrides
        assert result.overrides["write_file"] == frozenset({"chat", "test"})
        assert result.errors == []

    def test_run_shell_command_in_chat_and_test(self):
        """run_shell_command is implementable in chat and test."""
        from dokumen.user_tool_overrides import IMPLEMENTABLE_IN
        assert IMPLEMENTABLE_IN["run_shell_command"] == frozenset({"chat", "test"})

    def test_browser_tools_test_only(self):
        """Browser tools are only implementable in test."""
        from dokumen.user_tool_overrides import IMPLEMENTABLE_IN
        assert IMPLEMENTABLE_IN["browser_navigate"] == frozenset({"test"})
        assert IMPLEMENTABLE_IN["browser_click"] == frozenset({"test"})

    def test_all_tools_have_at_least_one_system(self):
        """Every tool in IMPLEMENTABLE_IN has at least one system."""
        from dokumen.user_tool_overrides import IMPLEMENTABLE_IN
        for tool_name, systems in IMPLEMENTABLE_IN.items():
            assert len(systems) > 0, f"{tool_name} has no systems"


class TestLoaderOverridesIntegration:
    """Tests for loader.py integration with tool overrides."""

    def _create_scaffold_project(self, tmp_path):
        """Helper: create a minimal project with scaffold and prompts."""
        # Create dokumen.yaml
        (tmp_path / "dokumen.yaml").write_text(yaml.dump({
            "version": "1.0",
            "provider": {"name": "mock"},
        }))

        # Create the prompts directory and file
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "documentation-validation.txt").write_text(
            "You are validating documentation."
        )
        (prompts_dir / "judges").mkdir()
        (prompts_dir / "judges" / "default.txt").write_text(
            "Judge the output."
        )

        # Create a minimal scaffold
        scaffold = {
            "name": "test-override",
            "reason": "Test overrides integration",
            "files": [{"path": "docs/test.md"}],
            "executor": {
                "system_prompt": "@prompts/documentation-validation.txt",
                "user_prompt": "Test the docs",
                "tools": ["read_file", "glob"],
            },
            "judges": [
                {
                    "name": "accuracy",
                    "system_prompt": "Judge the output.\nReturn JSON: {\"verdict\": \"PASS\", \"confidence\": 0.9, \"reason\": \"ok\"}",
                }
            ],
        }
        scaffold_path = tmp_path / "tests" / "test-override.test.yaml"
        scaffold_path.parent.mkdir(parents=True, exist_ok=True)
        scaffold_path.write_text(yaml.dump(scaffold))
        return scaffold_path

    def test_overrides_replace_blocked(self, tmp_path):
        """When overrides exist, tools.blocked is ignored."""
        from dokumen.user_tool_overrides import ToolOverridesResult
        from unittest.mock import patch

        scaffold_path = self._create_scaffold_project(tmp_path)

        # Create overrides dir with read_file disabled for test
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)
        (overrides_dir / "read_file.yaml").write_text(yaml.dump({
            "name": "read_file",
            "available_in": ["chat"],  # NOT test
        }))

        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )
        # read_file should NOT be in executor tools (disabled for test)
        executor_tool_names = [t.name for t in test_obj.executor.tools]
        assert "read_file" not in executor_tool_names

    def test_disabled_auto_injected_tool_raises(self, tmp_path):
        """Disabling an auto-injected tool raises ValueError."""
        scaffold_path = self._create_scaffold_project(tmp_path)

        # Create overrides dir with run_shell_command disabled
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)
        (overrides_dir / "run_shell_command.yaml").write_text(yaml.dump({
            "name": "run_shell_command",
            "available_in": ["chat"],  # NOT test -> auto-injected but disabled
        }))

        from dokumen.loader import load_scaffold
        with pytest.raises(ValueError, match="required for"):
            load_scaffold(
                str(scaffold_path),
                project_root=str(tmp_path),
            )

    def test_legacy_mode_no_overrides_dir(self, tmp_path):
        """Without .dokumen/tool-definitions/, legacy blocked mode applies."""
        scaffold_path = self._create_scaffold_project(tmp_path)

        # No overrides dir — legacy mode
        from dokumen.loader import load_scaffold
        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )
        # All tools should be present (SDK maps run_shell_command->Bash, read_file->Read)
        executor_tool_names = [t.name for t in test_obj.executor.tools]
        assert "Bash" in executor_tool_names or "run_shell_command" in executor_tool_names
        assert "Read" in executor_tool_names or "read_file" in executor_tool_names

    def test_disabled_auto_required_judge_tool_raises(self, tmp_path):
        """Disabling auto-added run_shell_command for judges raises ValueError."""
        scaffold_path = self._create_scaffold_project(tmp_path)

        # Create overrides dir with run_shell_command disabled
        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)
        (overrides_dir / "run_shell_command.yaml").write_text(yaml.dump({
            "name": "run_shell_command",
            "available_in": ["chat"],  # NOT test
        }))

        from dokumen.loader import load_scaffold
        with pytest.raises(ValueError, match="required|auto-required"):
            load_scaffold(
                str(scaffold_path),
                project_root=str(tmp_path),
            )

    def test_unknown_tool_in_overrides_logged_not_crash(self, tmp_path):
        """Unknown tools in overrides produce errors but don't crash loading."""
        scaffold_path = self._create_scaffold_project(tmp_path)

        overrides_dir = tmp_path / ".dokumen" / "tool-definitions"
        overrides_dir.mkdir(parents=True)
        (overrides_dir / "fake_tool.yaml").write_text(yaml.dump({
            "name": "fake_tool",
            "available_in": ["test"],
        }))

        from dokumen.loader import load_scaffold
        # Should not crash — unknown tools are just logged as errors
        test_obj = load_scaffold(
            str(scaffold_path),
            project_root=str(tmp_path),
        )
        executor_tool_names = [t.name for t in test_obj.executor.tools]
        assert "Bash" in executor_tool_names or "run_shell_command" in executor_tool_names


class TestExploreOverridesIntegration:
    """Tests for explore agent integration with tool overrides."""

    def test_explore_overrides_filters_tools(self):
        """get_effective_tools_for_system('explore') returns only explore-enabled tools."""
        from dokumen.user_tool_overrides import (
            get_effective_tools_for_system, ToolOverridesResult,
            EXPLORE_SAFE_TOOLS, CANONICAL_TO_CLI,
        )
        overrides = ToolOverridesResult(
            overrides={
                "read_file": frozenset({"explore"}),
                "glob_files": frozenset({"explore"}),
                "run_shell_command": frozenset({"test"}),  # NOT explore
            },
        )
        explore_tools = get_effective_tools_for_system("explore", overrides)
        safe_tools = explore_tools & EXPLORE_SAFE_TOOLS
        cli_names = [CANONICAL_TO_CLI.get(n, n) for n in safe_tools]

        assert "read_file" in cli_names
        assert "glob" in cli_names
        assert "run_shell_command" not in cli_names

    def test_explore_canonical_to_cli_maps_correctly(self):
        """Explore tools use CLI names after canonical mapping."""
        from dokumen.user_tool_overrides import (
            get_effective_tools_for_system, ToolOverridesResult,
            EXPLORE_SAFE_TOOLS, CANONICAL_TO_CLI,
        )
        overrides = ToolOverridesResult(
            overrides={
                "list_files": frozenset({"explore"}),
                "search_files": frozenset({"explore"}),
            },
        )
        explore_tools = get_effective_tools_for_system("explore", overrides)
        safe_tools = explore_tools & EXPLORE_SAFE_TOOLS
        cli_names = [CANONICAL_TO_CLI.get(n, n) for n in safe_tools]

        # list_files -> list_directory, search_files -> search_file_content
        assert "list_directory" in cli_names
        assert "search_file_content" in cli_names

    def test_explore_overridden_tool_excluded_defaults_kept(self):
        """Overridden tool loses explore, but un-overridden tools keep defaults."""
        from dokumen.user_tool_overrides import (
            get_effective_tools_for_system, ToolOverridesResult,
            EXPLORE_SAFE_TOOLS, IMPLEMENTABLE_IN,
        )
        overrides = ToolOverridesResult(
            overrides={
                "read_file": frozenset({"test"}),  # NOT explore — overridden
            },
        )
        explore_tools = get_effective_tools_for_system("explore", overrides)
        safe_tools = explore_tools & EXPLORE_SAFE_TOOLS
        # read_file is overridden to test-only, so it should NOT be in explore
        assert "read_file" not in safe_tools
        # Other explore-safe tools with "explore" in IMPLEMENTABLE_IN should still be present
        default_explore = {
            name for name, systems in IMPLEMENTABLE_IN.items()
            if "explore" in systems
        } & EXPLORE_SAFE_TOOLS
        expected = default_explore - {"read_file"}
        assert safe_tools == expected
