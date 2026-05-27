"""Tests for loader module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile


class TestResolvePromptReference:
    """Tests for resolve_prompt_reference function."""

    def test_literal_prompt_returned(self, tmp_path):
        """Literal prompts are returned as-is."""
        # resolve_prompt_reference removed in staging merge

        result = resolve_prompt_reference("You are a tester.", str(tmp_path))

        assert result == "You are a tester."

    def test_empty_prompt(self, tmp_path):
        """Empty prompts return empty string."""
        # resolve_prompt_reference removed in staging merge

        result = resolve_prompt_reference("", str(tmp_path))

        assert result == ""

    def test_none_prompt(self, tmp_path):
        """None prompts return empty string."""
        # resolve_prompt_reference removed in staging merge

        result = resolve_prompt_reference(None, str(tmp_path))

        assert result == ""

    def test_reference_loads_file(self, tmp_path):
        """@path/file.txt loads file content."""
        # resolve_prompt_reference removed in staging merge

        prompt_file = tmp_path / "prompts" / "test.txt"
        prompt_file.parent.mkdir()
        prompt_file.write_text("File content here")

        result = resolve_prompt_reference("@prompts/test.txt", str(tmp_path))

        assert result == "File content here"

    def test_reference_with_inline_append(self, tmp_path):
        """Reference with inline text appends content."""
        # resolve_prompt_reference removed in staging merge

        prompt_file = tmp_path / "base.txt"
        prompt_file.write_text("Base content")

        result = resolve_prompt_reference("@base.txt\nAdditional text", str(tmp_path))

        assert "Base content" in result
        assert "Additional text" in result

    def test_reference_file_not_found(self, tmp_path):
        """Missing reference file raises FileNotFoundError."""
        # resolve_prompt_reference removed in staging merge

        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_prompt_reference("@missing/file.txt", str(tmp_path))

    def test_variable_substitution_in_literal(self, tmp_path):
        """Variables are substituted in literal prompts."""
        # resolve_prompt_reference removed in staging merge

        result = resolve_prompt_reference(
            "Working dir: {working_dir}",
            str(tmp_path),
            variables={"working_dir": "/builds/test/project"}
        )

        assert result == "Working dir: /builds/test/project"

    def test_variable_substitution_in_file_reference(self, tmp_path):
        """Variables are substituted in file references."""
        # resolve_prompt_reference removed in staging merge

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Path is {working_dir}/docs")

        result = resolve_prompt_reference(
            "@prompt.txt",
            str(tmp_path),
            variables={"working_dir": "/my/path"}
        )

        assert result == "Path is /my/path/docs"

    def test_multiple_variable_substitutions(self, tmp_path):
        """Multiple variables are substituted."""
        # resolve_prompt_reference removed in staging merge

        result = resolve_prompt_reference(
            "{name} at {working_dir}",
            str(tmp_path),
            variables={"name": "Executor", "working_dir": "/home"}
        )

        assert result == "Executor at /home"

    def test_no_variables_returns_unchanged(self, tmp_path):
        """Prompt with no variables passed returns unchanged."""
        # resolve_prompt_reference removed in staging merge

        result = resolve_prompt_reference(
            "No {variables} here",
            str(tmp_path)
        )

        assert result == "No {variables} here"


class TestResolvePromptReferencePathTraversal:
    """Tests for path traversal protection in resolve_prompt_reference (security)."""

    def test_blocks_dot_dot_traversal(self, tmp_path):
        """Path traversal with .. is blocked."""
        # resolve_prompt_reference removed in staging merge

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            resolve_prompt_reference("@prompts/../../../etc/passwd", str(tmp_path))

    def test_blocks_absolute_path(self, tmp_path):
        """Absolute paths are blocked."""
        # resolve_prompt_reference removed in staging merge

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            resolve_prompt_reference("@/etc/passwd", str(tmp_path))

    def test_blocks_hidden_traversal(self, tmp_path):
        """Hidden traversal attempts are blocked."""
        # resolve_prompt_reference removed in staging merge

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            resolve_prompt_reference("@prompts/foo/../../../etc/passwd", str(tmp_path))

    def test_allows_valid_relative_path(self, tmp_path):
        """Valid relative paths within project are allowed."""
        # resolve_prompt_reference removed in staging merge

        # Create a valid prompt file
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        prompt_file = prompt_dir / "valid.txt"
        prompt_file.write_text("Valid prompt content")

        result = resolve_prompt_reference("@prompts/valid.txt", str(tmp_path))

        assert result == "Valid prompt content"

    def test_allows_nested_relative_path(self, tmp_path):
        """Nested relative paths within project are allowed."""
        # resolve_prompt_reference removed in staging merge

        # Create a nested prompt file
        nested_dir = tmp_path / "prompts" / "executors"
        nested_dir.mkdir(parents=True)
        prompt_file = nested_dir / "test.txt"
        prompt_file.write_text("Nested prompt")

        result = resolve_prompt_reference("@prompts/executors/test.txt", str(tmp_path))

        assert result == "Nested prompt"

    def test_blocks_dot_dot_in_middle(self, tmp_path):
        """.. in the middle of path is blocked."""
        # resolve_prompt_reference removed in staging merge

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            resolve_prompt_reference("@prompts/foo/bar/../../../etc/passwd", str(tmp_path))

    def test_blocks_encoded_traversal(self, tmp_path):
        """URL-encoded traversal attempts are blocked."""
        # resolve_prompt_reference removed in staging merge

        # Even if someone tries %2e%2e (url-encoded ..)
        # The raw string still contains ..
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            resolve_prompt_reference("@prompts/..%2f..%2fetc/passwd", str(tmp_path))


class TestSubstitutePromptVariables:
    """Tests for substitute_prompt_variables function."""

    def test_substitutes_single_variable(self):
        """Substitutes a single variable."""
        from dokumen.loader import substitute_prompt_variables

        result = substitute_prompt_variables(
            "Working in {working_dir}",
            {"working_dir": "/builds/project"}
        )

        assert result == "Working in /builds/project"

    def test_substitutes_multiple_occurrences(self):
        """Substitutes multiple occurrences of same variable."""
        from dokumen.loader import substitute_prompt_variables

        result = substitute_prompt_variables(
            "{path} and {path} again",
            {"path": "/home"}
        )

        assert result == "/home and /home again"

    def test_empty_prompt_returns_empty(self):
        """Empty prompt returns empty string."""
        from dokumen.loader import substitute_prompt_variables

        result = substitute_prompt_variables("", {"var": "value"})

        assert result == ""

    def test_none_prompt_returns_empty(self):
        """None prompt returns empty string."""
        from dokumen.loader import substitute_prompt_variables

        result = substitute_prompt_variables(None, {"var": "value"})

        assert result == ""

    def test_empty_variables_returns_unchanged(self):
        """Empty variables dict returns prompt unchanged."""
        from dokumen.loader import substitute_prompt_variables

        result = substitute_prompt_variables("Keep {this}", {})

        assert result == "Keep {this}"

    def test_none_variables_returns_unchanged(self):
        """None variables returns prompt unchanged."""
        from dokumen.loader import substitute_prompt_variables

        result = substitute_prompt_variables("Keep {this}", None)

        assert result == "Keep {this}"


class TestLoadExecutorPromptVariables:
    """Tests for load_executor_prompt with variable substitution."""

    def test_substitutes_working_dir(self):
        """Substitutes {working_dir} in executor prompts."""
        from dokumen.loader import load_executor_prompt

        result = load_executor_prompt(
            "@prompts/documentation-validation.txt",
            variables={"working_dir": "/builds/test-project"}
        )

        # The prompt should have {working_dir} replaced
        assert "/builds/test-project" in result
        assert "{working_dir}" not in result

    def test_without_variables_keeps_placeholder(self):
        """Without variables, placeholder remains in prompt."""
        from dokumen.loader import load_executor_prompt
        # Clear cache to ensure fresh load
        from dokumen import loader
        loader._executor_prompt_cache.clear()

        result = load_executor_prompt("@prompts/documentation-validation.txt")

        # The raw template should have {working_dir}
        assert "{working_dir}" in result


class TestFindProjectRoot:
    """Tests for find_project_root function."""

    def test_finds_dokumen_yaml(self, tmp_path):
        """Finds project root with dokumen.yaml."""
        from dokumen.loader import find_project_root

        # Create dokumen.yaml at root
        (tmp_path / "dokumen.yaml").write_text("version: 1.0")
        # Create nested directory
        nested = tmp_path / "tests" / "unit"
        nested.mkdir(parents=True)

        result = find_project_root(str(nested))

        assert Path(result) == tmp_path

    def test_returns_fallback_when_not_found(self, tmp_path):
        """Returns fallback when dokumen.yaml not found."""
        from dokumen.loader import find_project_root

        result = find_project_root(str(tmp_path))

        # Should return the path itself as fallback
        assert result is not None

    def test_handles_file_path(self, tmp_path):
        """Handles file path input."""
        from dokumen.loader import find_project_root

        (tmp_path / "dokumen.yaml").write_text("version: 1.0")
        test_file = tmp_path / "tests" / "test.yaml"
        test_file.parent.mkdir()
        test_file.write_text("test: true")

        result = find_project_root(str(test_file))

        assert Path(result) == tmp_path


class TestResolveTools:
    """Tests for resolve_tools function."""

    def test_resolves_read_file(self):
        """Resolves read_file tool."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(["read_file"], base_dir=".")

        assert len(tools) == 1
        assert tools[0].name == "read_file"

    def test_resolves_list_directory(self):
        """Resolves list_directory tool."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(["list_directory"], base_dir=".")

        assert len(tools) == 1
        assert tools[0].name == "list_directory"

    def test_resolves_glob(self):
        """Resolves glob tool."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(["glob"], base_dir=".")

        assert len(tools) == 1
        assert tools[0].name == "glob"

    def test_resolves_multiple_tools(self):
        """Resolves multiple tools."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(["read_file", "list_directory"], base_dir=".")

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "list_directory" in tool_names

    def test_unknown_tool_raises(self):
        """Unknown tools raise ValueError."""
        from dokumen.loader import resolve_tools

        with pytest.raises(ValueError, match="Unknown tool"):
            resolve_tools(["read_file", "unknown_tool"], base_dir=".")

    def test_resolves_browser_navigate_tool(self):
        """Resolves browser_navigate tool (Playwright MCP)."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(["browser_navigate"], base_dir=".")

        assert len(tools) == 1
        assert tools[0].name == "browser_navigate"

    def test_resolves_browser_click_tool(self):
        """Resolves browser_click tool (Playwright MCP)."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(["browser_click"], base_dir=".")

        assert len(tools) == 1
        assert tools[0].name == "browser_click"
        # Should have element parameter
        assert "element" in tools[0].parameters["properties"]

    def test_resolves_browser_type_tool(self):
        """Resolves browser_type tool (Playwright MCP)."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(["browser_type"], base_dir=".")

        assert len(tools) == 1
        assert tools[0].name == "browser_type"
        # Should have element and submit parameters
        assert "element" in tools[0].parameters["properties"]
        assert "submit" in tools[0].parameters["properties"]

    def test_resolves_all_browser_tools(self):
        """Resolves all browser automation tools."""
        from dokumen.loader import resolve_tools

        browser_tools = [
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_screenshot",
            "browser_take_screenshot",
            "browser_snapshot",
            "browser_close",
        ]
        tools = resolve_tools(browser_tools, base_dir=".")

        assert len(tools) == len(browser_tools)
        tool_names = [t.name for t in tools]
        for expected_name in browser_tools:
            assert expected_name in tool_names


class TestCreateProvider:
    """Tests for _create_provider function."""

    def test_create_mock_provider_returns_none(self):
        """Mock provider is no longer supported, returns None."""
        from dokumen.loader import _create_provider

        provider = _create_provider("mock", {})

        assert provider is None

    def test_create_anthropic_provider(self):
        """Creates anthropic provider (requires API key)."""
        from dokumen.loader import _create_provider

        # This will work or fail based on ANTHROPIC_API_KEY environment
        try:
            provider = _create_provider("anthropic", {"model": "claude-sonnet-4-5-20250929"})
            assert provider is not None
        except Exception:
            # Expected if no API key
            pass

    def test_unknown_provider_returns_none(self):
        """Unknown provider returns None."""
        from dokumen.loader import _create_provider

        result = _create_provider("unknown_provider")

        assert result is None


class TestLoadScaffold:
    """Tests for load_scaffold function."""

    def test_load_valid_scaffold(self, tmp_path):
        """Loads valid scaffold YAML."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: test-scaffold
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate the result."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.id == "test-scaffold"
        assert test.executor is not None
        assert len(test.judges) == 1

    def test_load_scaffold_with_files(self, tmp_path):
        """Loads scaffold with files section."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: file-test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.id == "file-test"

    def test_load_empty_file_raises(self, tmp_path):
        """Empty YAML file raises ValueError."""
        from dokumen.loader import load_scaffold

        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ValueError, match="Empty"):
            load_scaffold(str(empty_file))

    def test_load_missing_file_raises(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        from dokumen.loader import load_scaffold

        with pytest.raises(FileNotFoundError):
            load_scaffold(str(tmp_path / "missing.yaml"))

    def test_load_invalid_scaffold_raises(self, tmp_path):
        """Invalid scaffold raises ValueError."""
        from dokumen.loader import load_scaffold

        scaffold_content = """
name: test
# Missing executor and judges
"""
        scaffold_file = tmp_path / "invalid.yaml"
        scaffold_file.write_text(scaffold_content)

        with pytest.raises(ValueError, match="Invalid scaffold"):
            load_scaffold(str(scaffold_file))


class TestLoadAllScaffolds:
    """Tests for load_all_scaffolds function."""

    def test_load_all_scaffolds_from_directory(self, tmp_path):
        """Loads all scaffolds from directory."""
        from dokumen.loader import load_all_scaffolds
        from mock_provider import MockProvider

        # Create test files
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        scaffold_content = """
name: test-{num}
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Do test"
  tools: [read_file]
judges:
  - name: accuracy
    system_prompt: "Evaluate"
"""
        for i in range(2):
            (tests_dir / f"test{i}.test.yaml").write_text(
                scaffold_content.replace("{num}", str(i))
            )

        provider = MockProvider()
        tests, load_errors = load_all_scaffolds(str(tests_dir), provider=provider)

        assert len(tests) >= 2
        assert load_errors == {}

    def test_load_all_scaffolds_returns_load_errors(self, tmp_path):
        """load_all_scaffolds returns errors for scaffolds that fail to load."""
        from dokumen.loader import load_all_scaffolds
        from mock_provider import MockProvider

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Valid scaffold
        valid_content = """
name: valid-test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Do test"
  tools: [read_file]
judges:
  - name: accuracy
    system_prompt: "Evaluate"
"""
        (tests_dir / "valid.test.yaml").write_text(valid_content)

        # Invalid scaffold (missing required fields)
        invalid_content = "invalid: yaml: content: [["
        (tests_dir / "broken.test.yaml").write_text(invalid_content)

        provider = MockProvider()
        tests, load_errors = load_all_scaffolds(str(tests_dir), provider=provider)

        assert len(tests) == 1
        assert tests[0].id == "valid-test"
        assert len(load_errors) == 1
        assert "broken" in load_errors

    def test_load_all_scaffolds_extracts_name_from_yaml(self, tmp_path):
        """load_all_scaffolds uses test name from YAML when scaffold partially parses."""
        from dokumen.loader import load_all_scaffolds
        from mock_provider import MockProvider

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Scaffold with valid YAML but missing required fields for load_scaffold
        partial_content = """
name: my-named-test
files: []
executor:
  system_prompt: "test"
  user_prompt: "test"
judges: []
"""
        (tests_dir / "partial.test.yaml").write_text(partial_content)

        provider = MockProvider()
        tests, load_errors = load_all_scaffolds(str(tests_dir), provider=provider)

        # The test should fail to load (empty files/judges) and error should use the YAML name
        assert len(load_errors) == 1
        assert "my-named-test" in load_errors
        assert len(tests) == 0


class TestExtractTestName:
    """Tests for _extract_test_name function."""

    def test_extracts_name_from_yaml_content(self, tmp_path):
        """Extracts name field from valid YAML file."""
        from dokumen.loader import _extract_test_name

        scaffold = tmp_path / "my-test.test.yaml"
        scaffold.write_text("name: actual-test-name\nfiles: []\n")

        assert _extract_test_name(str(scaffold)) == "actual-test-name"

    def test_falls_back_to_filename(self, tmp_path):
        """Falls back to filename when YAML is unparseable."""
        from dokumen.loader import _extract_test_name

        scaffold = tmp_path / "fallback-test.test.yaml"
        scaffold.write_text("invalid: yaml: [[")

        assert _extract_test_name(str(scaffold)) == "fallback-test"

    def test_strips_test_yml_suffix(self, tmp_path):
        """Strips .test.yml suffix from filename."""
        from dokumen.loader import _extract_test_name

        scaffold = tmp_path / "another.test.yml"
        scaffold.write_text("broken content")

        assert _extract_test_name(str(scaffold)) == "another"

    def test_nonexistent_file(self):
        """Handles nonexistent file path by using filename."""
        from dokumen.loader import _extract_test_name

        assert _extract_test_name("/nonexistent/path/cool-test.test.yaml") == "cool-test"


class TestPlaceholderTool:
    """Tests for _create_placeholder_tool function."""

    def test_creates_placeholder(self):
        """Creates placeholder tool definition."""
        from dokumen.loader import _create_placeholder_tool

        tool = _create_placeholder_tool("my_tool")

        assert tool.name == "my_tool"
        assert "placeholder" in tool.description.lower() or "not available" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_placeholder_handler_returns_error(self):
        """Placeholder handler returns error when called."""
        from dokumen.loader import _create_placeholder_tool

        tool = _create_placeholder_tool("unavailable_tool")
        result = await tool.handler({})

        assert result.success is False
        assert "not available" in result.error.lower()


class TestResolveToolsAdvanced:
    """Advanced tests for resolve_tools edge cases."""

    def test_resolve_search_file_content(self, tmp_path, monkeypatch):
        """resolve_tools handles search_file_content tool."""
        from dokumen.loader import resolve_tools
        monkeypatch.chdir(tmp_path)

        tools = resolve_tools(["search_file_content"])

        assert len(tools) == 1
        assert tools[0].name == "search_file_content"

    def test_resolve_unknown_tool_raises(self, tmp_path, monkeypatch):
        """resolve_tools raises for unknown tool."""
        from dokumen.loader import resolve_tools
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="Unknown tool"):
            resolve_tools(["totally_made_up_tool_that_doesnt_exist"])


class TestJudgeTimeout:
    """Tests for per-judge timeout wiring in load_scaffold."""

    def _base_scaffold(self, **judge_overrides):
        """Build minimal scaffold YAML with optional judge field overrides."""
        judge_fields = {"name": "accuracy", "system_prompt": "Evaluate the result."}
        judge_fields.update(judge_overrides)
        judge_yaml_fields = "\n".join(
            f"    {k}: {v}" for k, v in judge_fields.items()
        )
        return f"""
name: timeout-test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - {judge_yaml_fields.strip()}
"""

    def test_load_scaffold_with_per_judge_timeout(self, tmp_path):
        """When judge YAML has timeout field, judge AgentObject gets that timeout."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: timeout-test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate the result."
    timeout: 60
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert len(test.judges) == 1
        assert test.judges[0].timeout == 60.0

    def test_load_scaffold_without_judge_timeout_uses_default(self, tmp_path):
        """When judge YAML has no timeout field, judge AgentObject.timeout is the default (60.0)."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: no-timeout-test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate the result."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert len(test.judges) == 1
        # Default SDK judge timeout is 120.0
        assert test.judges[0].timeout == 120.0

    def test_load_scaffold_multiple_judges_different_timeouts(self, tmp_path):
        """Multiple judges can each have their own timeout."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: multi-judge-timeout
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: fast-judge
    system_prompt: "Quick check."
    timeout: 30
  - name: slow-judge
    system_prompt: "Thorough evaluation."
    timeout: 300
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert len(test.judges) == 2
        assert test.judges[0].timeout == 30.0
        assert test.judges[1].timeout == 300.0


class TestFindConfigFileAdvanced:
    """Additional tests for _find_config_file edge cases."""

    def test_find_config_with_explicit_path(self, tmp_path, monkeypatch):
        """_find_config_file returns explicit path when it exists."""
        from dokumen.loader import _find_config_file
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "custom.yaml"
        config_file.write_text("version: '1.0'")

        result = _find_config_file(str(config_file))

        assert result == str(config_file)

    def test_find_config_searches_parent(self, tmp_path, monkeypatch):
        """_find_config_file searches parent directories."""
        from dokumen.loader import _find_config_file

        # Create config in parent
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        # Create and chdir to child
        child = tmp_path / "child"
        child.mkdir()
        monkeypatch.chdir(child)

        result = _find_config_file()

        assert result is not None
        assert "dokumen.yaml" in result


class TestGetConfiguredProvider:
    """Tests for get_configured_provider function."""

    def test_returns_provider(self, tmp_path, monkeypatch):
        """Returns configured provider."""
        from dokumen.loader import get_configured_provider

        # Create a config file with anthropic provider
        (tmp_path / "dokumen.yaml").write_text("""
version: "1.0"
provider:
  name: anthropic
""")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        provider = get_configured_provider()

        assert provider is not None


class TestToolsConfigMergeLogic:
    """Tests for ToolsConfig merge logic in load_scaffold."""

    def _create_scaffold(self, tmp_path, tools_list=None):
        """Helper to create a scaffold file with optional tools.

        tools_list=None → tools: [] (empty list means "use global defaults")
        tools_list=["read_file"] → tools: [read_file]
        """
        if tools_list is None:
            tools_yaml = "  tools: []"
        else:
            tools_entries = "\n".join(f"    - {t}" for t in tools_list)
            tools_yaml = f"  tools:\n{tools_entries}"

        scaffold_content = f"""
name: test-tools-config
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
{tools_yaml}
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        return str(scaffold_file)

    def test_scaffold_tools_override_global_defaults(self, tmp_path):
        """Scaffold tools take precedence over global defaults."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(defaults=["glob", "list_directory"])
        scaffold_path = self._create_scaffold(tmp_path, tools_list=["read_file"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        tool_names = [t.name for t in test.executor.tools]
        assert "Read" in tool_names
        # glob and list_directory should NOT be present (scaffold overrides defaults)
        assert "Glob" not in tool_names
        assert "list_directory" not in tool_names

    def test_global_defaults_used_when_scaffold_omits_tools(self, tmp_path):
        """Global defaults are used when scaffold omits tools section."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(defaults=["read_file", "glob"])
        scaffold_path = self._create_scaffold(tmp_path, tools_list=None)
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        tool_names = [t.name for t in test.executor.tools]
        assert "Read" in tool_names
        assert "Glob" in tool_names

    def test_disallowed_tool_raises_error(self, tmp_path):
        """Scaffold requesting tool not in allowed list raises ValueError with allowed tools."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(allowed=["read_file", "glob", "run_shell_command"])
        scaffold_path = self._create_scaffold(tmp_path, tools_list=["read_file", "web_fetch"])
        provider = MockProvider()

        with pytest.raises(ValueError, match="not in allowed") as exc_info:
            load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)
        assert "web_fetch" in str(exc_info.value)
        assert "Allowed tools:" in str(exc_info.value)
        assert "read_file" in str(exc_info.value)

    def test_shell_auto_inject_respects_allowed_list(self, tmp_path):
        """run_shell_command auto-injection is skipped when not in allowed list."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(allowed=["read_file", "glob"])
        scaffold_path = self._create_scaffold(tmp_path, tools_list=["read_file"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        tool_names = [t.name for t in test.executor.tools]
        assert "Bash" not in tool_names

    def test_resolve_tools_uses_shell_timeout_from_config(self):
        """resolve_tools passes shell timeout from ToolsConfig."""
        from dokumen.loader import resolve_tools
        from dokumen.config import ToolsConfig, ToolConfigMap, ShellToolConfig

        tools_config = ToolsConfig(
            config=ToolConfigMap(run_shell_command=ShellToolConfig(timeout=90.0))
        )

        tools = resolve_tools(
            ["run_shell_command"],
            base_dir=".",
            tools_config=tools_config,
        )

        assert len(tools) == 1
        assert tools[0].name == "run_shell_command"

    def test_resolve_tools_uses_web_fetch_timeout_from_config(self):
        """resolve_tools passes web_fetch timeout from ToolsConfig."""
        from dokumen.loader import resolve_tools
        from dokumen.config import ToolsConfig, ToolConfigMap, HttpToolConfig

        tools_config = ToolsConfig(
            config=ToolConfigMap(web_fetch=HttpToolConfig(timeout=10.0))
        )

        tools = resolve_tools(
            ["web_fetch"],
            base_dir=".",
            tools_config=tools_config,
        )

        assert len(tools) == 1
        assert tools[0].name == "web_fetch"

    def test_resolve_tools_web_search_merges_with_perplexity(self):
        """tools.config.web_search takes precedence over perplexity_config."""
        from dokumen.loader import resolve_tools
        from dokumen.config import ToolsConfig, ToolConfigMap, WebSearchToolConfig

        tools_config = ToolsConfig(
            config=ToolConfigMap(web_search=WebSearchToolConfig(model="sonar-pro", max_searches=10))
        )

        perplexity_config = {"model": "sonar", "max_searches": 5}

        tools = resolve_tools(
            ["web_search"],
            base_dir=".",
            tools_config=tools_config,
            perplexity_config=perplexity_config,
        )

        assert len(tools) == 1
        assert tools[0].name == "web_search"


class TestBlockedToolsFilter:
    """Tests for tools.blocked filtering in load_scaffold."""

    def _create_scaffold(self, tmp_path, tools_list=None):
        """Helper to create a scaffold file with optional tools."""
        if tools_list is None:
            tools_yaml = "  tools: []"
        else:
            tools_entries = "\n".join(f"    - {t}" for t in tools_list)
            tools_yaml = f"  tools:\n{tools_entries}"

        scaffold_content = f"""
name: test-blocked-tools
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
{tools_yaml}
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        return str(scaffold_file)

    def test_blocked_tool_removed_from_executor(self, tmp_path):
        """Blocked tools are removed from executor tool list."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(blocked=["web_fetch"])
        scaffold_path = self._create_scaffold(tmp_path, tools_list=["read_file", "web_fetch"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        tool_names = [t.name for t in test.executor.tools]
        assert "Read" in tool_names
        assert "WebFetch" not in tool_names

    def test_blocked_tool_removed_from_judge(self, tmp_path):
        """Blocked tools are removed from judge tool list."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(blocked=["run_shell_command"])
        scaffold_path = self._create_scaffold(tmp_path, tools_list=["read_file"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        # Judge tools get run_shell_command auto-injected normally,
        # but blocked should remove it
        for judge in test.judges:
            judge_tool_names = [t.name for t in judge.tools]
            assert "Bash" not in judge_tool_names

    def test_blocked_takes_precedence_over_defaults(self, tmp_path):
        """Blocked tools are removed even if they come from global defaults."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(
            defaults=["read_file", "glob", "web_fetch"],
            blocked=["web_fetch"],
        )
        # Empty tools list means use global defaults
        scaffold_path = self._create_scaffold(tmp_path, tools_list=None)
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        tool_names = [t.name for t in test.executor.tools]
        assert "Read" in tool_names
        assert "Glob" in tool_names
        assert "WebFetch" not in tool_names

    def test_none_blocked_no_filtering(self, tmp_path):
        """None blocked list means no filtering occurs."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(blocked=None)
        scaffold_path = self._create_scaffold(tmp_path, tools_list=["read_file", "web_fetch"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        tool_names = [t.name for t in test.executor.tools]
        assert "Read" in tool_names
        assert "WebFetch" in tool_names

    def test_empty_blocked_no_filtering(self, tmp_path):
        """Empty blocked list means no filtering occurs."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(blocked=[])
        scaffold_path = self._create_scaffold(tmp_path, tools_list=["read_file", "web_fetch"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        tool_names = [t.name for t in test.executor.tools]
        assert "Read" in tool_names
        assert "WebFetch" in tool_names

    def test_blocked_auto_injected_run_shell_command(self, tmp_path):
        """Blocking run_shell_command prevents auto-injection into executor tools."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        tools_config = ToolsConfig(blocked=["run_shell_command"])
        scaffold_path = self._create_scaffold(tmp_path, tools_list=["read_file"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        tool_names = [t.name for t in test.executor.tools]
        assert "Bash" not in tool_names
        assert "Read" in tool_names


class TestResearchType:
    """Tests for research test type behavior."""

    def test_research_type_default_iterations(self):
        """Research type defaults to 100 iterations when web_search is in tools."""
        from dokumen.loader import _default_executor_iterations

        result = _default_executor_iterations(["web_search", "read_file"])

        assert result == 100

    def test_research_type_web_search_only(self):
        """Default iterations is 100 with only web_search."""
        from dokumen.loader import _default_executor_iterations

        result = _default_executor_iterations(["web_search"])

        assert result == 100

    def test_standard_type_default_iterations(self):
        """Standard tests default to 100 iterations."""
        from dokumen.loader import _default_executor_iterations

        result = _default_executor_iterations(["read_file", "glob"])

        assert result == 100

    def test_research_type_skips_run_shell_command_auto_inject(self, tmp_path):
        """Research type scaffolds do not get run_shell_command auto-injected."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: test-research
type: research
files:
  - path: docs/reference.md
executor:
  system_prompt: "@prompts/research.txt"
  user_prompt: "Research this topic."
  tools:
    - web_search
    - read_file
judges:
  - name: verification
    system_prompt: "Verify the research."
    tools:
      - web_search
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        executor_tool_names = [t.name for t in test.executor.tools]
        assert "WebSearch" in executor_tool_names or "web_search" in executor_tool_names
        assert "Read" in executor_tool_names or "read_file" in executor_tool_names
        assert "Bash" not in executor_tool_names and "run_shell_command" not in executor_tool_names

        judge_tool_names = [t.name for t in test.judges[0].tools]
        # SDK judge wrappers have empty tools list (tools resolved at SDK level)
        # This assertion is no longer meaningful for SDK judges

    def test_research_type_valid_scaffold(self, tmp_path):
        """type: research scaffolds pass validation."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: test-research-valid
type: research
files:
  - path: docs/context.md
executor:
  system_prompt: "@prompts/research.txt"
  user_prompt: "Investigate this topic using web search."
  tools:
    - web_search
    - read_file
judges:
  - name: fact-check
    system_prompt: "Verify findings."
timeout: 300
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.id == "test-research-valid"
        assert test.timeout == 300.0


class TestTypeResolutionFromExecutor:
    """Tests for resolving 'type' from executor block as fallback."""

    def test_type_at_top_level_produces_test_type(self, tmp_path):
        """type: research at top level sets test_type='research'."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: test-top-level-type
type: research
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/research.txt"
  user_prompt: "Research this."
  tools:
    - web_search
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.test_type == "research"

    def test_type_in_executor_block_produces_test_type(self, tmp_path):
        """type: research in executor block sets test_type='research' as fallback."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: test-executor-type
files:
  - path: docs/api.md
executor:
  type: research
  system_prompt: "@prompts/research.txt"
  user_prompt: "Research this."
  tools:
    - web_search
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.test_type == "research"

    def test_type_at_both_levels_uses_top_level(self, tmp_path):
        """When type appears at both levels, top-level takes precedence."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: test-both-types
type: browser
files:
  - path: docs/api.md
browser:
  headless: true
executor:
  type: research
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: "Test this."
  tools:
    - browser_navigate
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.test_type == "browser"

    def test_no_type_anywhere_produces_none(self, tmp_path):
        """When type is absent at both levels, test_type is None."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: test-no-type
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.test_type is None

    def test_executor_type_research_auto_injects_judges(self, tmp_path):
        """executor.type: research triggers auto-injection of sources and verdict judges."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: test-executor-type-judges
files:
  - path: docs/api.md
executor:
  type: research
  system_prompt: "@prompts/research.txt"
  user_prompt: "Research this."
  tools:
    - web_search
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        judge_names = [j.id for j in test.judges]
        assert "sources" in judge_names
        assert "verdict" in judge_names

    def test_executor_type_research_auto_injects_web_search(self, tmp_path):
        """executor.type: research triggers auto-injection of web_search tool."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: test-executor-type-websearch
files:
  - path: docs/api.md
executor:
  type: research
  system_prompt: "@prompts/research.txt"
  user_prompt: "Research this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)

        provider = MockProvider()
        test = load_scaffold(str(scaffold_file), provider=provider)

        executor_tool_names = [t.name for t in test.executor.tools]
        assert "WebSearch" in executor_tool_names or "web_search" in executor_tool_names
        # run_shell_command (Bash) should NOT be injected for research type
        assert "Bash" not in executor_tool_names and "run_shell_command" not in executor_tool_names


class TestBrowserToolAutoInjection:
    """Tests for browser tool auto-injection when type=browser."""

    def _create_browser_scaffold(self, tmp_path, tools_list=None, test_type="browser"):
        """Helper to create a browser scaffold file."""
        if tools_list is None:
            tools_yaml = "  tools:\n    - web_fetch"
        else:
            tools_entries = "\n".join(f"    - {t}" for t in tools_list)
            tools_yaml = f"  tools:\n{tools_entries}"

        scaffold_content = f"""
name: test-browser-inject
type: {test_type}
files:
  - path: docs/api.md
browser:
  headless: true
executor:
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: "Test browser interaction."
{tools_yaml}
judges:
  - name: validation
    system_prompt: "Evaluate browser test."
timeout: 120
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        return str(scaffold_file)

    def test_browser_type_auto_injects_browser_tools(self, tmp_path):
        """type: browser scaffold with tools: [web_fetch] gets all browser tools + read_file added."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from dokumen.playwright_tools import BROWSER_TOOLS

        scaffold_path = self._create_browser_scaffold(tmp_path, tools_list=["web_fetch"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider)

        tool_names = [t.name for t in test.executor.tools]
        # Original tool should be present
        assert "WebFetch" in tool_names
        # All browser tools should be auto-injected (SDK adds mcp__playwright__ prefix)
        for browser_tool in BROWSER_TOOLS:
            matching = [t for t in tool_names if browser_tool in t]
            assert len(matching) > 0, f"Expected {browser_tool} (or prefixed variant) to be auto-injected, got {tool_names}"
        # read_file should be auto-injected
        assert "Read" in tool_names

    def test_browser_auto_inject_no_duplicates(self, tmp_path):
        """Scaffold already listing some browser tools doesn't get duplicates."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from dokumen.playwright_tools import BROWSER_TOOLS

        scaffold_path = self._create_browser_scaffold(
            tmp_path,
            tools_list=["browser_navigate", "browser_click", "read_file"]
        )
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider)

        tool_names = [t.name for t in test.executor.tools]
        # No duplicates
        assert tool_names.count("mcp__playwright__browser_navigate") == 1
        assert tool_names.count("mcp__playwright__browser_click") == 1
        assert tool_names.count("Read") == 1
        # Other browser tools should still be added (SDK uses mcp__playwright__ prefix)
        for browser_tool in BROWSER_TOOLS:
            matching = [t for t in tool_names if browser_tool in t]
            assert len(matching) > 0, f"Expected {browser_tool} (or prefixed variant) in {tool_names}"

    def test_browser_auto_inject_bypasses_allowed_list(self, tmp_path):
        """Browser tools bypass allowed list for type=browser tests (they're essential)."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider
        from dokumen.playwright_tools import BROWSER_TOOLS

        tools_config = ToolsConfig(
            allowed=["web_fetch", "browser_navigate", "browser_snapshot", "read_file"]
        )
        scaffold_path = self._create_browser_scaffold(tmp_path, tools_list=["web_fetch"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=tools_config)

        tool_names = [t.name for t in test.executor.tools]
        # ALL browser tools should be present, even those not in allowed list
        for browser_tool in BROWSER_TOOLS:
            matching = [t for t in tool_names if browser_tool in t]; assert len(matching) > 0, f"Expected {browser_tool} in {tool_names}"
        assert "Read" in tool_names

    def test_browser_auto_inject_with_no_tools_config(self, tmp_path):
        """tools_config=None means all browser tools are injected."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from dokumen.playwright_tools import BROWSER_TOOLS

        scaffold_path = self._create_browser_scaffold(tmp_path, tools_list=["web_fetch"])
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider, tools_config=None)

        tool_names = [t.name for t in test.executor.tools]
        for browser_tool in BROWSER_TOOLS:
            matching = [t for t in tool_names if browser_tool in t]; assert len(matching) > 0, f"Expected {browser_tool} in {tool_names}"
        assert "Read" in tool_names

    def test_non_browser_test_unaffected(self, tmp_path):
        """Standard tests don't get browser tools auto-injected."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        # Create a standard scaffold (not browser type)
        scaffold_content = """
name: test-standard
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        test = load_scaffold(str(scaffold_file), provider=provider)

        tool_names = [t.name for t in test.executor.tools]
        assert "mcp__playwright__browser_navigate" not in tool_names
        assert "mcp__playwright__browser_click" not in tool_names

    def test_browser_type_read_file_auto_injected(self, tmp_path):
        """read_file is auto-injected for browser tests (needed for credentials)."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_path = self._create_browser_scaffold(
            tmp_path,
            tools_list=["browser_navigate"]
        )
        provider = MockProvider()

        test = load_scaffold(scaffold_path, provider=provider)

        tool_names = [t.name for t in test.executor.tools]
        assert "Read" in tool_names


class TestToolProvenance:
    """Tests for ToolProvenance dataclass and provenance tracking in load_scaffold."""

    def test_provenance_dataclass_defaults(self):
        """ToolProvenance initializes with empty defaults."""
        from dokumen.loader import ToolProvenance

        provenance = ToolProvenance()

        assert provenance.executor_tools == {}
        assert provenance.judge_tools == {}
        assert provenance.explore_tools == {}
        assert provenance.overrides_active is False
        assert provenance.removed_tools == []

    def test_provenance_scaffold_source(self, tmp_path):
        """Tools from scaffold YAML get source 'scaffold'."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-scaffold
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
    - glob
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.tool_provenance is not None
        assert test.tool_provenance.executor_tools.get("read_file") == "scaffold"
        assert test.tool_provenance.executor_tools.get("glob") == "scaffold"

    def test_provenance_defaults_source(self, tmp_path):
        """Tools from global defaults get source 'defaults'."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-defaults
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools: []
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()
        tools_config = ToolsConfig(defaults=["read_file", "glob"])

        test = load_scaffold(str(scaffold_file), provider=provider, tools_config=tools_config)

        assert test.tool_provenance is not None
        assert test.tool_provenance.executor_tools.get("read_file") == "defaults"
        assert test.tool_provenance.executor_tools.get("glob") == "defaults"

    def test_provenance_auto_inject_standard(self, tmp_path):
        """run_shell_command auto-injected for standard tests gets 'auto:standard'."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-auto-std
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.tool_provenance is not None
        assert test.tool_provenance.executor_tools.get("run_shell_command") == "auto:standard"
        # Scaffold tools should still be "scaffold"
        assert test.tool_provenance.executor_tools.get("read_file") == "scaffold"

    def test_provenance_auto_inject_browser(self, tmp_path):
        """Browser tools auto-injected get 'auto:browser'."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-auto-browser
type: browser
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
browser:
  headless: true
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.tool_provenance is not None
        # Provenance tracks legacy tool names (before SDK mapping)
        assert test.tool_provenance.executor_tools.get("browser_navigate") == "auto:browser"
        assert test.tool_provenance.executor_tools.get("browser_click") == "auto:browser"

    def test_provenance_auto_inject_research(self, tmp_path):
        """web_search auto-injected for research tests gets 'auto:research'."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-auto-research
type: research
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/research.txt"
  user_prompt: "Research this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.tool_provenance is not None
        assert test.tool_provenance.executor_tools.get("web_search") == "auto:research"

    def test_provenance_judge_tools(self, tmp_path):
        """Judge tools are tracked per judge name."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-judge
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
    tools:
      - read_file
  - name: completeness
    system_prompt: "Check completeness."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.tool_provenance is not None
        # Each judge should have its tools tracked
        assert "accuracy" in test.tool_provenance.judge_tools
        assert test.tool_provenance.judge_tools["accuracy"].get("read_file") == "scaffold"
        # run_shell_command auto-added to judges for standard tests
        assert test.tool_provenance.judge_tools["accuracy"].get("run_shell_command") == "auto:standard"
        assert "completeness" in test.tool_provenance.judge_tools
        assert test.tool_provenance.judge_tools["completeness"].get("run_shell_command") == "auto:standard"

    def test_provenance_removed_tools(self, tmp_path):
        """Filtered tools are tracked in removed_tools."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-removed
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
    - glob
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()
        tools_config = ToolsConfig(blocked=["glob"])

        test = load_scaffold(str(scaffold_file), provider=provider, tools_config=tools_config)

        assert test.tool_provenance is not None
        assert "glob" in test.tool_provenance.removed_tools

    def test_provenance_overrides_flag(self, tmp_path):
        """overrides_active reflects whether tool overrides are present."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-overrides
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        # Without overrides
        test = load_scaffold(str(scaffold_file), provider=provider)
        assert test.tool_provenance is not None
        assert test.tool_provenance.overrides_active is False

    def test_provenance_stored_on_test_object(self, tmp_path):
        """load_scaffold() returns TestObject with provenance attribute."""
        from dokumen.loader import load_scaffold, ToolProvenance
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-stored
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        test = load_scaffold(str(scaffold_file), provider=provider)

        assert hasattr(test, 'tool_provenance')
        assert isinstance(test.tool_provenance, ToolProvenance)

    @pytest.mark.xfail(reason="Code tools not yet mapped in SDK tool resolver")
    def test_provenance_auto_inject_cross_reference(self, tmp_path):
        """Code tools auto-injected for cross-reference tests get 'auto:cross-reference'."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-auto-xref
type: cross-reference
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/cross-reference.txt"
  user_prompt: "Cross-reference this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
code_files:
  - repo: my-repo
    path: src/main.py
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        # Create code repo directory structure
        code_dir = tmp_path / "code-my-repo"
        code_dir.mkdir()
        (code_dir / "src").mkdir()
        (code_dir / "src" / "main.py").write_text("# main\n")

        test = load_scaffold(
            str(scaffold_file),
            provider=provider,
            code_repos_config=[{
                "name": "my-repo",
                "base_dir": str(code_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )

        assert test.tool_provenance is not None
        assert test.tool_provenance.executor_tools.get("code_read_file") == "auto:cross-reference"
        assert test.tool_provenance.executor_tools.get("code_search") == "auto:cross-reference"
        assert test.tool_provenance.executor_tools.get("code_glob") == "auto:cross-reference"

    def test_provenance_to_dict(self):
        """ToolProvenance.to_dict() serializes all fields."""
        from dokumen.loader import ToolProvenance

        provenance = ToolProvenance(
            executor_tools={"read_file": "scaffold", "run_shell_command": "auto:standard"},
            judge_tools={"accuracy": {"run_shell_command": "auto:standard"}},
            explore_tools={"glob": "explore:config"},
            overrides_active=True,
            removed_tools=["web_fetch"],
        )

        result = provenance.to_dict()

        assert result['executor_tools'] == {"read_file": "scaffold", "run_shell_command": "auto:standard"}
        assert result['judge_tools'] == {"accuracy": {"run_shell_command": "auto:standard"}}
        assert result['explore_tools'] == {"glob": "explore:config"}
        assert result['overrides_active'] is True
        assert result['removed_tools'] == ["web_fetch"]

    def test_provenance_to_dict_returns_copies(self):
        """to_dict() returns copies, not references to internal state."""
        from dokumen.loader import ToolProvenance

        provenance = ToolProvenance(
            executor_tools={"read_file": "scaffold"},
            judge_tools={"accuracy": {"read_file": "scaffold"}},
        )

        result = provenance.to_dict()
        result['executor_tools']['injected'] = "bad"
        result['judge_tools']['accuracy']['injected'] = "bad"

        assert "injected" not in provenance.executor_tools
        assert "injected" not in provenance.judge_tools["accuracy"]

    def test_provenance_overrides_active_true(self, tmp_path):
        """overrides_active is True when tool overrides are detected."""
        from unittest.mock import patch
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from dokumen.user_tool_overrides import ToolOverridesResult

        scaffold_content = """
name: prov-overrides-true
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        mock_overrides = ToolOverridesResult(
            overrides={
                "read_file": frozenset({"test"}),
                "run_shell_command": frozenset({"test"}),
            },
        )

        with patch('dokumen.loader.load_overrides_from_dir', return_value=mock_overrides):
            test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.tool_provenance is not None
        assert test.tool_provenance.overrides_active is True

    def test_provenance_research_auto_judges_tracked(self, tmp_path):
        """Auto-injected sources/verdict judges for research tests have provenance."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-research-judges
type: research
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/research.txt"
  user_prompt: "Research this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.tool_provenance is not None
        # Auto-injected research judges should be tracked
        assert "sources" in test.tool_provenance.judge_tools
        assert "verdict" in test.tool_provenance.judge_tools

    def test_provenance_judge_filtering_cleans_provenance(self, tmp_path):
        """When judge tools are filtered, provenance is updated to remove them."""
        from dokumen.loader import load_scaffold
        from dokumen.config import ToolsConfig
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-judge-filter
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
    tools:
      - read_file
      - glob
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()
        tools_config = ToolsConfig(blocked=["glob"])

        test = load_scaffold(str(scaffold_file), provider=provider, tools_config=tools_config)

        assert test.tool_provenance is not None
        # glob should NOT be in the judge provenance since it was filtered out
        assert "glob" not in test.tool_provenance.judge_tools.get("accuracy", {})

    def test_provenance_scaffold_includes_auto_injected_tool(self, tmp_path):
        """When scaffold explicitly includes run_shell_command, it keeps 'scaffold' source."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: prov-explicit-shell
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
    - run_shell_command
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.tool_provenance is not None
        # Should be "scaffold", not "auto:standard" since it was explicitly listed
        assert test.tool_provenance.executor_tools.get("run_shell_command") == "scaffold"


class TestPerTestModelOverride:
    """Tests for per-test executor_model and judge_model override in load_scaffold."""

    def _write_scaffold(self, tmp_path, extra_fields=""):
        """Helper to write a minimal scaffold YAML with optional extra top-level fields."""
        content = f"""
name: model-override-test
reason: Test per-test model override
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
{extra_fields}
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(content)
        return str(scaffold_file)

    def test_load_scaffold_uses_scaffold_executor_model(self, tmp_path):
        """Scaffold with executor_model creates provider with that model."""
        from dokumen.loader import load_scaffold, _create_provider
        from mock_provider import MockProvider
        from unittest.mock import patch, call

        yaml_path = self._write_scaffold(tmp_path, "executor_model: claude-opus-4-6")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            mock_create.return_value = MockProvider()
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # _create_provider should have been called with the scaffold's executor model
        executor_calls = [c for c in mock_create.call_args_list
                         if c == call("mock", "test-key", "claude-opus-4-6")]
        assert len(executor_calls) >= 1, (
            f"Expected _create_provider called with executor model 'claude-opus-4-6', "
            f"got calls: {mock_create.call_args_list}"
        )

    def test_load_scaffold_uses_scaffold_judge_model(self, tmp_path):
        """Scaffold with judge_model creates provider with that model."""
        from dokumen.loader import load_scaffold, _create_provider
        from mock_provider import MockProvider
        from unittest.mock import patch, call

        yaml_path = self._write_scaffold(tmp_path, "judge_model: claude-haiku-4-5-20251001")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            mock_create.return_value = MockProvider()
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # _create_provider should have been called with the scaffold's judge model
        judge_calls = [c for c in mock_create.call_args_list
                      if c == call("mock", "test-key", "claude-haiku-4-5-20251001")]
        assert len(judge_calls) >= 1, (
            f"Expected _create_provider called with judge model 'claude-haiku-4-5-20251001', "
            f"got calls: {mock_create.call_args_list}"
        )

    def test_load_scaffold_falls_back_to_project_model(self, tmp_path):
        """Scaffold without model fields uses project-level provider."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch

        yaml_path = self._write_scaffold(tmp_path)  # No executor_model or judge_model
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # _create_provider should NOT have been called (no scaffold override)
        mock_create.assert_not_called()

    def test_load_scaffold_scaffold_model_overrides_project_model(self, tmp_path):
        """Scaffold model takes priority over project config model."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch, call

        yaml_path = self._write_scaffold(tmp_path, "executor_model: claude-opus-4-6")

        # Create project-level executor provider with a different model
        project_executor = MockProvider()
        project_judge = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            mock_create.return_value = MockProvider()
            test = load_scaffold(
                yaml_path,
                provider=project_executor,
                executor_provider=project_executor,
                judge_provider=project_judge,
                provider_name="mock",
                api_key="test-key",
            )

        # Even though executor_provider was provided, scaffold model should create a new provider
        executor_calls = [c for c in mock_create.call_args_list
                         if c == call("mock", "test-key", "claude-opus-4-6")]
        assert len(executor_calls) >= 1, (
            f"Scaffold executor_model should override project executor provider. "
            f"Calls: {mock_create.call_args_list}"
        )

    def test_load_scaffold_both_models_different(self, tmp_path):
        """executor_model and judge_model can be set to different values."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch, call

        yaml_path = self._write_scaffold(
            tmp_path,
            "executor_model: claude-opus-4-6\njudge_model: claude-haiku-4-5-20251001",
        )
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            mock_create.return_value = MockProvider()
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # Both models should be created
        executor_calls = [c for c in mock_create.call_args_list
                         if c == call("mock", "test-key", "claude-opus-4-6")]
        judge_calls = [c for c in mock_create.call_args_list
                      if c == call("mock", "test-key", "claude-haiku-4-5-20251001")]
        assert len(executor_calls) >= 1, (
            f"Expected executor model 'claude-opus-4-6'. Calls: {mock_create.call_args_list}"
        )
        assert len(judge_calls) >= 1, (
            f"Expected judge model 'claude-haiku-4-5-20251001'. Calls: {mock_create.call_args_list}"
        )

    def test_load_scaffold_empty_string_model_falls_back(self, tmp_path):
        """Empty string executor_model (normalized to None by schema) falls back to project provider."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch

        # Schema normalizes "" to None, so data.get('executor_model') returns None
        yaml_path = self._write_scaffold(tmp_path, 'executor_model: ""')
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # With empty string → None, should NOT call _create_provider for executor
        mock_create.assert_not_called()

    def test_load_all_scaffolds_isolates_per_test_models(self, tmp_path):
        """Two scaffolds with different models get distinct providers; no cross-leak."""
        from dokumen.loader import load_all_scaffolds, _create_provider
        from mock_provider import MockProvider
        from unittest.mock import patch, call
        import os

        # Create tests directory with two scaffolds
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        scaffold_a = tests_dir / "a.test.yaml"
        scaffold_a.write_text("""
name: scaffold-a
reason: Test A
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test A."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate A."
executor_model: claude-opus-4-6
""")

        scaffold_b = tests_dir / "b.test.yaml"
        scaffold_b.write_text("""
name: scaffold-b
reason: Test B
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test B."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate B."
executor_model: claude-haiku-4-5-20251001
""")

        # Create docs directory and a minimal config
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "api.md").write_text("# API\n")

        config_file = tmp_path / "dokumen.yaml"
        config_file.write_text("""
version: "1.0"
provider:
  name: mock
  model: claude-sonnet-4-6
""")

        created_models = []
        original_create = _create_provider

        def tracking_create(name, api_key=None, model=None, **kwargs):
            created_models.append(model)
            return MockProvider()

        with patch('dokumen.loader._create_provider', side_effect=tracking_create):
            with patch.dict(os.environ, {"MOCK_API_KEY": "test-key"}, clear=False):
                tests, _load_errors = load_all_scaffolds(
                    tests_dir=str(tests_dir),
                    config_path=str(config_file),
                )

        # Both scaffold models should appear in created_models, proving isolation
        assert "claude-opus-4-6" in created_models, (
            f"Expected 'claude-opus-4-6' in created models. Got: {created_models}"
        )
        assert "claude-haiku-4-5-20251001" in created_models, (
            f"Expected 'claude-haiku-4-5-20251001' in created models. Got: {created_models}"
        )


class TestNormalizeRawModel:
    """Direct unit tests for _normalize_raw_model helper."""

    def test_none_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model(None) is None

    def test_empty_string_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model("") is None

    def test_whitespace_only_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model("   ") is None

    def test_integer_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model(123) is None

    def test_dict_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model({"nested": True}) is None

    def test_list_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model(["haiku"]) is None

    def test_bool_false_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model(False) is None

    def test_valid_alias_resolved(self):
        from dokumen.loader import _normalize_raw_model
        result = _normalize_raw_model("haiku")
        assert result == "claude-haiku-4-5-20251001"

    def test_valid_full_id_passthrough(self):
        from dokumen.loader import _normalize_raw_model
        result = _normalize_raw_model("claude-sonnet-4-6")
        assert result == "claude-sonnet-4-6"

    def test_strips_whitespace(self):
        from dokumen.loader import _normalize_raw_model
        result = _normalize_raw_model("  haiku  ")
        assert result == "claude-haiku-4-5-20251001"

    def test_unknown_model_passthrough(self):
        from dokumen.loader import _normalize_raw_model
        result = _normalize_raw_model("custom-model-v1")
        assert result == "custom-model-v1"

    def test_oversized_string_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model("a" * 201) is None

    def test_special_chars_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model("model; rm -rf /") is None

    def test_unicode_returns_none(self):
        from dokumen.loader import _normalize_raw_model
        assert _normalize_raw_model("mödeł") is None


class TestPerJudgeModelOverride:
    """Tests for per-judge model override in load_scaffold."""

    def _write_scaffold(self, tmp_path, judges_yaml=""):
        """Helper to write a scaffold YAML with customizable judges block."""
        if not judges_yaml:
            judges_yaml = """judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        content = f"""
name: per-judge-model-test
reason: Test per-judge model override
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
{judges_yaml}
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(content)
        return str(scaffold_file)

    def test_per_judge_model_creates_separate_provider(self, tmp_path):
        """Judge with model: haiku gets its own provider."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch, call

        yaml_path = self._write_scaffold(tmp_path, """judges:
  - name: accuracy
    system_prompt: "Evaluate."
    model: claude-haiku-4-5-20251001
""")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            mock_create.return_value = MockProvider()
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # _create_provider should have been called with the per-judge model
        judge_calls = [c for c in mock_create.call_args_list
                      if c == call("mock", "test-key", "claude-haiku-4-5-20251001")]
        assert len(judge_calls) >= 1, (
            f"Expected _create_provider called with judge model 'claude-haiku-4-5-20251001', "
            f"got calls: {mock_create.call_args_list}"
        )

    def test_per_judge_model_falls_back_to_test_level(self, tmp_path):
        """Judge without model uses test-level judge_model provider."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch

        yaml_path = self._write_scaffold(tmp_path, """judges:
  - name: accuracy
    system_prompt: "Evaluate."
""")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # _create_provider should NOT have been called (no per-judge or test-level override)
        mock_create.assert_not_called()

    def test_per_judge_model_ignored_without_api_key(self, tmp_path):
        """Per-judge model is safely ignored when provider_name or api_key is missing."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch

        yaml_path = self._write_scaffold(tmp_path, """judges:
  - name: accuracy
    system_prompt: "Evaluate."
    model: claude-haiku-4-5-20251001
""")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            # No api_key → per-judge model should be ignored
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
            )

        # _create_provider should NOT have been called
        mock_create.assert_not_called()

    def test_per_judge_model_whitespace_normalized(self, tmp_path):
        """Whitespace-only per-judge model treated as None, uses fallback."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch

        yaml_path = self._write_scaffold(tmp_path, """judges:
  - name: accuracy
    system_prompt: "Evaluate."
    model: "  "
""")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # Whitespace model → no per-judge provider creation
        mock_create.assert_not_called()

    def test_per_judge_model_alias_resolved(self, tmp_path):
        """Per-judge model alias 'haiku' resolved to full model ID."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch, call

        yaml_path = self._write_scaffold(tmp_path, """judges:
  - name: accuracy
    system_prompt: "Evaluate."
    model: haiku
""")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            mock_create.return_value = MockProvider()
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # Should resolve 'haiku' alias to full ID
        judge_calls = [c for c in mock_create.call_args_list
                      if c == call("mock", "test-key", "claude-haiku-4-5-20251001")]
        assert len(judge_calls) >= 1, (
            f"Expected 'haiku' resolved to 'claude-haiku-4-5-20251001', "
            f"got calls: {mock_create.call_args_list}"
        )

    def test_top_level_model_aliases_normalized_in_loader(self, tmp_path):
        """Top-level executor_model alias 'haiku' is resolved to full model ID by loader."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch, call

        yaml_path = self._write_scaffold(tmp_path)
        # Manually add executor_model alias to the scaffold
        content = (tmp_path / "test.test.yaml").read_text()
        content = "executor_model: haiku\n" + content
        (tmp_path / "test.test.yaml").write_text(content)

        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            mock_create.return_value = MockProvider()
            test = load_scaffold(
                str(tmp_path / "test.test.yaml"),
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # Should resolve 'haiku' alias to full ID
        haiku_calls = [c for c in mock_create.call_args_list
                      if c == call("mock", "test-key", "claude-haiku-4-5-20251001")]
        assert len(haiku_calls) >= 1, (
            f"Expected 'haiku' resolved to 'claude-haiku-4-5-20251001', "
            f"got calls: {mock_create.call_args_list}"
        )

    def test_per_judge_model_ignored_without_provider_name(self, tmp_path):
        """Per-judge model ignored when provider_name is None but api_key is set."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch

        yaml_path = self._write_scaffold(tmp_path, """judges:
  - name: accuracy
    system_prompt: "Evaluate."
    model: haiku
""")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name=None,
                api_key="test-key",
            )

        mock_create.assert_not_called()

    def test_per_judge_model_ignored_without_only_api_key(self, tmp_path):
        """Per-judge model ignored when api_key is None but provider_name is set."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch

        yaml_path = self._write_scaffold(tmp_path, """judges:
  - name: accuracy
    system_prompt: "Evaluate."
    model: haiku
""")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key=None,
            )

        mock_create.assert_not_called()

    def test_combined_precedence_per_judge_over_test_level(self, tmp_path):
        """Per-judge model overrides test-level judge_model for that judge."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch, call

        content = """
name: combined-precedence-test
reason: Test precedence
judge_model: sonnet
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: deep-check
    system_prompt: "Evaluate deeply."
    model: opus
  - name: simple-check
    system_prompt: "Quick check."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(content)
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            mock_create.return_value = MockProvider()
            test = load_scaffold(
                str(scaffold_file),
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        # Should create providers for: test-level judge_model (sonnet) + per-judge (opus)
        opus_calls = [c for c in mock_create.call_args_list
                     if c == call("mock", "test-key", "claude-opus-4-6")]
        sonnet_calls = [c for c in mock_create.call_args_list
                       if c == call("mock", "test-key", "claude-sonnet-4-6")]
        assert len(opus_calls) >= 1, f"Expected opus provider, got: {mock_create.call_args_list}"
        assert len(sonnet_calls) >= 1, f"Expected sonnet provider, got: {mock_create.call_args_list}"

    def test_per_judge_model_invalid_format_ignored(self, tmp_path):
        """Per-judge model with invalid characters is safely ignored."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider
        from unittest.mock import patch

        yaml_path = self._write_scaffold(tmp_path, """judges:
  - name: accuracy
    system_prompt: "Evaluate."
    model: "model; rm -rf /"
""")
        project_provider = MockProvider()

        with patch('dokumen.loader._create_provider') as mock_create:
            test = load_scaffold(
                yaml_path,
                provider=project_provider,
                provider_name="mock",
                api_key="test-key",
            )

        mock_create.assert_not_called()


class TestAgentFieldMigration:
    """Tests for agent field and type→agent migration in loader."""

    def _write_scaffold(self, tmp_path, extra_yaml=""):
        """Write a minimal test scaffold with optional extra YAML fields."""
        # Create prompts directory with required prompts
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        for name in ("general", "browser-testing", "research", "cross-reference", "documentation-validation"):
            (prompts_dir / f"{name}.txt").write_text(f"You are a {name} executor.")

        scaffold = f"""name: test-agent-migration
reason: Test agent field migration
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/general.txt"
  user_prompt: "Test the docs."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate for correctness."
{extra_yaml}"""
        yaml_path = tmp_path / "test.test.yaml"
        yaml_path.write_text(scaffold)
        return str(yaml_path)

    def test_agent_field_passed_to_test_object(self, tmp_path):
        """Scaffold with agent field should pass it to TestObject."""
        from dokumen.loader import load_scaffold

        yaml_path = self._write_scaffold(tmp_path, "agent: doc-validator")
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        assert test_obj.agent == "doc-validator"

    def test_outputs_field_passed_to_test_object(self, tmp_path):
        """Scaffold with outputs field should pass list to TestObject."""
        from dokumen.loader import load_scaffold

        yaml_path = self._write_scaffold(tmp_path, "outputs:\n  - .dokumen-cache/report.md\n  - .dokumen-cache/screenshot.png")
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        assert test_obj.outputs == [".dokumen-cache/report.md", ".dokumen-cache/screenshot.png"]

    def test_type_browser_infers_agent(self, tmp_path):
        """type: browser should infer agent: browser-tester."""
        from dokumen.loader import load_scaffold

        scaffold = """name: test-browser-agent
files:
  - path: docs/api.md
type: browser
executor:
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: "Test the app."
  tools:
    - browser_navigate
judges:
  - name: accuracy
    system_prompt: "Evaluate."
browser:
  headless: true
"""
        yaml_path = tmp_path / "test.test.yaml"
        yaml_path.write_text(scaffold)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        (prompts_dir / "browser-testing.txt").write_text("Browser executor.")

        test_obj = load_scaffold(str(yaml_path), project_root=str(tmp_path))

        assert test_obj.agent == "browser-tester"
        assert test_obj.test_type == "browser"  # backward compat preserved

    def test_type_research_infers_agent(self, tmp_path):
        """type: research should infer agent: researcher."""
        from dokumen.loader import load_scaffold

        scaffold = """name: test-research-agent
files:
  - path: docs/api.md
type: research
executor:
  system_prompt: "@prompts/research.txt"
  user_prompt: "Research the topic."
  tools:
    - web_search
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        yaml_path = tmp_path / "test.test.yaml"
        yaml_path.write_text(scaffold)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        (prompts_dir / "research.txt").write_text("Research executor.")

        test_obj = load_scaffold(str(yaml_path), project_root=str(tmp_path))

        assert test_obj.agent == "researcher"

    @pytest.mark.xfail(reason="Code tools not yet mapped in SDK tool resolver")
    def test_type_cross_reference_infers_agent(self, tmp_path):
        """type: cross-reference should infer agent: code-reviewer."""
        from dokumen.loader import load_scaffold

        scaffold = """name: test-cross-ref-agent
files:
  - path: docs/api.md
type: cross-reference
executor:
  system_prompt: "@prompts/cross-reference.txt"
  user_prompt: "Cross reference docs with code."
  tools:
    - read_file
judges:
  - name: consistency
    system_prompt: "Evaluate."
"""
        yaml_path = tmp_path / "test.test.yaml"
        yaml_path.write_text(scaffold)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        (prompts_dir / "cross-reference.txt").write_text("Cross-ref executor.")

        # Provide dummy code_repos_config so auto-injected code tools can resolve
        test_obj = load_scaffold(
            str(yaml_path),
            project_root=str(tmp_path),
            code_repos_config=[{"name": "test-repo", "path": str(tmp_path)}],
        )

        assert test_obj.agent == "code-reviewer"

    def test_no_type_no_agent_leaves_both_none(self, tmp_path):
        """Standard test without type or agent should have agent=None."""
        from dokumen.loader import load_scaffold

        yaml_path = self._write_scaffold(tmp_path)
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        assert test_obj.agent is None
        assert test_obj.outputs is None

    def test_explicit_agent_not_overridden_by_type(self, tmp_path):
        """If both agent and type are set, explicit agent is preserved."""
        from dokumen.loader import load_scaffold

        scaffold = """name: test-both-fields
files:
  - path: docs/api.md
type: browser
agent: browser-tester
executor:
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: "Test."
  tools:
    - browser_navigate
judges:
  - name: accuracy
    system_prompt: "Evaluate."
browser:
  headless: true
"""
        yaml_path = tmp_path / "test.test.yaml"
        yaml_path.write_text(scaffold)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        (prompts_dir / "browser-testing.txt").write_text("Browser executor.")

        test_obj = load_scaffold(str(yaml_path), project_root=str(tmp_path))

        assert test_obj.agent == "browser-tester"

    def test_unknown_agent_raises_hard_error(self, tmp_path):
        """Unknown agent name raises ValueError in loader."""
        from dokumen.loader import load_scaffold
        import pytest

        scaffold = """name: test-unknown
files:
  - path: docs/api.md
agent: nonexistent-agent
executor:
  system_prompt: "@prompts/general.txt"
  user_prompt: "Test."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        yaml_path = tmp_path / "test.test.yaml"
        yaml_path.write_text(scaffold)

        with pytest.raises(ValueError, match="not found"):
            load_scaffold(str(yaml_path), project_root=str(tmp_path))

    def test_agent_and_outputs_together(self, tmp_path):
        """Both agent and outputs should be passed to TestObject."""
        from dokumen.loader import load_scaffold

        yaml_path = self._write_scaffold(tmp_path, "agent: browser-tester\noutputs:\n  - .dokumen-cache/recording.webm")
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        assert test_obj.agent == "browser-tester"
        assert test_obj.outputs == [".dokumen-cache/recording.webm"]


class TestAgentBasedToolInjection:
    """Tests that agent field (without type) triggers correct tool auto-injection."""

    def _create_agent_scaffold(self, tmp_path, agent, tools=None, extra_yaml=""):
        """Helper to create scaffold with agent field (no type)."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        for name in ("general", "browser-testing", "research", "cross-reference", "documentation-validation"):
            (prompts_dir / f"{name}.txt").write_text(f"You are a {name} executor.")

        tools_yaml = ""
        if tools:
            tools_entries = "\n".join(f"    - {t}" for t in tools)
            tools_yaml = f"  tools:\n{tools_entries}"
        else:
            tools_yaml = "  tools:\n    - read_file"

        scaffold = f"""name: test-agent-injection
agent: {agent}
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/general.txt"
  user_prompt: "Test the docs."
{tools_yaml}
judges:
  - name: accuracy
    system_prompt: "Evaluate."
{extra_yaml}"""
        yaml_path = tmp_path / "test.test.yaml"
        yaml_path.write_text(scaffold)
        return str(yaml_path)

    def test_browser_agent_injects_browser_tools_without_type(self, tmp_path):
        """agent: browser-tester (no type) should auto-inject browser tools."""
        from dokumen.loader import load_scaffold
        from dokumen.playwright_tools import BROWSER_TOOLS

        yaml_path = self._create_agent_scaffold(
            tmp_path, "browser-tester",
            tools=["web_fetch"],
            extra_yaml="browser:\n  headless: true",
        )
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        tool_names = [t.name for t in test_obj.executor.tools]
        for browser_tool in BROWSER_TOOLS:
            matching = [t for t in tool_names if browser_tool in t]; assert len(matching) > 0, f"Expected {browser_tool} in {tool_names}"
        assert "Read" in tool_names

    def test_browser_agent_skips_run_shell_command(self, tmp_path):
        """agent: browser-tester should NOT get run_shell_command auto-injected."""
        from dokumen.loader import load_scaffold

        yaml_path = self._create_agent_scaffold(
            tmp_path, "browser-tester",
            tools=["browser_navigate"],
            extra_yaml="browser:\n  headless: true",
        )
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        tool_names = [t.name for t in test_obj.executor.tools]
        assert "Bash" not in tool_names

    def test_researcher_agent_injects_web_search_without_type(self, tmp_path):
        """agent: researcher (no type) should auto-inject web_search."""
        from dokumen.loader import load_scaffold

        yaml_path = self._create_agent_scaffold(
            tmp_path, "researcher",
            tools=["read_file"],
        )
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        tool_names = [t.name for t in test_obj.executor.tools]
        assert "WebSearch" in tool_names

    def test_researcher_agent_skips_run_shell_command(self, tmp_path):
        """agent: researcher should NOT get run_shell_command auto-injected."""
        from dokumen.loader import load_scaffold

        yaml_path = self._create_agent_scaffold(
            tmp_path, "researcher",
            tools=["read_file"],
        )
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        tool_names = [t.name for t in test_obj.executor.tools]
        assert "Bash" not in tool_names

    def test_researcher_agent_judge_skips_run_shell_command(self, tmp_path):
        """Judges for agent: researcher should NOT get run_shell_command auto-injected."""
        from dokumen.loader import load_scaffold

        yaml_path = self._create_agent_scaffold(
            tmp_path, "researcher",
            tools=["read_file"],
        )
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        judge_tool_names = [t.name for t in test_obj.judges[0].tools]
        assert "Bash" not in judge_tool_names

    @pytest.mark.xfail(reason="Code tools not yet mapped in SDK tool resolver")
    def test_code_reviewer_agent_injects_code_tools_without_type(self, tmp_path):
        """agent: code-reviewer (no type) should auto-inject code tools."""
        from dokumen.loader import load_scaffold

        yaml_path = self._create_agent_scaffold(
            tmp_path, "code-reviewer",
            tools=["read_file"],
        )
        test_obj = load_scaffold(
            yaml_path,
            project_root=str(tmp_path),
            code_repos_config=[{"name": "test-repo", "path": str(tmp_path)}],
        )

        tool_names = [t.name for t in test_obj.executor.tools]
        assert "code_read_file" in tool_names or "mcp__code__read_file" in tool_names
        assert "code_search" in tool_names or "mcp__code__search" in tool_names
        assert "code_glob" in tool_names or "mcp__code__glob" in tool_names

    def test_unknown_agent_raises_hard_error(self, tmp_path):
        """Unknown agent should raise ValueError (hard error)."""
        from dokumen.loader import load_scaffold

        yaml_path = self._create_agent_scaffold(
            tmp_path, "my-custom-agent",
            tools=["read_file"],
        )
        with pytest.raises(ValueError, match="not found"):
            load_scaffold(yaml_path, project_root=str(tmp_path))

    def test_doc_validator_agent_gets_standard_behavior(self, tmp_path):
        """agent: doc-validator should get run_shell_command auto-injected (standard)."""
        from dokumen.loader import load_scaffold

        yaml_path = self._create_agent_scaffold(
            tmp_path, "doc-validator",
            tools=["read_file"],
        )
        test_obj = load_scaffold(yaml_path, project_root=str(tmp_path))

        tool_names = [t.name for t in test_obj.executor.tools]
        assert "Bash" in tool_names


class TestAgentDbToolMerge:
    """Tests for merging agent tools from DB via DOKUMEN_AGENT_ID."""

    def test_agent_tools_merged_into_executor(self, tmp_path):
        """Agent tools from DB are merged into executor tool list."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: agent-merge-test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        # get_agent_tools already maps DB names to CLI names
        with patch("dokumen.agent_loader.get_agent_tools", return_value=["list_directory", "search_file_content"]):
            test = load_scaffold(str(scaffold_file), provider=provider)

        tool_names = [t.name for t in test.executor.tools]
        assert "Read" in tool_names
        assert "Bash" in tool_names
        assert "Grep" in tool_names

    def test_agent_tools_tracked_in_provenance(self, tmp_path):
        """Agent tools get provenance source 'agent:db'."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: agent-prov-test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        # get_agent_tools returns CLI-mapped names
        with patch("dokumen.agent_loader.get_agent_tools", return_value=["glob"]):
            test = load_scaffold(str(scaffold_file), provider=provider)

        assert test.tool_provenance is not None
        assert test.tool_provenance.executor_tools.get("read_file") == "scaffold"
        assert test.tool_provenance.executor_tools.get("glob") == "agent:db"

    def test_no_agent_tools_when_env_not_set(self, tmp_path):
        """No agent tools merged when DOKUMEN_AGENT_ID is not set."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: no-agent-test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        with patch("dokumen.agent_loader.get_agent_tools", return_value=[]):
            test = load_scaffold(str(scaffold_file), provider=provider)

        # Only scaffold tools + auto-injected run_shell_command
        tool_names = [t.name for t in test.executor.tools]
        assert "Read" in tool_names
        assert "Bash" in tool_names
        # No agent:db in provenance
        for source in test.tool_provenance.executor_tools.values():
            assert source != "agent:db"

    def test_agent_tools_no_duplicates(self, tmp_path):
        """Agent tools already in scaffold are not duplicated."""
        from dokumen.loader import load_scaffold
        from mock_provider import MockProvider

        scaffold_content = """
name: agent-dedup-test
files:
  - path: docs/api.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: "Test this."
  tools:
    - read_file
    - glob
judges:
  - name: accuracy
    system_prompt: "Evaluate."
"""
        scaffold_file = tmp_path / "test.test.yaml"
        scaffold_file.write_text(scaffold_content)
        provider = MockProvider()

        # Agent also has read_file — should NOT duplicate (already mapped to CLI names)
        with patch("dokumen.agent_loader.get_agent_tools", return_value=["read_file", "list_directory"]):
            test = load_scaffold(str(scaffold_file), provider=provider)

        tool_names = [t.name for t in test.executor.tools]
        assert tool_names.count("Read") == 1  # Not duplicated
        assert test.tool_provenance.executor_tools.get("read_file") == "scaffold"  # Original source preserved
