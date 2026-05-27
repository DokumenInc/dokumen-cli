"""
Unit tests for code graph tools (code_graph_find, code_graph_relationships,
code_graph_dead_code, code_graph_complexity).

Tests the code graph tool system including:
- Constants include new tool names in VALID_EXECUTOR_TOOLS and CLI_RESOLVABLE_TOOLS
- Tool factory functions exist and return correct schema
- Tool names match registry names
- Graceful handling when codegraphcontext not installed
- CODE_GRAPH_TOOLS registry
- resolve_tools() integration with code graph tools
- Auto-injection for cross-reference tests
"""
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ============================================================================
# Constants tests
# ============================================================================


class TestCodeGraphConstants:
    """Tests that constants.py includes the 4 code graph tool names."""

    CODE_GRAPH_TOOL_NAMES = {
        "code_graph_find",
        "code_graph_relationships",
        "code_graph_dead_code",
        "code_graph_complexity",
    }

    def test_valid_executor_tools_contains_code_graph_tools(self):
        """VALID_EXECUTOR_TOOLS includes all 4 code graph tool names."""
        from dokumen_schema.constants import VALID_EXECUTOR_TOOLS

        for name in self.CODE_GRAPH_TOOL_NAMES:
            assert name in VALID_EXECUTOR_TOOLS, (
                f"'{name}' missing from VALID_EXECUTOR_TOOLS"
            )

    def test_cli_resolvable_tools_contains_code_graph_tools(self):
        """CLI_RESOLVABLE_TOOLS includes all 4 code graph tool names."""
        from dokumen_schema.constants import CLI_RESOLVABLE_TOOLS

        for name in self.CODE_GRAPH_TOOL_NAMES:
            assert name in CLI_RESOLVABLE_TOOLS, (
                f"'{name}' missing from CLI_RESOLVABLE_TOOLS"
            )

    def test_code_graph_tools_are_subset_of_valid_tools(self):
        """All code graph tools are a subset of VALID_EXECUTOR_TOOLS."""
        from dokumen_schema.constants import VALID_EXECUTOR_TOOLS

        assert self.CODE_GRAPH_TOOL_NAMES.issubset(VALID_EXECUTOR_TOOLS)

    def test_existing_code_tools_still_present(self):
        """Adding code graph tools did not remove existing code_* tools."""
        from dokumen_schema.constants import VALID_EXECUTOR_TOOLS

        existing = {"code_read_file", "code_glob", "code_search", "code_list_directory"}
        assert existing.issubset(VALID_EXECUTOR_TOOLS)


# ============================================================================
# CODE_GRAPH_TOOLS registry tests
# ============================================================================


class TestCodeGraphToolsRegistry:
    """Tests for CODE_GRAPH_TOOLS dict in tools_object.py."""

    def test_registry_exists(self):
        """CODE_GRAPH_TOOLS dict exists in tools_object module."""
        from dokumen.tools_object import CODE_GRAPH_TOOLS

        assert isinstance(CODE_GRAPH_TOOLS, dict)

    def test_contains_all_four_tools(self):
        """CODE_GRAPH_TOOLS contains all 4 code graph tools."""
        from dokumen.tools_object import CODE_GRAPH_TOOLS

        assert "code_graph_find" in CODE_GRAPH_TOOLS
        assert "code_graph_relationships" in CODE_GRAPH_TOOLS
        assert "code_graph_dead_code" in CODE_GRAPH_TOOLS
        assert "code_graph_complexity" in CODE_GRAPH_TOOLS
        assert len(CODE_GRAPH_TOOLS) == 4

    def test_factories_are_callable(self):
        """All CODE_GRAPH_TOOLS values are callable factory functions."""
        from dokumen.tools_object import CODE_GRAPH_TOOLS

        for name, factory in CODE_GRAPH_TOOLS.items():
            assert callable(factory), f"CODE_GRAPH_TOOLS['{name}'] is not callable"

    def test_code_graph_tools_in_get_all_tool_names(self):
        """Code graph tools appear in get_all_tool_names()."""
        from dokumen.tools_object import get_all_tool_names

        all_names = get_all_tool_names()
        assert "code_graph_find" in all_names
        assert "code_graph_relationships" in all_names
        assert "code_graph_dead_code" in all_names
        assert "code_graph_complexity" in all_names


# ============================================================================
# Tool factory return value tests
# ============================================================================


class TestCodeGraphFindToolFactory:
    """Tests for create_code_graph_find_tool factory."""

    def test_returns_tool_definition(self, tmp_path):
        """Factory returns a ToolDefinition."""
        from dokumen.tools_object import create_code_graph_find_tool, ToolDefinition

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")
        assert isinstance(tool, ToolDefinition)

    def test_tool_name(self, tmp_path):
        """Tool name is 'code_graph_find'."""
        from dokumen.tools_object import create_code_graph_find_tool

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")
        assert tool.name == "code_graph_find"

    def test_description_mentions_graph(self, tmp_path):
        """Tool description mentions code graph."""
        from dokumen.tools_object import create_code_graph_find_tool

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")
        assert "graph" in tool.description.lower() or "code" in tool.description.lower()

    def test_parameters_schema_has_query(self, tmp_path):
        """Tool parameters schema includes 'query' property."""
        from dokumen.tools_object import create_code_graph_find_tool

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")
        assert "query" in tool.parameters["properties"]
        assert "query" in tool.parameters["required"]

    def test_handler_is_async_callable(self, tmp_path):
        """Tool handler is an async callable."""
        from dokumen.tools_object import create_code_graph_find_tool

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")
        assert callable(tool.handler)


class TestCodeGraphRelationshipsToolFactory:
    """Tests for create_code_graph_relationships_tool factory."""

    def test_returns_tool_definition(self, tmp_path):
        """Factory returns a ToolDefinition."""
        from dokumen.tools_object import create_code_graph_relationships_tool, ToolDefinition

        tool = create_code_graph_relationships_tool(str(tmp_path), "test-repo")
        assert isinstance(tool, ToolDefinition)

    def test_tool_name(self, tmp_path):
        """Tool name is 'code_graph_relationships'."""
        from dokumen.tools_object import create_code_graph_relationships_tool

        tool = create_code_graph_relationships_tool(str(tmp_path), "test-repo")
        assert tool.name == "code_graph_relationships"

    def test_parameters_schema_has_required_fields(self, tmp_path):
        """Tool parameters schema includes 'query_type' and 'target'."""
        from dokumen.tools_object import create_code_graph_relationships_tool

        tool = create_code_graph_relationships_tool(str(tmp_path), "test-repo")
        props = tool.parameters["properties"]
        assert "query_type" in props
        assert "target" in props
        assert "query_type" in tool.parameters["required"]
        assert "target" in tool.parameters["required"]


class TestCodeGraphDeadCodeToolFactory:
    """Tests for create_code_graph_dead_code_tool factory."""

    def test_returns_tool_definition(self, tmp_path):
        """Factory returns a ToolDefinition."""
        from dokumen.tools_object import create_code_graph_dead_code_tool, ToolDefinition

        tool = create_code_graph_dead_code_tool(str(tmp_path), "test-repo")
        assert isinstance(tool, ToolDefinition)

    def test_tool_name(self, tmp_path):
        """Tool name is 'code_graph_dead_code'."""
        from dokumen.tools_object import create_code_graph_dead_code_tool

        tool = create_code_graph_dead_code_tool(str(tmp_path), "test-repo")
        assert tool.name == "code_graph_dead_code"

    def test_parameters_schema(self, tmp_path):
        """Tool parameters schema has the right shape."""
        from dokumen.tools_object import create_code_graph_dead_code_tool

        tool = create_code_graph_dead_code_tool(str(tmp_path), "test-repo")
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters


class TestCodeGraphComplexityToolFactory:
    """Tests for create_code_graph_complexity_tool factory."""

    def test_returns_tool_definition(self, tmp_path):
        """Factory returns a ToolDefinition."""
        from dokumen.tools_object import create_code_graph_complexity_tool, ToolDefinition

        tool = create_code_graph_complexity_tool(str(tmp_path), "test-repo")
        assert isinstance(tool, ToolDefinition)

    def test_tool_name(self, tmp_path):
        """Tool name is 'code_graph_complexity'."""
        from dokumen.tools_object import create_code_graph_complexity_tool

        tool = create_code_graph_complexity_tool(str(tmp_path), "test-repo")
        assert tool.name == "code_graph_complexity"

    def test_parameters_schema_has_limit(self, tmp_path):
        """Tool parameters schema includes optional 'limit' property."""
        from dokumen.tools_object import create_code_graph_complexity_tool

        tool = create_code_graph_complexity_tool(str(tmp_path), "test-repo")
        assert "limit" in tool.parameters["properties"]


# ============================================================================
# Graceful degradation when codegraphcontext not installed
# ============================================================================


class TestGracefulDegradation:
    """Tests that tools handle missing codegraphcontext gracefully."""

    def test_find_tool_returns_error_when_cgc_missing(self, tmp_path):
        """code_graph_find returns error ToolResult when CGC not importable."""
        from dokumen.tools_object import create_code_graph_find_tool

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")

        # Mock codegraphcontext import failure
        with patch.dict(sys.modules, {"codegraphcontext": None, "codegraphcontext.server": None}):
            with patch("dokumen.tools_object._is_cgc_available", return_value=False):
                result = asyncio.run(tool.handler({"query": "test"}))
                assert result.success is False
                assert "not installed" in result.error.lower() or "not available" in result.error.lower()

    def test_relationships_tool_returns_error_when_cgc_missing(self, tmp_path):
        """code_graph_relationships returns error when CGC not importable."""
        from dokumen.tools_object import create_code_graph_relationships_tool

        tool = create_code_graph_relationships_tool(str(tmp_path), "test-repo")

        with patch("dokumen.tools_object._is_cgc_available", return_value=False):
            result = asyncio.run(tool.handler({
                "query_type": "find_callers",
                "target": "some_func",
            }))
            assert result.success is False
            assert "not installed" in result.error.lower() or "not available" in result.error.lower()

    def test_dead_code_tool_returns_error_when_cgc_missing(self, tmp_path):
        """code_graph_dead_code returns error when CGC not importable."""
        from dokumen.tools_object import create_code_graph_dead_code_tool

        tool = create_code_graph_dead_code_tool(str(tmp_path), "test-repo")

        with patch("dokumen.tools_object._is_cgc_available", return_value=False):
            result = asyncio.run(tool.handler({}))
            assert result.success is False
            assert "not installed" in result.error.lower() or "not available" in result.error.lower()

    def test_complexity_tool_returns_error_when_cgc_missing(self, tmp_path):
        """code_graph_complexity returns error when CGC not importable."""
        from dokumen.tools_object import create_code_graph_complexity_tool

        tool = create_code_graph_complexity_tool(str(tmp_path), "test-repo")

        with patch("dokumen.tools_object._is_cgc_available", return_value=False):
            result = asyncio.run(tool.handler({}))
            assert result.success is False
            assert "not installed" in result.error.lower() or "not available" in result.error.lower()


# ============================================================================
# Handler input validation tests
# ============================================================================


class TestFindToolInputValidation:
    """Tests for code_graph_find input validation."""

    def test_missing_query_returns_error(self, tmp_path):
        """code_graph_find returns error when query is missing."""
        from dokumen.tools_object import create_code_graph_find_tool

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            result = asyncio.run(tool.handler({}))
            assert result.success is False
            assert "query" in result.error.lower()

    def test_empty_query_returns_error(self, tmp_path):
        """code_graph_find returns error when query is empty string."""
        from dokumen.tools_object import create_code_graph_find_tool

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            result = asyncio.run(tool.handler({"query": ""}))
            assert result.success is False
            assert "empty" in result.error.lower() or "query" in result.error.lower()


class TestRelationshipsToolInputValidation:
    """Tests for code_graph_relationships input validation."""

    def test_missing_query_type_returns_error(self, tmp_path):
        """code_graph_relationships returns error when query_type is missing."""
        from dokumen.tools_object import create_code_graph_relationships_tool

        tool = create_code_graph_relationships_tool(str(tmp_path), "test-repo")

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            result = asyncio.run(tool.handler({"target": "func"}))
            assert result.success is False
            assert "query_type" in result.error.lower()

    def test_missing_target_returns_error(self, tmp_path):
        """code_graph_relationships returns error when target is missing."""
        from dokumen.tools_object import create_code_graph_relationships_tool

        tool = create_code_graph_relationships_tool(str(tmp_path), "test-repo")

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            result = asyncio.run(tool.handler({"query_type": "find_callers"}))
            assert result.success is False
            assert "target" in result.error.lower()


# ============================================================================
# Handler with mocked CGC (success paths)
# ============================================================================


class TestFindToolWithMockedCGC:
    """Tests for code_graph_find handler with mocked CGC service."""

    def test_find_delegates_to_cgc_service(self, tmp_path):
        """code_graph_find delegates to CGCIndexService.find_code."""
        from dokumen.tools_object import create_code_graph_find_tool

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")

        mock_service = MagicMock()
        mock_service.find_code.return_value = {
            "results": [{"name": "my_func", "file": "main.py"}]
        }

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            with patch("dokumen.tools_object._get_cgc_service", return_value=mock_service):
                result = asyncio.run(tool.handler({"query": "my_func"}))
                assert result.success is True
                mock_service.find_code.assert_called_once()

    def test_find_handles_cgc_error_result(self, tmp_path):
        """code_graph_find handles error returned by CGC service."""
        from dokumen.tools_object import create_code_graph_find_tool

        tool = create_code_graph_find_tool(str(tmp_path), "test-repo")

        mock_service = MagicMock()
        mock_service.find_code.return_value = {"error": "Index build failed"}

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            with patch("dokumen.tools_object._get_cgc_service", return_value=mock_service):
                result = asyncio.run(tool.handler({"query": "my_func"}))
                assert result.success is False
                assert "failed" in result.error.lower() or "error" in result.error.lower()


class TestRelationshipsToolWithMockedCGC:
    """Tests for code_graph_relationships handler with mocked CGC service."""

    def test_relationships_delegates_to_cgc_service(self, tmp_path):
        """code_graph_relationships delegates to CGCIndexService.analyze_relationships."""
        from dokumen.tools_object import create_code_graph_relationships_tool

        tool = create_code_graph_relationships_tool(str(tmp_path), "test-repo")

        mock_service = MagicMock()
        mock_service.analyze_relationships.return_value = {
            "results": [{"caller": "func_a", "callee": "func_b"}]
        }

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            with patch("dokumen.tools_object._get_cgc_service", return_value=mock_service):
                result = asyncio.run(tool.handler({
                    "query_type": "find_callers",
                    "target": "func_b",
                }))
                assert result.success is True
                mock_service.analyze_relationships.assert_called_once()


class TestDeadCodeToolWithMockedCGC:
    """Tests for code_graph_dead_code handler with mocked CGC service."""

    def test_dead_code_delegates_to_cgc_service(self, tmp_path):
        """code_graph_dead_code delegates to CGCIndexService.find_dead_code."""
        from dokumen.tools_object import create_code_graph_dead_code_tool

        tool = create_code_graph_dead_code_tool(str(tmp_path), "test-repo")

        mock_service = MagicMock()
        mock_service.find_dead_code.return_value = {
            "results": [{"name": "unused_func", "file": "utils.py"}]
        }

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            with patch("dokumen.tools_object._get_cgc_service", return_value=mock_service):
                result = asyncio.run(tool.handler({}))
                assert result.success is True
                mock_service.find_dead_code.assert_called_once()


class TestComplexityToolWithMockedCGC:
    """Tests for code_graph_complexity handler with mocked CGC service."""

    def test_complexity_delegates_to_cgc_service(self, tmp_path):
        """code_graph_complexity delegates to CGCIndexService.find_most_complex."""
        from dokumen.tools_object import create_code_graph_complexity_tool

        tool = create_code_graph_complexity_tool(str(tmp_path), "test-repo")

        mock_service = MagicMock()
        mock_service.find_most_complex.return_value = {
            "results": [{"name": "complex_func", "complexity": 15}]
        }

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            with patch("dokumen.tools_object._get_cgc_service", return_value=mock_service):
                result = asyncio.run(tool.handler({"limit": 5}))
                assert result.success is True
                mock_service.find_most_complex.assert_called_once()

    def test_complexity_default_limit(self, tmp_path):
        """code_graph_complexity uses default limit when not specified."""
        from dokumen.tools_object import create_code_graph_complexity_tool

        tool = create_code_graph_complexity_tool(str(tmp_path), "test-repo")

        mock_service = MagicMock()
        mock_service.find_most_complex.return_value = {"results": []}

        with patch("dokumen.tools_object._is_cgc_available", return_value=True):
            with patch("dokumen.tools_object._get_cgc_service", return_value=mock_service):
                result = asyncio.run(tool.handler({}))
                assert result.success is True
                # Default limit should be passed
                call_kwargs = mock_service.find_most_complex.call_args
                assert call_kwargs is not None


# ============================================================================
# resolve_tools integration tests
# ============================================================================


class TestResolveCodeGraphTools:
    """Tests for resolve_tools with code graph tools."""

    def test_resolve_code_graph_find(self, tmp_path):
        """resolve_tools resolves code_graph_find."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_graph_find"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(tmp_path),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 1
        assert tools[0].name == "code_graph_find"

    def test_resolve_code_graph_relationships(self, tmp_path):
        """resolve_tools resolves code_graph_relationships."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_graph_relationships"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(tmp_path),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 1
        assert tools[0].name == "code_graph_relationships"

    def test_resolve_code_graph_dead_code(self, tmp_path):
        """resolve_tools resolves code_graph_dead_code."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_graph_dead_code"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(tmp_path),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 1
        assert tools[0].name == "code_graph_dead_code"

    def test_resolve_code_graph_complexity(self, tmp_path):
        """resolve_tools resolves code_graph_complexity."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_graph_complexity"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(tmp_path),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 1
        assert tools[0].name == "code_graph_complexity"

    def test_resolve_code_graph_without_config_raises(self):
        """resolve_tools raises ValueError for code graph tools without code_repos_config."""
        from dokumen.loader import resolve_tools

        with pytest.raises(ValueError, match="code.*repo"):
            resolve_tools(["code_graph_find"], base_dir=".")

    def test_resolve_mixed_code_and_graph_tools(self, tmp_path):
        """resolve_tools handles a mix of code tools and code graph tools."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_read_file", "code_graph_find"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(tmp_path),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert "code_read_file" in tool_names
        assert "code_graph_find" in tool_names


# ============================================================================
# Auto-injection tests (via load_scaffold with YAML)
# ============================================================================


@pytest.fixture
def cross_ref_scaffold_dir(tmp_path: Path) -> Path:
    """Create a directory with a cross-reference scaffold for code graph injection testing."""
    # Create docs directory
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "api.md").write_text("# API\n\nDocumentation.\n")

    # Create tests directory with scaffold
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    scaffold = {
        "name": "api-cross-ref",
        "reason": "Cross-reference API docs with code",
        "type": "cross-reference",
        "files": [{"path": "docs/api.md"}],
        "code_files": [{"repo": "backend", "path": "src/routes.py"}],
        "executor": {
            "system_prompt": "@prompts/cross-reference.txt",
            "user_prompt": "Compare documentation with code.",
            "tools": ["read_file"],
        },
        "judges": [
            {"name": "accuracy", "system_prompt": "Evaluate correctness."}
        ],
        "timeout": 120,
    }
    scaffold_path = tests_dir / "api-cross-ref.test.yaml"
    scaffold_path.write_text(yaml.dump(scaffold))

    # Create a standard (non-cross-reference) scaffold
    standard_scaffold = {
        "name": "standard-test",
        "reason": "Normal doc validation",
        "files": [{"path": "docs/api.md"}],
        "executor": {
            "system_prompt": "@prompts/documentation-validation.txt",
            "user_prompt": "Validate the docs.",
            "tools": ["read_file"],
        },
        "judges": [
            {"name": "accuracy", "system_prompt": "Check accuracy."}
        ],
        "timeout": 60,
    }
    standard_path = tests_dir / "standard-test.test.yaml"
    standard_path.write_text(yaml.dump(standard_scaffold))

    # Create scaffold with code_graph_find explicitly in tools
    explicit_scaffold = {
        "name": "explicit-graph-test",
        "reason": "Explicitly uses code graph find",
        "type": "cross-reference",
        "files": [{"path": "docs/api.md"}],
        "code_files": [{"repo": "backend", "path": "src/routes.py"}],
        "executor": {
            "system_prompt": "@prompts/cross-reference.txt",
            "user_prompt": "Compare docs with code graph.",
            "tools": ["read_file", "code_graph_find"],
        },
        "judges": [
            {"name": "accuracy", "system_prompt": "Check accuracy."}
        ],
        "timeout": 120,
    }
    explicit_path = tests_dir / "explicit-graph-test.test.yaml"
    explicit_path.write_text(yaml.dump(explicit_scaffold))

    # Create code repo directory
    code_dir = tmp_path / "code-backend"
    code_dir.mkdir()
    (code_dir / "src").mkdir()
    (code_dir / "src" / "routes.py").write_text("# routes\n")

    return tmp_path


@pytest.mark.xfail(reason="Code tools not yet mapped in SDK tool resolver")
class TestCrossReferenceAutoInjection:
    """Tests that code graph tools are auto-injected for cross-reference tests."""

    def test_cross_reference_auto_injects_code_graph_tools(self, cross_ref_scaffold_dir: Path):
        """Cross-reference tests auto-inject code_graph_* tools alongside code_* tools."""
        from dokumen.loader import load_scaffold

        scaffold_path = str(cross_ref_scaffold_dir / "tests" / "api-cross-ref.test.yaml")
        code_dir = cross_ref_scaffold_dir / "code-backend"

        test_obj = load_scaffold(
            scaffold_path,
            project_root=str(cross_ref_scaffold_dir),
            code_repos_config=[{
                "name": "backend",
                "base_dir": str(code_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )

        executor_tool_names = [t.name for t in test_obj.executor.tools]

        # Should have original code tools
        assert "code_read_file" in executor_tool_names
        assert "code_search" in executor_tool_names
        assert "code_glob" in executor_tool_names
        # Should also have code graph tools
        assert "code_graph_find" in executor_tool_names
        assert "code_graph_relationships" in executor_tool_names
        assert "code_graph_dead_code" in executor_tool_names
        assert "code_graph_complexity" in executor_tool_names

    def test_standard_test_does_not_inject_code_graph_tools(self, cross_ref_scaffold_dir: Path):
        """Standard tests do NOT auto-inject code graph tools."""
        from dokumen.loader import load_scaffold

        scaffold_path = str(cross_ref_scaffold_dir / "tests" / "standard-test.test.yaml")

        test_obj = load_scaffold(
            scaffold_path,
            project_root=str(cross_ref_scaffold_dir),
        )

        executor_tool_names = [t.name for t in test_obj.executor.tools]
        assert "code_graph_find" not in executor_tool_names
        assert "code_graph_relationships" not in executor_tool_names
        assert "code_graph_dead_code" not in executor_tool_names
        assert "code_graph_complexity" not in executor_tool_names

    def test_no_duplicate_injection_when_explicitly_listed(self, cross_ref_scaffold_dir: Path):
        """If code_graph_find is explicitly listed, it is not duplicated."""
        from dokumen.loader import load_scaffold

        scaffold_path = str(cross_ref_scaffold_dir / "tests" / "explicit-graph-test.test.yaml")
        code_dir = cross_ref_scaffold_dir / "code-backend"

        test_obj = load_scaffold(
            scaffold_path,
            project_root=str(cross_ref_scaffold_dir),
            code_repos_config=[{
                "name": "backend",
                "base_dir": str(code_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )

        executor_tool_names = [t.name for t in test_obj.executor.tools]
        # code_graph_find was explicit AND would be auto-injected -- should appear only once
        assert executor_tool_names.count("code_graph_find") == 1
