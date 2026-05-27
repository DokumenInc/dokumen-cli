"""
Unit tests for code repository tools (code_read_file, code_glob, code_search, code_list_directory).

Tests the code tool system including:
- Helper functions (_matches_pattern, _validate_code_path)
- Code tool factory functions
- CODE_TOOLS registry
- resolve_tools() integration with code tools
"""
import os
import pytest
import asyncio
from pathlib import Path


# ============================================================================
# Fixtures for code tool testing
# ============================================================================


@pytest.fixture
def code_repo_dir(tmp_path: Path) -> Path:
    """Create a sample code repository directory structure."""
    # Create source files
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("def main():\n    print('hello')\n")
    (src_dir / "utils.py").write_text("def helper():\n    return 42\n")
    (src_dir / "config.py").write_text("DEBUG = True\nPORT = 8080\n")

    # Create nested directories
    api_dir = src_dir / "api"
    api_dir.mkdir()
    (api_dir / "routes.py").write_text("from fastapi import APIRouter\n")
    (api_dir / "models.py").write_text("from pydantic import BaseModel\n")

    # Create test files
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text("def test_main():\n    assert True\n")

    # Create docs
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "README.md").write_text("# Project\n\nDocumentation.\n")

    # Create a hidden file
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n")

    return tmp_path


@pytest.fixture
def code_repo_with_binary(code_repo_dir: Path) -> Path:
    """Code repo with a binary file."""
    (code_repo_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return code_repo_dir


# ============================================================================
# Tests for _matches_pattern helper
# ============================================================================


class TestMatchesPattern:
    """Tests for _matches_pattern helper function."""

    def test_empty_patterns_match_everything(self):
        """Empty patterns list should match any path."""
        from dokumen.tools_object import _matches_pattern

        assert _matches_pattern("src/main.py", []) is True

    def test_matching_glob_pattern(self):
        """Path matching a glob pattern returns True."""
        from dokumen.tools_object import _matches_pattern

        assert _matches_pattern("src/main.py", ["src/*.py"]) is True

    def test_non_matching_glob_pattern(self):
        """Path not matching any pattern returns False."""
        from dokumen.tools_object import _matches_pattern

        assert _matches_pattern("docs/readme.md", ["src/*.py"]) is False

    def test_multiple_patterns_any_match(self):
        """Returns True if path matches any pattern in list."""
        from dokumen.tools_object import _matches_pattern

        assert _matches_pattern("docs/readme.md", ["src/*.py", "docs/*.md"]) is True

    def test_recursive_glob_pattern(self):
        """Recursive glob pattern (**) matches nested paths."""
        from dokumen.tools_object import _matches_pattern

        assert _matches_pattern("src/api/routes.py", ["**/*.py"]) is True

    def test_exact_filename_pattern(self):
        """Exact filename pattern matches."""
        from dokumen.tools_object import _matches_pattern

        assert _matches_pattern("Makefile", ["Makefile"]) is True


# ============================================================================
# Tests for _validate_code_path helper
# ============================================================================


class TestValidateCodePath:
    """Tests for _validate_code_path helper function."""

    def test_valid_relative_path(self, code_repo_dir: Path):
        """Valid relative path within base_dir passes validation."""
        from dokumen.tools_object import _validate_code_path

        valid, err = _validate_code_path(
            "src/main.py", str(code_repo_dir), [], []
        )
        assert valid is True
        assert err == ""

    def test_path_traversal_rejected(self, code_repo_dir: Path):
        """Path with .. is rejected."""
        from dokumen.tools_object import _validate_code_path

        valid, err = _validate_code_path(
            "../../../etc/passwd", str(code_repo_dir), [], []
        )
        assert valid is False
        assert "traversal" in err.lower() or "outside" in err.lower()

    def test_absolute_path_rejected(self, code_repo_dir: Path):
        """Absolute path is rejected."""
        from dokumen.tools_object import _validate_code_path

        valid, err = _validate_code_path(
            "/etc/passwd", str(code_repo_dir), [], []
        )
        assert valid is False
        assert "traversal" in err.lower() or "outside" in err.lower()

    def test_include_patterns_allow(self, code_repo_dir: Path):
        """Path matching include patterns is allowed."""
        from dokumen.tools_object import _validate_code_path

        valid, err = _validate_code_path(
            "src/main.py", str(code_repo_dir), ["src/*.py"], []
        )
        assert valid is True

    def test_include_patterns_reject(self, code_repo_dir: Path):
        """Path not matching include patterns is rejected."""
        from dokumen.tools_object import _validate_code_path

        valid, err = _validate_code_path(
            "docs/README.md", str(code_repo_dir), ["src/*.py"], []
        )
        assert valid is False
        assert "include" in err.lower()

    def test_exclude_patterns_reject(self, code_repo_dir: Path):
        """Path matching exclude patterns is rejected."""
        from dokumen.tools_object import _validate_code_path

        valid, err = _validate_code_path(
            "tests/test_main.py", str(code_repo_dir), [], ["tests/*"]
        )
        assert valid is False
        assert "exclude" in err.lower()

    def test_empty_include_allows_all(self, code_repo_dir: Path):
        """Empty include patterns list means all paths are included."""
        from dokumen.tools_object import _validate_code_path

        valid, err = _validate_code_path(
            "docs/README.md", str(code_repo_dir), [], []
        )
        assert valid is True

    def test_empty_exclude_blocks_nothing(self, code_repo_dir: Path):
        """Empty exclude patterns list means nothing is excluded."""
        from dokumen.tools_object import _validate_code_path

        valid, err = _validate_code_path(
            "src/main.py", str(code_repo_dir), [], []
        )
        assert valid is True

    def test_symlink_traversal_rejected(self, code_repo_dir: Path):
        """Symlink pointing outside base_dir is rejected."""
        from dokumen.tools_object import _validate_code_path

        # Create a symlink pointing outside
        evil_link = code_repo_dir / "evil_link"
        try:
            evil_link.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlink")

        valid, err = _validate_code_path(
            "evil_link/passwd", str(code_repo_dir), [], []
        )
        assert valid is False


# ============================================================================
# Tests for create_code_read_file_tool
# ============================================================================


class TestCreateCodeReadFileTool:
    """Tests for create_code_read_file_tool factory function."""

    def test_returns_tool_definition(self, code_repo_dir: Path):
        """Factory returns a ToolDefinition."""
        from dokumen.tools_object import create_code_read_file_tool, ToolDefinition

        tool = create_code_read_file_tool(str(code_repo_dir))
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "code_read_file"

    def test_reads_file_within_repo(self, code_repo_dir: Path):
        """code_read_file reads files within code repo directory."""
        from dokumen.tools_object import create_code_read_file_tool

        tool = create_code_read_file_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"file_path": "src/main.py"})
        )
        assert result.success is True
        assert "def main():" in result.output

    def test_rejects_path_traversal(self, code_repo_dir: Path):
        """code_read_file rejects path traversal attempts."""
        from dokumen.tools_object import create_code_read_file_tool

        tool = create_code_read_file_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"file_path": "../../../etc/passwd"})
        )
        assert result.success is False
        assert result.error is not None

    def test_applies_include_filter(self, code_repo_dir: Path):
        """code_read_file applies paths_include filter."""
        from dokumen.tools_object import create_code_read_file_tool

        tool = create_code_read_file_tool(
            str(code_repo_dir),
            include_patterns=["src/*.py"]
        )
        # src/main.py should be allowed
        result = asyncio.run(
            tool.handler({"file_path": "src/main.py"})
        )
        assert result.success is True

        # docs/README.md should be rejected
        result = asyncio.run(
            tool.handler({"file_path": "docs/README.md"})
        )
        assert result.success is False
        assert "include" in result.error.lower()

    def test_applies_exclude_filter(self, code_repo_dir: Path):
        """code_read_file applies paths_exclude filter."""
        from dokumen.tools_object import create_code_read_file_tool

        tool = create_code_read_file_tool(
            str(code_repo_dir),
            exclude_patterns=["tests/*"]
        )
        # tests/test_main.py should be rejected
        result = asyncio.run(
            tool.handler({"file_path": "tests/test_main.py"})
        )
        assert result.success is False
        assert "exclude" in result.error.lower()

    def test_file_not_found(self, code_repo_dir: Path):
        """code_read_file returns error for nonexistent file."""
        from dokumen.tools_object import create_code_read_file_tool

        tool = create_code_read_file_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"file_path": "nonexistent.py"})
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_missing_file_path_param(self, code_repo_dir: Path):
        """code_read_file returns error when file_path param is missing."""
        from dokumen.tools_object import create_code_read_file_tool

        tool = create_code_read_file_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({})
        )
        assert result.success is False

    def test_description_mentions_code_repo(self, code_repo_dir: Path):
        """Tool description mentions code repository."""
        from dokumen.tools_object import create_code_read_file_tool

        tool = create_code_read_file_tool(str(code_repo_dir))
        assert "code" in tool.description.lower()

    def test_reads_nested_file(self, code_repo_dir: Path):
        """code_read_file reads deeply nested files."""
        from dokumen.tools_object import create_code_read_file_tool

        tool = create_code_read_file_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"file_path": "src/api/routes.py"})
        )
        assert result.success is True
        assert "APIRouter" in result.output


# ============================================================================
# Tests for create_code_glob_tool
# ============================================================================


class TestCreateCodeGlobTool:
    """Tests for create_code_glob_tool factory function."""

    def test_returns_tool_definition(self, code_repo_dir: Path):
        """Factory returns a ToolDefinition."""
        from dokumen.tools_object import create_code_glob_tool, ToolDefinition

        tool = create_code_glob_tool(str(code_repo_dir))
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "code_glob"

    def test_finds_files_within_repo(self, code_repo_dir: Path):
        """code_glob finds files matching pattern within code repo."""
        from dokumen.tools_object import create_code_glob_tool

        tool = create_code_glob_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"pattern": "**/*.py"})
        )
        assert result.success is True
        assert "main.py" in result.output

    def test_results_are_relative_paths(self, code_repo_dir: Path):
        """code_glob returns paths relative to code repo root."""
        from dokumen.tools_object import create_code_glob_tool

        tool = create_code_glob_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"pattern": "**/*.py"})
        )
        assert result.success is True
        # Should not contain absolute paths
        assert str(code_repo_dir) not in result.output

    def test_description_mentions_code_repo(self, code_repo_dir: Path):
        """Tool description mentions code repository."""
        from dokumen.tools_object import create_code_glob_tool

        tool = create_code_glob_tool(str(code_repo_dir))
        assert "code" in tool.description.lower()

    def test_missing_pattern_param(self, code_repo_dir: Path):
        """code_glob returns error when pattern param is missing."""
        from dokumen.tools_object import create_code_glob_tool

        tool = create_code_glob_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({})
        )
        assert result.success is False

    def test_no_matches_returns_message(self, code_repo_dir: Path):
        """code_glob returns informative message when no files match."""
        from dokumen.tools_object import create_code_glob_tool

        tool = create_code_glob_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"pattern": "**/*.xyz"})
        )
        assert result.success is True
        assert "no files" in result.output.lower() or "0" in result.output


# ============================================================================
# Tests for create_code_search_tool
# ============================================================================


class TestCreateCodeSearchTool:
    """Tests for create_code_search_tool factory function."""

    def test_returns_tool_definition(self, code_repo_dir: Path):
        """Factory returns a ToolDefinition."""
        from dokumen.tools_object import create_code_search_tool, ToolDefinition

        tool = create_code_search_tool(str(code_repo_dir))
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "code_search"

    def test_searches_file_content(self, code_repo_dir: Path):
        """code_search finds matching content in code files."""
        from dokumen.tools_object import create_code_search_tool

        tool = create_code_search_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"pattern": "def main"})
        )
        assert result.success is True
        assert "main.py" in result.output

    def test_missing_pattern_param(self, code_repo_dir: Path):
        """code_search returns error when pattern param is missing."""
        from dokumen.tools_object import create_code_search_tool

        tool = create_code_search_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({})
        )
        assert result.success is False

    def test_description_mentions_code_repo(self, code_repo_dir: Path):
        """Tool description mentions code repository."""
        from dokumen.tools_object import create_code_search_tool

        tool = create_code_search_tool(str(code_repo_dir))
        assert "code" in tool.description.lower()

    def test_no_matches_returns_message(self, code_repo_dir: Path):
        """code_search returns message when no matches found."""
        from dokumen.tools_object import create_code_search_tool

        tool = create_code_search_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"pattern": "nonexistent_unique_string_xyz"})
        )
        assert result.success is True
        # Should indicate no matches or empty output


# ============================================================================
# Tests for create_code_list_directory_tool
# ============================================================================


class TestCreateCodeListDirectoryTool:
    """Tests for create_code_list_directory_tool factory function."""

    def test_returns_tool_definition(self, code_repo_dir: Path):
        """Factory returns a ToolDefinition."""
        from dokumen.tools_object import create_code_list_directory_tool, ToolDefinition

        tool = create_code_list_directory_tool(str(code_repo_dir))
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "code_list_directory"

    def test_lists_directory_contents(self, code_repo_dir: Path):
        """code_list_directory lists contents of code repo directory."""
        from dokumen.tools_object import create_code_list_directory_tool

        tool = create_code_list_directory_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"path": "src"})
        )
        assert result.success is True
        assert "main.py" in result.output

    def test_lists_root_directory(self, code_repo_dir: Path):
        """code_list_directory lists code repo root when no path given."""
        from dokumen.tools_object import create_code_list_directory_tool

        tool = create_code_list_directory_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({})
        )
        assert result.success is True
        assert "src" in result.output

    def test_description_mentions_code_repo(self, code_repo_dir: Path):
        """Tool description mentions code repository."""
        from dokumen.tools_object import create_code_list_directory_tool

        tool = create_code_list_directory_tool(str(code_repo_dir))
        assert "code" in tool.description.lower()

    def test_nonexistent_directory(self, code_repo_dir: Path):
        """code_list_directory returns error for nonexistent directory."""
        from dokumen.tools_object import create_code_list_directory_tool

        tool = create_code_list_directory_tool(str(code_repo_dir))
        result = asyncio.run(
            tool.handler({"path": "nonexistent_dir"})
        )
        assert result.success is False


# ============================================================================
# Tests for CODE_TOOLS registry
# ============================================================================


class TestCodeToolsRegistry:
    """Tests for CODE_TOOLS dict."""

    def test_contains_all_four_tools(self):
        """CODE_TOOLS contains all 4 code tools."""
        from dokumen.tools_object import CODE_TOOLS

        assert "code_read_file" in CODE_TOOLS
        assert "code_glob" in CODE_TOOLS
        assert "code_search" in CODE_TOOLS
        assert "code_list_directory" in CODE_TOOLS
        assert len(CODE_TOOLS) == 4

    def test_keys_match_schema_constants(self):
        """CODE_TOOLS keys match the shared schema constants."""
        from dokumen.tools_object import CODE_TOOLS
        from dokumen_schema.constants import CLI_RESOLVABLE_TOOLS

        code_tool_names = {"code_read_file", "code_glob", "code_search", "code_list_directory"}
        # All code tools should be in CLI_RESOLVABLE_TOOLS
        assert code_tool_names.issubset(CLI_RESOLVABLE_TOOLS)
        # And in CODE_TOOLS
        assert code_tool_names == set(CODE_TOOLS.keys())

    def test_factories_are_callable(self):
        """All CODE_TOOLS values are callable factory functions."""
        from dokumen.tools_object import CODE_TOOLS

        for name, factory in CODE_TOOLS.items():
            assert callable(factory), f"CODE_TOOLS['{name}'] is not callable"

    def test_code_tools_in_get_all_tool_names(self):
        """Code tools appear in get_all_tool_names()."""
        from dokumen.tools_object import get_all_tool_names

        all_names = get_all_tool_names()
        assert "code_read_file" in all_names
        assert "code_glob" in all_names
        assert "code_search" in all_names
        assert "code_list_directory" in all_names


# ============================================================================
# Tests for resolve_tools with code tools
# ============================================================================


class TestResolveCodeTools:
    """Tests for resolve_tools with code tools."""

    def test_resolve_code_read_file(self, code_repo_dir: Path):
        """resolve_tools handles code_read_file."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_read_file"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(code_repo_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 1
        assert tools[0].name == "code_read_file"

    def test_resolve_code_glob(self, code_repo_dir: Path):
        """resolve_tools handles code_glob."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_glob"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(code_repo_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 1
        assert tools[0].name == "code_glob"

    def test_resolve_code_search(self, code_repo_dir: Path):
        """resolve_tools handles code_search."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_search"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(code_repo_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 1
        assert tools[0].name == "code_search"

    def test_resolve_code_list_directory(self, code_repo_dir: Path):
        """resolve_tools handles code_list_directory."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_list_directory"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(code_repo_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 1
        assert tools[0].name == "code_list_directory"

    def test_resolve_code_tools_without_config_raises(self):
        """resolve_tools raises ValueError for code tools without code_repos_config."""
        from dokumen.loader import resolve_tools

        with pytest.raises(ValueError, match="code.*repo"):
            resolve_tools(["code_read_file"], base_dir=".")

    def test_resolve_mixed_tools(self, code_repo_dir: Path):
        """resolve_tools handles a mix of regular and code tools."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["read_file", "code_read_file"],
            base_dir=str(code_repo_dir),
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(code_repo_dir),
                "include_patterns": [],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert "read_file" in tool_names
        assert "code_read_file" in tool_names

    def test_resolve_code_tool_with_include_patterns(self, code_repo_dir: Path):
        """resolve_tools passes include patterns to code tool factory."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["code_read_file"],
            base_dir=".",
            code_repos_config=[{
                "name": "my-code",
                "base_dir": str(code_repo_dir),
                "include_patterns": ["src/**/*.py"],
                "exclude_patterns": [],
            }],
        )
        assert len(tools) == 1
        # The tool should enforce the include pattern
        result = asyncio.run(
            tools[0].handler({"file_path": "docs/README.md"})
        )
        assert result.success is False
