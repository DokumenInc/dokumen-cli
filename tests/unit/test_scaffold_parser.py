"""Tests for scaffold_parser module — YAML parsing utilities."""
import os
import pytest
from unittest.mock import patch

from dokumen.scaffold_parser import (
    substitute_prompt_variables,
    find_project_root,
    normalize_raw_model,
    parse_max_iterations,
    default_executor_iterations,
    parse_browser_config,
    parse_viewport_size,
    extract_test_name,
    extract_file_paths,
)


class TestSubstitutePromptVariables:
    """Tests for substitute_prompt_variables."""

    def test_replaces_single_variable(self):
        result = substitute_prompt_variables("Hello {name}!", {"name": "world"})
        assert result == "Hello world!"

    def test_replaces_multiple_variables(self):
        result = substitute_prompt_variables(
            "{greeting} {name}, welcome to {place}",
            {"greeting": "Hi", "name": "Alice", "place": "Dokumen"}
        )
        assert result == "Hi Alice, welcome to Dokumen"

    def test_empty_prompt_returns_empty(self):
        assert substitute_prompt_variables("", {"key": "val"}) == ""

    def test_none_prompt_returns_empty(self):
        assert substitute_prompt_variables(None, {"key": "val"}) == ""

    def test_empty_variables_returns_prompt(self):
        assert substitute_prompt_variables("hello", {}) == "hello"

    def test_none_variables_returns_prompt(self):
        assert substitute_prompt_variables("hello", None) == "hello"

    def test_no_matching_variables_unchanged(self):
        result = substitute_prompt_variables("no {vars} here", {"other": "val"})
        assert result == "no {vars} here"

    def test_working_dir_variable(self):
        result = substitute_prompt_variables(
            "Base: {working_dir}/docs", {"working_dir": "/project"}
        )
        assert result == "Base: /project/docs"


class TestFindProjectRoot:
    """Tests for find_project_root."""

    def test_finds_root_with_dokumen_yaml(self, tmp_path):
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")
        subdir = tmp_path / "tests" / "unit"
        subdir.mkdir(parents=True)
        test_file = subdir / "test.yaml"
        test_file.write_text("name: test")

        result = find_project_root(str(test_file))
        assert result == str(tmp_path)

    def test_returns_start_dir_when_not_found(self, tmp_path):
        subdir = tmp_path / "orphan"
        subdir.mkdir()

        result = find_project_root(str(subdir))
        assert result == str(subdir)

    def test_handles_file_path(self, tmp_path):
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")
        test_file = tmp_path / "test.yaml"
        test_file.write_text("test")

        result = find_project_root(str(test_file))
        assert result == str(tmp_path)


class TestNormalizeRawModel:
    """Tests for normalize_raw_model."""

    def test_resolves_alias(self):
        result = normalize_raw_model("sonnet")
        assert result is not None
        # Should resolve to a full model ID

    def test_returns_full_model_id_unchanged(self):
        result = normalize_raw_model("claude-sonnet-4-6")
        assert result == "claude-sonnet-4-6"

    def test_none_returns_none(self):
        assert normalize_raw_model(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_raw_model("") is None

    def test_whitespace_only_returns_none(self):
        assert normalize_raw_model("   ") is None

    def test_non_string_returns_none(self):
        assert normalize_raw_model(123) is None

    def test_too_long_returns_none(self):
        assert normalize_raw_model("a" * 201) is None

    def test_invalid_chars_returns_none(self):
        assert normalize_raw_model("model name with spaces") is None

    def test_strips_whitespace(self):
        result = normalize_raw_model("  claude-sonnet-4-6  ")
        assert result == "claude-sonnet-4-6"


class TestParseMaxIterations:
    """Tests for parse_max_iterations."""

    def test_none_returns_default(self):
        assert parse_max_iterations(None, default=10) == 10

    def test_none_returns_none_without_default(self):
        assert parse_max_iterations(None) is None

    def test_int_value(self):
        assert parse_max_iterations(50) == 50

    def test_string_int_value(self):
        assert parse_max_iterations("50") == 50

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="Invalid max_iterations"):
            parse_max_iterations("not_a_number")


class TestDefaultExecutorIterations:
    """Tests for default_executor_iterations."""

    def test_standard_tools(self):
        result = default_executor_iterations(["read_file", "glob"])
        assert result == 100

    def test_browser_tools(self):
        result = default_executor_iterations(["browser_navigate", "read_file"])
        assert result == 100

    def test_research_tools(self):
        result = default_executor_iterations(["web_search"])
        assert result == 100

    def test_empty_tools(self):
        result = default_executor_iterations([])
        assert result == 100


class TestParseBrowserConfig:
    """Tests for parse_browser_config."""

    def test_none_returns_none(self):
        assert parse_browser_config(None) is None

    def test_empty_dict_returns_none(self):
        assert parse_browser_config({}) is None

    def test_non_dict_returns_none(self):
        assert parse_browser_config("not a dict") is None

    def test_full_config(self):
        result = parse_browser_config({
            "headless": True,
            "save_video": True,
            "viewport": "1920x1080",
        })
        assert result is not None
        assert result.headless is True
        assert result.save_video is True
        assert result.viewport_size == "1920x1080"

    def test_viewport_size_alias(self):
        result = parse_browser_config({
            "viewport_size": "1280x720",
        })
        assert result is not None
        assert result.viewport_size == "1280x720"


class TestParseViewportSize:
    """Tests for parse_viewport_size."""

    def test_none_returns_none(self):
        assert parse_viewport_size(None) is None

    def test_string_passthrough(self):
        assert parse_viewport_size("1280x720") == "1280x720"

    def test_list_format(self):
        assert parse_viewport_size([1280, 720]) == "1280x720"

    def test_tuple_format(self):
        assert parse_viewport_size((1920, 1080)) == "1920x1080"

    def test_dict_format(self):
        assert parse_viewport_size({"width": 800, "height": 600}) == "800x600"

    def test_dict_missing_width_returns_none(self):
        assert parse_viewport_size({"height": 600}) is None

    def test_dict_missing_height_returns_none(self):
        assert parse_viewport_size({"width": 800}) is None

    def test_unsupported_type_returns_none(self):
        assert parse_viewport_size(42) is None


class TestExtractTestName:
    """Tests for extract_test_name."""

    def test_extracts_from_yaml_content(self, tmp_path):
        yaml_file = tmp_path / "my-test.test.yaml"
        yaml_file.write_text("name: actual-test-name\nreason: test\n")
        assert extract_test_name(str(yaml_file)) == "actual-test-name"

    def test_falls_back_to_filename(self, tmp_path):
        yaml_file = tmp_path / "fallback-name.test.yaml"
        yaml_file.write_text("invalid: yaml: {{")  # intentionally bad
        assert extract_test_name(str(yaml_file)) == "fallback-name"

    def test_strips_test_yml_suffix(self, tmp_path):
        yaml_file = tmp_path / "my-test.test.yml"
        yaml_file.write_text("not: yaml: {{")
        assert extract_test_name(str(yaml_file)) == "my-test"

    def test_nonexistent_file_uses_filename(self):
        result = extract_test_name("/nonexistent/path/cool-test.test.yaml")
        assert result == "cool-test"


class TestExtractFilePaths:
    """Tests for extract_file_paths."""

    def test_dict_format(self):
        files = [{"path": "docs/a.md"}, {"path": "docs/b.md"}]
        assert extract_file_paths(files) == ["docs/a.md", "docs/b.md"]

    def test_string_format(self):
        files = ["docs/a.md", "docs/b.md"]
        assert extract_file_paths(files) == ["docs/a.md", "docs/b.md"]

    def test_mixed_format(self):
        files = [{"path": "docs/a.md"}, "docs/b.md"]
        assert extract_file_paths(files) == ["docs/a.md", "docs/b.md"]

    def test_empty_list(self):
        assert extract_file_paths([]) == []

    def test_dict_without_path_key(self):
        files = [{"name": "no-path"}]
        assert extract_file_paths(files) == [""]
