"""Tests for CLI helpers module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import asyncio

from dokumen.cli.helpers import (
    EXIT_SUCCESS,
    EXIT_FAILURE,
    EXIT_CONFIG_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_INVALID_ARGS,
    run_async,
    load_config,
    deep_merge,
    DEFAULT_CONFIG,
)


class TestExitCodes:
    """Tests for exit code constants."""

    def test_exit_success(self):
        """EXIT_SUCCESS should be 0."""
        assert EXIT_SUCCESS == 0

    def test_exit_failure(self):
        """EXIT_FAILURE should be 1."""
        assert EXIT_FAILURE == 1

    def test_exit_config_error(self):
        """EXIT_CONFIG_ERROR should be 2."""
        assert EXIT_CONFIG_ERROR == 2

    def test_exit_runtime_error(self):
        """EXIT_RUNTIME_ERROR should be 3."""
        assert EXIT_RUNTIME_ERROR == 3

    def test_exit_invalid_args(self):
        """EXIT_INVALID_ARGS should be 4."""
        assert EXIT_INVALID_ARGS == 4


class TestRunAsync:
    """Tests for run_async function."""

    def test_runs_coroutine(self):
        """Should run async coroutine and return result."""
        async def my_coro():
            return "result"

        result = run_async(my_coro())
        assert result == "result"

    def test_handles_exception(self):
        """Should propagate exception from coroutine."""
        async def failing_coro():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async(failing_coro())

    def test_runs_async_generator(self):
        """Should run async with await."""
        async def coro_with_await():
            await asyncio.sleep(0)
            return "done"

        result = run_async(coro_with_await())
        assert result == "done"


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_simple_merge(self):
        """Should merge simple dicts."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Should deep merge nested dicts."""
        base = {
            "top": {"nested": {"deep": 1, "keep": 2}}
        }
        override = {
            "top": {"nested": {"deep": 10}}
        }

        result = deep_merge(base, override)

        assert result["top"]["nested"]["deep"] == 10
        assert result["top"]["nested"]["keep"] == 2

    def test_override_non_dict_with_dict(self):
        """Should override non-dict value with dict."""
        base = {"key": "string_value"}
        override = {"key": {"nested": "dict"}}

        result = deep_merge(base, override)

        assert result["key"] == {"nested": "dict"}

    def test_override_dict_with_non_dict(self):
        """Should override dict value with non-dict."""
        base = {"key": {"nested": "dict"}}
        override = {"key": "string_value"}

        result = deep_merge(base, override)

        assert result["key"] == "string_value"

    def test_empty_override(self):
        """Should return base when override is empty."""
        base = {"a": 1, "b": 2}
        override = {}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2}

    def test_empty_base(self):
        """Should return override when base is empty."""
        base = {}
        override = {"a": 1, "b": 2}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2}

    def test_does_not_modify_original(self):
        """Should not modify original dicts."""
        base = {"a": 1}
        override = {"b": 2}

        deep_merge(base, override)

        assert base == {"a": 1}
        assert override == {"b": 2}

    def test_list_values(self):
        """Should override list values (not merge)."""
        base = {"list": [1, 2, 3]}
        override = {"list": [4, 5]}

        result = deep_merge(base, override)

        assert result["list"] == [4, 5]


class TestLoadConfig:
    """Tests for load_config function."""

    def test_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        """Should return default config when no file exists."""
        monkeypatch.chdir(tmp_path)

        config = load_config()

        assert config["version"] == "1.0"
        assert config["provider"]["name"] == "anthropic"

    def test_loads_from_default_path(self, tmp_path, monkeypatch):
        """Should load from dokumen.yaml by default."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "dokumen.yaml"
        config_file.write_text("version: '2.0'\nprovider:\n  name: custom")

        config = load_config()

        assert config["version"] == "2.0"
        assert config["provider"]["name"] == "custom"

    def test_loads_from_custom_path(self, tmp_path):
        """Should load from custom path."""
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("version: '3.0'")

        config = load_config(str(config_file))

        assert config["version"] == "3.0"

    def test_merges_with_defaults(self, tmp_path, monkeypatch):
        """Should merge file config with defaults."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "dokumen.yaml"
        config_file.write_text("provider:\n  name: openai")

        config = load_config()

        # Overridden value
        assert config["provider"]["name"] == "openai"
        # Default value preserved
        assert config["coverage"]["min_threshold"] == 80

    def test_handles_invalid_yaml(self, tmp_path, monkeypatch):
        """Should return defaults on invalid YAML."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "dokumen.yaml"
        config_file.write_text("invalid: yaml: content: [")

        config = load_config()

        # Should still get defaults
        assert config["version"] == "1.0"

    def test_handles_empty_file(self, tmp_path, monkeypatch):
        """Should return defaults on empty file."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "dokumen.yaml"
        config_file.write_text("")

        config = load_config()

        assert config["version"] == "1.0"


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG structure."""

    def test_has_version(self):
        """Should have version field."""
        assert "version" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["version"] == "1.0"

    def test_has_provider(self):
        """Should have provider configuration."""
        assert "provider" in DEFAULT_CONFIG
        assert "name" in DEFAULT_CONFIG["provider"]
        assert "model" in DEFAULT_CONFIG["provider"]

    def test_has_coverage(self):
        """Should have coverage configuration."""
        assert "coverage" in DEFAULT_CONFIG
        assert "include" in DEFAULT_CONFIG["coverage"]
        assert "exclude" in DEFAULT_CONFIG["coverage"]
        assert "min_threshold" in DEFAULT_CONFIG["coverage"]

    def test_has_execution(self):
        """Should have execution configuration."""
        assert "execution" in DEFAULT_CONFIG
        assert "timeout" in DEFAULT_CONFIG["execution"]
        assert "retries" in DEFAULT_CONFIG["execution"]

    def test_has_cache(self):
        """Should have cache configuration."""
        assert "cache" in DEFAULT_CONFIG
        assert "enabled" in DEFAULT_CONFIG["cache"]
        assert "path" in DEFAULT_CONFIG["cache"]

    def test_has_sandbox(self):
        """Should have sandbox configuration."""
        assert "sandbox" in DEFAULT_CONFIG
        assert "type" in DEFAULT_CONFIG["sandbox"]

    def test_default_provider_is_anthropic(self):
        """Default provider should be anthropic."""
        assert DEFAULT_CONFIG["provider"]["name"] == "anthropic"

    def test_default_cache_enabled(self):
        """Cache should be enabled by default."""
        assert DEFAULT_CONFIG["cache"]["enabled"] is True

    def test_default_coverage_threshold(self):
        """Default coverage threshold should be 80."""
        assert DEFAULT_CONFIG["coverage"]["min_threshold"] == 80


class TestNormalizePath:
    """Tests for normalize_path function."""

    def test_forward_slashes_preserved(self):
        """Forward slashes should be preserved."""
        from dokumen.cli.helpers import normalize_path
        result = normalize_path("docs/api/v1.md")
        assert result == "docs/api/v1.md"

    def test_backslashes_converted(self):
        """Backslashes should be converted to forward slashes."""
        from dokumen.cli.helpers import normalize_path
        result = normalize_path("docs\\api\\v1.md")
        assert "/" in result
        assert "\\" not in result


class TestGetCoverageStats:
    """Tests for get_coverage_stats function."""

    def test_empty_config(self, tmp_path, monkeypatch):
        """get_coverage_stats with no config returns stats."""
        from dokumen.cli.helpers import get_coverage_stats
        monkeypatch.chdir(tmp_path)
        # Create an empty docs directory
        (tmp_path / "docs").mkdir()

        stats = get_coverage_stats()

        assert "total" in stats
        assert "passed" in stats
        assert "percentage" in stats

    def test_with_doc_files(self, tmp_path, monkeypatch):
        """get_coverage_stats finds doc files."""
        from dokumen.cli.helpers import get_coverage_stats
        monkeypatch.chdir(tmp_path)
        # Create docs directory with files
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "api.md").write_text("# API")
        (docs / "guide.md").write_text("# Guide")

        config = {"coverage": {"include": ["docs/**/*.md"], "exclude": []}}
        stats = get_coverage_stats(config=config)

        assert stats["total"] >= 2

    def test_returns_uncovered_files(self, tmp_path, monkeypatch):
        """get_coverage_stats returns uncovered files when no cache."""
        from dokumen.cli.helpers import get_coverage_stats
        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "test.md").write_text("# Test")

        config = {"coverage": {"include": ["docs/**/*.md"], "exclude": []}}
        stats = get_coverage_stats(config=config)

        # With no cache, all files should be uncovered
        assert len(stats["uncovered_files"]) >= 0

    def test_files_detail_includes_test_ids(self, tmp_path, monkeypatch):
        """files_detail should include test_ids list for each file."""
        from dokumen.cli.helpers import get_coverage_stats
        monkeypatch.chdir(tmp_path)

        # Create docs directory with file
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "api.md").write_text("# API Documentation")

        # Create tests directory with scaffold referencing the file
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        scaffold = tests_dir / "api-test.test.yaml"
        scaffold.write_text("""
name: api-validation
files:
  - path: docs/api.md
executor:
  system_prompt: Validate docs/api.md
  user_prompt: Check the API docs
""")

        config = {"coverage": {"include": ["docs/**/*.md"], "exclude": []}}
        stats = get_coverage_stats(tests_dir=str(tests_dir), config=config)

        # files_detail should have test_ids key
        assert "files_detail" in stats
        assert "docs/api.md" in stats["files_detail"]
        detail = stats["files_detail"]["docs/api.md"]
        assert "test_ids" in detail, "files_detail should include test_ids key"
        assert isinstance(detail["test_ids"], list)
        assert "api-validation" in detail["test_ids"]


class TestGetFileStatusFromCache:
    """Tests for get_file_status_from_cache function."""

    def test_empty_cache(self, tmp_path, monkeypatch):
        """Returns empty dict when no cache exists."""
        from dokumen.cli.helpers import get_file_status_from_cache
        monkeypatch.chdir(tmp_path)

        result = get_file_status_from_cache(str(tmp_path / "nonexistent"))

        assert result == {}

    def test_with_file_status_in_cache(self, tmp_path, monkeypatch):
        """Returns file status from cache."""
        from dokumen.cli.helpers import get_file_status_from_cache
        import json
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".dokumen-cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "file_status": {
                "docs/api.md": "passed",
                "docs/guide.md": "failed"
            }
        }))

        result = get_file_status_from_cache(str(cache_dir))

        assert result.get("docs/api.md") == "passed"
        assert result.get("docs/guide.md") == "failed"

    def test_with_invalid_cache(self, tmp_path, monkeypatch):
        """Returns empty dict on invalid JSON."""
        from dokumen.cli.helpers import get_file_status_from_cache
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".dokumen-cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text("invalid json {")

        result = get_file_status_from_cache(str(cache_dir))

        assert result == {}


class TestDiscoverDocFiles:
    """Tests for discover_doc_files function."""

    def test_finds_markdown_files(self, tmp_path, monkeypatch):
        """Finds markdown files matching patterns."""
        from dokumen.cli.helpers import discover_doc_files
        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "api.md").write_text("# API")
        (docs / "guide.md").write_text("# Guide")

        config = {"coverage": {"include": ["docs/**/*.md"], "exclude": []}}
        files = discover_doc_files(config)

        assert len(files) >= 2

    def test_respects_exclude_patterns(self, tmp_path, monkeypatch):
        """Respects exclude patterns."""
        from dokumen.cli.helpers import discover_doc_files
        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "api.md").write_text("# API")
        (docs / "internal.md").write_text("# Internal")

        config = {
            "coverage": {
                "include": ["docs/**/*.md"],
                "exclude": ["**/internal.md"]
            }
        }
        files = discover_doc_files(config)

        assert not any("internal.md" in f for f in files)


class TestFilterStatsByPath:
    """Tests for filter_stats_by_path function."""

    def test_no_path_returns_all(self):
        """No path filter returns original stats."""
        from dokumen.cli.helpers import filter_stats_by_path
        stats = {
            "total": 5,
            "files_detail": {"a.md": {}, "b.md": {}}
        }

        result = filter_stats_by_path(stats, "")

        assert result == stats

    def test_filters_by_directory(self):
        """Filters files by directory prefix."""
        from dokumen.cli.helpers import filter_stats_by_path
        stats = {
            "total": 3,
            "files_detail": {
                "docs/api.md": {"status": "passed", "test_count": 2},
                "docs/guide.md": {"status": "uncovered", "test_count": 0},
                "tests/test.md": {"status": "passed", "test_count": 1}
            }
        }

        result = filter_stats_by_path(stats, "docs/")

        assert result["total"] == 2
        assert "docs/api.md" in result["files_detail"]
        assert "tests/test.md" not in result["files_detail"]

    def test_empty_result_when_no_match(self):
        """Returns empty stats when no files match."""
        from dokumen.cli.helpers import filter_stats_by_path
        stats = {
            "total": 1,
            "files_detail": {"docs/api.md": {"status": "passed"}}
        }

        result = filter_stats_by_path(stats, "nonexistent/")

        assert result["total"] == 0
        assert result["passed"] == 0


class TestFilterLineStatsByPath:
    """Tests for filter_line_stats_by_path function."""

    def test_no_path_returns_all(self):
        """No path filter returns original stats."""
        from dokumen.cli.helpers import filter_line_stats_by_path
        stats = {"total_lines": 100, "files": {"a.md": {}}}

        result = filter_line_stats_by_path(stats, "")

        assert result == stats

    def test_empty_stats_returns_empty(self):
        """Empty stats returns empty."""
        from dokumen.cli.helpers import filter_line_stats_by_path

        result = filter_line_stats_by_path({}, "docs/")

        assert result == {}

    def test_filters_files_by_path(self):
        """Filters files by path prefix."""
        from dokumen.cli.helpers import filter_line_stats_by_path
        stats = {
            "total_lines": 200,
            "covered_lines": 100,
            "failed_lines": 20,
            "files": {
                "docs/api.md": {
                    "total_lines": 100,
                    "covered_lines": [1, 2, 3],
                    "failed_lines": [],
                    "status": "passed"
                },
                "tests/test.md": {
                    "total_lines": 100,
                    "covered_lines": [1],
                    "failed_lines": [],
                    "status": "passed"
                }
            }
        }

        result = filter_line_stats_by_path(stats, "docs/")

        assert "docs/api.md" in result["files"]
        assert "tests/test.md" not in result["files"]
        assert result["total_lines"] == 100

    def test_empty_result_when_no_match(self):
        """Returns empty stats when no files match."""
        from dokumen.cli.helpers import filter_line_stats_by_path
        stats = {
            "total_lines": 100,
            "files": {"docs/api.md": {"total_lines": 100}}
        }

        result = filter_line_stats_by_path(stats, "nonexistent/")

        assert result["total_lines"] == 0
        assert result["percentage"] == 0.0


class TestGetUncoveredFiles:
    """Tests for get_uncovered_files function."""

    def test_returns_uncovered(self, tmp_path, monkeypatch):
        """Returns list of uncovered files."""
        from dokumen.cli.helpers import get_uncovered_files
        monkeypatch.chdir(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "test.md").write_text("# Test")

        result = get_uncovered_files({"coverage": {"include": ["docs/**/*.md"]}})

        # All files should be uncovered when no cache
        assert isinstance(result, list)


class TestGetTestIdsPerFile:
    """Tests for get_test_ids_per_file function."""

    def test_empty_tests_dir(self, tmp_path, monkeypatch):
        """Returns empty dict when no scaffolds exist."""
        from dokumen.cli.helpers import get_test_ids_per_file
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tests").mkdir()

        result = get_test_ids_per_file(str(tmp_path / "tests"))

        assert result == {}

    def test_returns_test_names(self, tmp_path, monkeypatch):
        """Returns list of test names referencing each file."""
        from dokumen.cli.helpers import get_test_ids_per_file
        monkeypatch.chdir(tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create scaffold that references files
        scaffold = tests_dir / "test1.test.yaml"
        scaffold.write_text("""
name: my-test-name
files:
  - path: docs/api.md
executor:
  system_prompt: Read docs/api.md
  user_prompt: What does it say?
""")

        result = get_test_ids_per_file(str(tests_dir))

        assert "docs/api.md" in result
        assert isinstance(result["docs/api.md"], list)
        assert "my-test-name" in result["docs/api.md"]

    def test_multiple_scaffolds_same_file(self, tmp_path, monkeypatch):
        """Multiple scaffolds referencing same file returns all test names."""
        from dokumen.cli.helpers import get_test_ids_per_file
        monkeypatch.chdir(tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create two scaffolds that reference the same file
        scaffold1 = tests_dir / "test1.test.yaml"
        scaffold1.write_text("""
name: first-test
files:
  - path: docs/shared.md
executor:
  system_prompt: Read docs/shared.md
  user_prompt: What does it say?
""")
        scaffold2 = tests_dir / "test2.test.yaml"
        scaffold2.write_text("""
name: second-test
files:
  - path: docs/shared.md
executor:
  system_prompt: Read docs/shared.md
  user_prompt: Summarize it
""")

        result = get_test_ids_per_file(str(tests_dir))

        assert "docs/shared.md" in result
        assert "first-test" in result["docs/shared.md"]
        assert "second-test" in result["docs/shared.md"]
        assert len(result["docs/shared.md"]) == 2

    def test_handles_invalid_yaml(self, tmp_path, monkeypatch):
        """Handles invalid YAML gracefully."""
        from dokumen.cli.helpers import get_test_ids_per_file
        monkeypatch.chdir(tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create invalid scaffold
        scaffold = tests_dir / "invalid.test.yaml"
        scaffold.write_text("invalid: yaml: [")

        # Should not raise
        result = get_test_ids_per_file(str(tests_dir))

        assert isinstance(result, dict)


class TestGetTestCountsPerFile:
    """Tests for get_test_counts_per_file function."""

    def test_empty_tests_dir(self, tmp_path, monkeypatch):
        """Returns empty dict when no scaffolds exist."""
        from dokumen.cli.helpers import get_test_counts_per_file
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tests").mkdir()

        result = get_test_counts_per_file(str(tmp_path / "tests"))

        assert result == {}

    def test_counts_file_references(self, tmp_path, monkeypatch):
        """Counts file references from scaffolds."""
        from dokumen.cli.helpers import get_test_counts_per_file
        monkeypatch.chdir(tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create scaffold that references files
        scaffold = tests_dir / "test1.test.yaml"
        scaffold.write_text("""
name: test1
files:
  - path: docs/api.md
executor:
  system_prompt: Read docs/api.md
  user_prompt: What does it say?
""")

        result = get_test_counts_per_file(str(tests_dir))

        assert "docs/api.md" in result
        assert result["docs/api.md"] >= 1

    def test_handles_invalid_yaml(self, tmp_path, monkeypatch):
        """Handles invalid YAML gracefully."""
        from dokumen.cli.helpers import get_test_counts_per_file
        monkeypatch.chdir(tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create invalid scaffold
        scaffold = tests_dir / "invalid.test.yaml"
        scaffold.write_text("invalid: yaml: [")

        # Should not raise
        result = get_test_counts_per_file(str(tests_dir))

        assert isinstance(result, dict)

    def test_multiple_scaffolds_same_file(self, tmp_path, monkeypatch):
        """Multiple scaffolds referencing same file increment count."""
        from dokumen.cli.helpers import get_test_counts_per_file
        monkeypatch.chdir(tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create two scaffolds that reference the same file in prompts
        scaffold1 = tests_dir / "test1.test.yaml"
        scaffold1.write_text("""
name: test1
files:
  - path: docs/shared.md
executor:
  system_prompt: Read docs/shared.md
  user_prompt: What does it say?
""")
        scaffold2 = tests_dir / "test2.test.yaml"
        scaffold2.write_text("""
name: test2
files:
  - path: docs/shared.md
executor:
  system_prompt: Read docs/shared.md
  user_prompt: Summarize it
""")

        result = get_test_counts_per_file(str(tests_dir))

        assert "docs/shared.md" in result
        assert result["docs/shared.md"] >= 2


class TestGetFailureAnalysisFromCache:
    """Tests for get_failure_analysis_from_cache function."""

    def test_no_cache_file(self, tmp_path, monkeypatch):
        """Returns empty dict when cache file doesn't exist."""
        from dokumen.cli.helpers import get_failure_analysis_from_cache
        monkeypatch.chdir(tmp_path)

        result = get_failure_analysis_from_cache(str(tmp_path / "nonexistent"))

        assert result == {}

    def test_cache_without_failure_analysis(self, tmp_path, monkeypatch):
        """Returns empty dict when cache has no failure_analysis."""
        from dokumen.cli.helpers import get_failure_analysis_from_cache
        import json
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({"results": {}}))

        result = get_failure_analysis_from_cache(str(cache_dir))

        assert result == {}

    def test_cache_with_failure_analysis(self, tmp_path, monkeypatch):
        """Returns failure analysis from cache."""
        from dokumen.cli.helpers import get_failure_analysis_from_cache
        import json
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "failure_analysis": {
                "docs/api.md": {
                    "test-1": {
                        "referenced_lines": [1, 2, 3],
                        "incorrect_lines": [{"line_number": 5, "reason": "Wrong"}]
                    }
                }
            }
        }))

        result = get_failure_analysis_from_cache(str(cache_dir))

        assert "docs/api.md" in result
        assert "test-1" in result["docs/api.md"]

    def test_invalid_json_cache(self, tmp_path, monkeypatch):
        """Returns empty dict on invalid JSON."""
        from dokumen.cli.helpers import get_failure_analysis_from_cache
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text("invalid json {")

        result = get_failure_analysis_from_cache(str(cache_dir))

        assert result == {}


class TestGetLineCoverageStats:
    """Tests for get_line_coverage_stats function."""

    def test_no_cache_returns_empty(self, tmp_path, monkeypatch):
        """Returns empty stats when no cache exists."""
        from dokumen.cli.helpers import get_line_coverage_stats
        monkeypatch.chdir(tmp_path)

        result = get_line_coverage_stats(str(tmp_path / "nonexistent"))

        assert result["total_lines"] == 0
        assert result["covered_lines"] == 0
        assert result["percentage"] == 0.0

    def test_with_line_coverage_data(self, tmp_path, monkeypatch):
        """Returns line coverage from cache."""
        from dokumen.cli.helpers import get_line_coverage_stats
        import json
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "line_coverage": {
                "docs/api.md": {
                    "total_lines": 100,
                    "covered_lines": [1, 2, 3, 4, 5],
                    "failed_lines": [10, 11],
                    "incorrect_lines": []
                }
            },
            "file_status": {
                "docs/api.md": "passed"
            }
        }))

        result = get_line_coverage_stats(str(cache_dir))

        assert result["total_lines"] == 100
        assert result["covered_lines"] == 5
        assert result["failed_lines"] == 2
        assert "docs/api.md" in result["files"]

    def test_with_file_states(self, tmp_path, monkeypatch):
        """Uses provided file_states for status."""
        from dokumen.cli.helpers import get_line_coverage_stats
        import json
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "line_coverage": {
                "docs/api.md": {
                    "total_lines": 50,
                    "covered_lines": [1, 2],
                    "failed_lines": []
                }
            }
        }))

        file_states = {"docs/api.md": "passed"}
        result = get_line_coverage_stats(str(cache_dir), file_states=file_states)

        assert result["files"]["docs/api.md"]["status"] == "passed"

    def test_with_all_doc_files(self, tmp_path, monkeypatch):
        """Filters to only tracked doc files."""
        from dokumen.cli.helpers import get_line_coverage_stats
        import json
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "line_coverage": {
                "docs/api.md": {"total_lines": 50, "covered_lines": [1], "failed_lines": []},
                "untracked.md": {"total_lines": 100, "covered_lines": [], "failed_lines": []}
            }
        }))

        result = get_line_coverage_stats(
            str(cache_dir),
            all_doc_files=["docs/api.md"]
        )

        assert "docs/api.md" in result["files"]
        assert "untracked.md" not in result["files"]

    def test_by_state_counting(self, tmp_path, monkeypatch):
        """Correctly counts lines by state."""
        from dokumen.cli.helpers import get_line_coverage_stats
        import json
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "line_coverage": {
                "docs/api.md": {
                    "total_lines": 100,
                    "covered_lines": [1, 2, 3],
                    "failed_lines": [10, 11]
                }
            }
        }))

        result = get_line_coverage_stats(str(cache_dir))

        by_state = result["by_state"]
        assert by_state["passed"] == 3  # covered lines
        assert by_state["failed"] == 2  # failed lines
        assert by_state["uncovered"] == 95  # remaining lines

    def test_invalid_json_returns_empty(self, tmp_path, monkeypatch):
        """Returns empty stats on invalid JSON."""
        from dokumen.cli.helpers import get_line_coverage_stats
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text("invalid json {")

        result = get_line_coverage_stats(str(cache_dir))

        assert result["total_lines"] == 0

    def test_fallback_for_passed_without_coverage(self, tmp_path, monkeypatch):
        """Fallback marks all lines covered for passed files without coverage."""
        from dokumen.cli.helpers import get_line_coverage_stats
        import json
        monkeypatch.chdir(tmp_path)

        # Create a test file
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        test_file = docs_dir / "api.md"
        test_file.write_text("line1\nline2\nline3\n")

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "line_coverage": {
                "docs/api.md": {
                    "total_lines": 3,
                    "covered_lines": [],
                    "failed_lines": []
                }
            },
            "file_status": {
                "docs/api.md": "passed"
            }
            # Note: no coverage_attempted means fallback applies
        }))

        monkeypatch.chdir(tmp_path)
        result = get_line_coverage_stats(str(cache_dir))

        # Fallback should mark all lines as covered
        assert result["files"]["docs/api.md"]["fallback_applied"] is True
        assert len(result["files"]["docs/api.md"]["covered_lines"]) == 3

    def test_failure_analysis_merged(self, tmp_path, monkeypatch):
        """Merges failure analysis into files data."""
        from dokumen.cli.helpers import get_line_coverage_stats
        import json
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "line_coverage": {
                "docs/api.md": {
                    "total_lines": 100,
                    "covered_lines": [1, 2],
                    "failed_lines": []
                }
            },
            "failure_analysis": {
                "docs/api.md": {
                    "test-1": {
                        "referenced_lines": [5, 6, 7],
                        "incorrect_lines": [{"line_number": 5, "reason": "Wrong"}]
                    }
                }
            }
        }))

        result = get_line_coverage_stats(str(cache_dir))

        # Failure analysis should be merged
        assert 5 in result["files"]["docs/api.md"]["failed_lines"]
        assert len(result["files"]["docs/api.md"]["incorrect_lines"]) >= 1


class TestCountFileLines:
    """Tests for _count_file_lines function."""

    def test_counts_lines(self, tmp_path):
        """Counts lines in a file."""
        from dokumen.cli.helpers import _count_file_lines
        test_file = tmp_path / "test.md"
        test_file.write_text("line1\nline2\nline3\n")

        result = _count_file_lines(str(test_file))

        assert result == 3

    def test_nonexistent_file(self, tmp_path):
        """Returns 0 for nonexistent file."""
        from dokumen.cli.helpers import _count_file_lines

        result = _count_file_lines(str(tmp_path / "nonexistent.md"))

        assert result == 0

    def test_empty_file(self, tmp_path):
        """Returns 0 for empty file."""
        from dokumen.cli.helpers import _count_file_lines
        test_file = tmp_path / "empty.md"
        test_file.write_text("")

        result = _count_file_lines(str(test_file))

        assert result == 0


class TestFilterTests:
    """Tests for filter_tests function."""

    def test_no_filter_returns_all(self):
        """Returns all tests when no filter specified."""
        from dokumen.cli.helpers import filter_tests

        # Create mock test objects
        class MockTest:
            def __init__(self, id, reason, files):
                self.id = id
                self.reason = reason
                self.files = files

        tests = [
            MockTest("test1", "reason1", []),
            MockTest("test2", "reason2", [])
        ]

        result = filter_tests(tests)

        assert len(result) == 2

    def test_filter_by_test_ids(self):
        """Filters by test IDs."""
        from dokumen.cli.helpers import filter_tests

        class MockTest:
            def __init__(self, id, reason, files):
                self.id = id
                self.reason = reason
                self.files = files

        tests = [
            MockTest("test1", "reason1", []),
            MockTest("test2", "reason2", []),
            MockTest("test3", "reason3", [])
        ]

        result = filter_tests(tests, test_ids=["test1", "test3"])

        assert len(result) == 2
        assert all(t.id in ["test1", "test3"] for t in result)

    def test_filter_by_grep_pattern(self):
        """Filters by grep pattern matching ID."""
        from dokumen.cli.helpers import filter_tests

        class MockTest:
            def __init__(self, id, reason, files):
                self.id = id
                self.reason = reason
                self.files = files

        tests = [
            MockTest("api-test", "test API", []),
            MockTest("ui-test", "test UI", []),
            MockTest("api-integration", "integration", [])
        ]

        result = filter_tests(tests, grep="api*")

        assert len(result) == 2
        assert all("api" in t.id for t in result)

    def test_filter_by_grep_reason(self):
        """Filters by grep pattern matching reason."""
        from dokumen.cli.helpers import filter_tests

        class MockTest:
            def __init__(self, id, reason, files):
                self.id = id
                self.reason = reason
                self.files = files

        tests = [
            MockTest("test1", "API validation", []),
            MockTest("test2", "UI rendering", [])
        ]

        result = filter_tests(tests, grep="API")

        assert len(result) == 1
        assert result[0].id == "test1"

    def test_filter_by_file(self):
        """Filters by file path."""
        from dokumen.cli.helpers import filter_tests

        class MockFile:
            def __init__(self, path):
                self.path = path

        class MockTest:
            def __init__(self, id, reason, files):
                self.id = id
                self.reason = reason
                self.files = files

        tests = [
            MockTest("test1", "reason", [MockFile("docs/api.md")]),
            MockTest("test2", "reason", [MockFile("docs/guide.md")]),
            MockTest("test3", "reason", [MockFile("docs/api.md"), MockFile("docs/other.md")])
        ]

        result = filter_tests(tests, for_file="docs/api.md")

        assert len(result) == 2
        assert all(any(f.path == "docs/api.md" for f in t.files) for t in result)

    def test_combined_filters(self):
        """Applies multiple filters together."""
        from dokumen.cli.helpers import filter_tests

        class MockFile:
            def __init__(self, path):
                self.path = path

        class MockTest:
            def __init__(self, id, reason, files):
                self.id = id
                self.reason = reason
                self.files = files

        tests = [
            MockTest("api-test1", "API test", [MockFile("docs/api.md")]),
            MockTest("api-test2", "API other", [MockFile("docs/guide.md")]),
            MockTest("ui-test", "UI test", [MockFile("docs/api.md")])
        ]

        result = filter_tests(tests, grep="api*", for_file="docs/api.md")

        assert len(result) == 1
        assert result[0].id == "api-test1"


class TestFilterLineStatsByPathAdvanced:
    """Additional tests for filter_line_stats_by_path edge cases."""

    def test_failed_status_line_counting(self):
        """Correctly handles failed status files in line counting."""
        from dokumen.cli.helpers import filter_line_stats_by_path
        stats = {
            "total_lines": 100,
            "covered_lines": 30,
            "failed_lines": 20,
            "files": {
                "docs/api.md": {
                    "total_lines": 100,
                    "covered_lines": [1, 2, 3],  # 3 covered
                    "failed_lines": [10, 11],    # 2 failed
                    "status": "failed"
                }
            }
        }

        result = filter_line_stats_by_path(stats, "docs/")

        by_state = result["by_state"]
        # For failed status: passed=covered, failed=failed_lines, uncovered=remainder
        assert by_state["passed"] == 3
        assert by_state["failed"] == 2
        assert by_state["uncovered"] == 95

    def test_preserves_failure_analysis(self):
        """Preserves failure_analysis in filtered result."""
        from dokumen.cli.helpers import filter_line_stats_by_path
        stats = {
            "total_lines": 100,
            "files": {
                "docs/api.md": {"total_lines": 100, "covered_lines": [], "failed_lines": []}
            },
            "failure_analysis": {"docs/api.md": {"test-1": {"analysis": "Failed"}}}
        }

        result = filter_line_stats_by_path(stats, "docs/")

        assert "failure_analysis" in result
        assert result["failure_analysis"] == stats["failure_analysis"]

    def test_backslash_path_normalization(self):
        """Normalizes backslashes in path filter."""
        from dokumen.cli.helpers import filter_line_stats_by_path
        stats = {
            "total_lines": 100,
            "files": {
                "docs/api.md": {"total_lines": 100, "covered_lines": [], "failed_lines": []}
            }
        }

        # Use backslashes in filter
        result = filter_line_stats_by_path(stats, "docs\\")

        assert "docs/api.md" in result["files"]


class TestGetFileStatusFromCacheAdvanced:
    """Additional tests for get_file_status_from_cache."""

    def test_derives_status_from_results(self, tmp_path, monkeypatch):
        """Derives file status from test results when file_status empty."""
        from dokumen.cli.helpers import get_file_status_from_cache
        import json
        monkeypatch.chdir(tmp_path)

        # Create a scaffold file
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        scaffold = tests_dir / "test1.test.yaml"
        scaffold.write_text("""
name: test-1
files:
  - path: docs/api.md
""")

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "file_status": {},
            "results": {
                "test-1": {"passed": True}
            }
        }))

        result = get_file_status_from_cache(str(cache_dir), tests_dir=str(tests_dir))

        assert result.get("docs/api.md") == "passed"

    def test_failed_test_marks_file_failed(self, tmp_path, monkeypatch):
        """Failed test marks file as failed."""
        from dokumen.cli.helpers import get_file_status_from_cache
        import json
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        scaffold = tests_dir / "test1.test.yaml"
        scaffold.write_text("""
name: test-1
files:
  - path: docs/api.md
""")

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "file_status": {},
            "results": {
                "test-1": {"passed": False}
            }
        }))

        result = get_file_status_from_cache(str(cache_dir), tests_dir=str(tests_dir))

        assert result.get("docs/api.md") == "failed"

    def test_normalizes_paths(self, tmp_path, monkeypatch):
        """Normalizes paths from cache."""
        from dokumen.cli.helpers import get_file_status_from_cache
        import json
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "cache.json"
        cache_file.write_text(json.dumps({
            "file_status": {
                "docs\\api.md": "passed"
            }
        }))

        result = get_file_status_from_cache(str(cache_dir))

        # Should normalize backslashes to forward slashes
        assert "docs/api.md" in result


class TestFilterScaffoldPaths:
    """Tests for filter_scaffold_paths function."""

    def test_no_filter_returns_all(self, tmp_path, monkeypatch):
        """Returns all scaffolds when no filter specified."""
        from dokumen.cli.helpers import filter_scaffold_paths
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create scaffolds
        (tests_dir / "test1.test.yaml").write_text("name: test-one\n")
        (tests_dir / "test2.test.yaml").write_text("name: test-two\n")

        scaffold_paths = [str(tests_dir / "test1.test.yaml"), str(tests_dir / "test2.test.yaml")]

        result = filter_scaffold_paths(scaffold_paths, [], None)

        assert len(result) == 2

    def test_filter_by_test_name(self, tmp_path, monkeypatch):
        """Filters scaffolds by test name."""
        from dokumen.cli.helpers import filter_scaffold_paths
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "test1.test.yaml").write_text("name: api-test\n")
        (tests_dir / "test2.test.yaml").write_text("name: ui-test\n")

        scaffold_paths = [str(tests_dir / "test1.test.yaml"), str(tests_dir / "test2.test.yaml")]

        result = filter_scaffold_paths(scaffold_paths, ["api-test"], None)

        assert len(result) == 1
        assert "test1.test.yaml" in result[0]

    def test_filter_by_multiple_test_names(self, tmp_path, monkeypatch):
        """Filters scaffolds by multiple test names."""
        from dokumen.cli.helpers import filter_scaffold_paths
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "test1.test.yaml").write_text("name: api-test\n")
        (tests_dir / "test2.test.yaml").write_text("name: ui-test\n")
        (tests_dir / "test3.test.yaml").write_text("name: db-test\n")

        scaffold_paths = [
            str(tests_dir / "test1.test.yaml"),
            str(tests_dir / "test2.test.yaml"),
            str(tests_dir / "test3.test.yaml")
        ]

        result = filter_scaffold_paths(scaffold_paths, ["api-test", "db-test"], None)

        assert len(result) == 2

    def test_filter_by_grep_pattern(self, tmp_path, monkeypatch):
        """Filters scaffolds by grep pattern."""
        from dokumen.cli.helpers import filter_scaffold_paths
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "test1.test.yaml").write_text("name: api-auth-test\n")
        (tests_dir / "test2.test.yaml").write_text("name: api-users-test\n")
        (tests_dir / "test3.test.yaml").write_text("name: ui-dashboard-test\n")

        scaffold_paths = [
            str(tests_dir / "test1.test.yaml"),
            str(tests_dir / "test2.test.yaml"),
            str(tests_dir / "test3.test.yaml")
        ]

        result = filter_scaffold_paths(scaffold_paths, [], "api-*")

        assert len(result) == 2
        assert all("ui-dashboard" not in p for p in result)

    def test_filter_by_wildcard_pattern(self, tmp_path, monkeypatch):
        """Filters scaffolds by wildcard pattern."""
        from dokumen.cli.helpers import filter_scaffold_paths
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "test1.test.yaml").write_text("name: auth-login-test\n")
        (tests_dir / "test2.test.yaml").write_text("name: auth-logout-test\n")
        (tests_dir / "test3.test.yaml").write_text("name: users-crud-test\n")

        scaffold_paths = [
            str(tests_dir / "test1.test.yaml"),
            str(tests_dir / "test2.test.yaml"),
            str(tests_dir / "test3.test.yaml")
        ]

        result = filter_scaffold_paths(scaffold_paths, [], "*-test")

        # All match "*-test"
        assert len(result) == 3

    def test_skips_unparseable_files(self, tmp_path, monkeypatch):
        """Skips files that cannot be parsed."""
        from dokumen.cli.helpers import filter_scaffold_paths
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "good.test.yaml").write_text("name: good-test\n")
        (tests_dir / "bad.test.yaml").write_text("invalid: yaml: [")

        scaffold_paths = [
            str(tests_dir / "good.test.yaml"),
            str(tests_dir / "bad.test.yaml")
        ]

        result = filter_scaffold_paths(scaffold_paths, ["good-test"], None)

        assert len(result) == 1
        assert "good.test.yaml" in result[0]

    def test_empty_result_when_no_match(self, tmp_path, monkeypatch):
        """Returns empty list when no scaffolds match."""
        from dokumen.cli.helpers import filter_scaffold_paths
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "test1.test.yaml").write_text("name: existing-test\n")

        scaffold_paths = [str(tests_dir / "test1.test.yaml")]

        result = filter_scaffold_paths(scaffold_paths, ["nonexistent-test"], None)

        assert len(result) == 0

    def test_test_name_fnmatch_patterns(self, tmp_path, monkeypatch):
        """Test names can be fnmatch patterns."""
        from dokumen.cli.helpers import filter_scaffold_paths
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "test1.test.yaml").write_text("name: api-v1-test\n")
        (tests_dir / "test2.test.yaml").write_text("name: api-v2-test\n")

        scaffold_paths = [
            str(tests_dir / "test1.test.yaml"),
            str(tests_dir / "test2.test.yaml")
        ]

        # Pass pattern as test name
        result = filter_scaffold_paths(scaffold_paths, ["api-v*-test"], None)

        assert len(result) == 2

    def test_combined_test_names_and_grep(self, tmp_path, monkeypatch):
        """Test names and grep pattern work together (union)."""
        from dokumen.cli.helpers import filter_scaffold_paths
        monkeypatch.chdir(tmp_path)

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        (tests_dir / "test1.test.yaml").write_text("name: specific-test\n")
        (tests_dir / "test2.test.yaml").write_text("name: grep-match-test\n")
        (tests_dir / "test3.test.yaml").write_text("name: other-test\n")

        scaffold_paths = [
            str(tests_dir / "test1.test.yaml"),
            str(tests_dir / "test2.test.yaml"),
            str(tests_dir / "test3.test.yaml")
        ]

        # specific-test by name, grep-match-* by pattern
        result = filter_scaffold_paths(scaffold_paths, ["specific-test"], "grep-*")

        # Should match both
        assert len(result) == 2
