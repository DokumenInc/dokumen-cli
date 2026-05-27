"""Tests for coverage_parser module."""

import json
import pytest
from pathlib import Path

from dokumen.cli.coverage_parser import (
    CommandCoverage,
    parse_coverage_json,
    _convert_status_json,
    map_module_to_command,
    extract_cli_coverage,
    format_coverage_prompt_injection,
)


class TestCommandCoverage:
    """Tests for CommandCoverage dataclass."""

    def test_create_command_coverage(self):
        """CommandCoverage should store all fields."""
        cc = CommandCoverage(
            command="dokumen run",
            module_path="dokumen/cli/commands/run.py",
            percent_covered=85.5,
            covered_lines=100,
            total_lines=117,
        )
        assert cc.command == "dokumen run"
        assert cc.module_path == "dokumen/cli/commands/run.py"
        assert cc.percent_covered == 85.5
        assert cc.covered_lines == 100
        assert cc.total_lines == 117


class TestParseCoverageJson:
    """Tests for parse_coverage_json function."""

    def test_parse_standard_format(self, tmp_path):
        """Should parse standard coverage.json format."""
        coverage_data = {
            "files": {
                "dokumen/cli/commands/run.py": {
                    "summary": {
                        "percent_covered": 92.2,
                        "covered_lines": 71,
                        "num_statements": 77,
                    }
                }
            }
        }
        coverage_file = tmp_path / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))

        result = parse_coverage_json(str(coverage_file))

        assert "dokumen/cli/commands/run.py" in result
        assert result["dokumen/cli/commands/run.py"]["summary"]["percent_covered"] == 92.2

    def test_parse_htmlcov_status_format(self, tmp_path):
        """Should parse htmlcov/status.json format and convert it."""
        status_data = {
            "files": {
                "z_hash_run_py": {
                    "index": {
                        "file": "dokumen\\cli\\commands\\run.py",
                        "nums": {"n_statements": 77, "n_missing": 6},
                    }
                }
            }
        }
        status_file = tmp_path / "status.json"
        status_file.write_text(json.dumps(status_data))

        result = parse_coverage_json(str(status_file))

        # Path should be normalized (backslash to forward slash)
        assert "dokumen/cli/commands/run.py" in result
        summary = result["dokumen/cli/commands/run.py"]["summary"]
        assert summary["num_statements"] == 77
        assert summary["covered_lines"] == 71  # 77 - 6
        assert summary["missing_lines"] == 6

    def test_parse_missing_file_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_coverage_json("/nonexistent/coverage.json")

    def test_parse_invalid_json_raises(self, tmp_path):
        """Should raise JSONDecodeError for invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            parse_coverage_json(str(bad_file))

    def test_parse_missing_files_key_raises(self, tmp_path):
        """Should raise ValueError if 'files' key is missing."""
        no_files = tmp_path / "no_files.json"
        no_files.write_text('{"meta": {}, "totals": {}}')

        with pytest.raises(ValueError, match="missing 'files' key"):
            parse_coverage_json(str(no_files))

    def test_parse_empty_files(self, tmp_path):
        """Should handle empty files dict."""
        empty_files = tmp_path / "empty.json"
        empty_files.write_text('{"files": {}}')

        result = parse_coverage_json(str(empty_files))
        assert result == {}


class TestConvertStatusJson:
    """Tests for _convert_status_json function."""

    def test_convert_basic(self):
        """Should convert single file entry."""
        files = {
            "z_hash_run_py": {
                "index": {
                    "file": "dokumen/cli/commands/run.py",
                    "nums": {"n_statements": 100, "n_missing": 10},
                }
            }
        }

        result = _convert_status_json(files)

        assert "dokumen/cli/commands/run.py" in result
        summary = result["dokumen/cli/commands/run.py"]["summary"]
        assert summary["percent_covered"] == 90.0
        assert summary["covered_lines"] == 90
        assert summary["num_statements"] == 100
        assert summary["missing_lines"] == 10

    def test_convert_multiple_files(self):
        """Should convert multiple file entries."""
        files = {
            "hash1": {
                "index": {
                    "file": "dokumen/cli/commands/run.py",
                    "nums": {"n_statements": 100, "n_missing": 0},
                }
            },
            "hash2": {
                "index": {
                    "file": "dokumen/cli/commands/list_cmd.py",
                    "nums": {"n_statements": 50, "n_missing": 25},
                }
            },
        }

        result = _convert_status_json(files)

        assert len(result) == 2
        assert result["dokumen/cli/commands/run.py"]["summary"]["percent_covered"] == 100.0
        assert result["dokumen/cli/commands/list_cmd.py"]["summary"]["percent_covered"] == 50.0

    def test_convert_path_normalization(self):
        """Should normalize backslashes to forward slashes."""
        files = {
            "hash": {
                "index": {
                    "file": "dokumen\\cli\\commands\\run.py",
                    "nums": {"n_statements": 100, "n_missing": 0},
                }
            }
        }

        result = _convert_status_json(files)

        assert "dokumen/cli/commands/run.py" in result
        assert "dokumen\\cli\\commands\\run.py" not in result

    def test_convert_missing_file_path(self):
        """Should skip entries with empty file path."""
        files = {
            "hash1": {
                "index": {
                    "file": "",
                    "nums": {"n_statements": 100, "n_missing": 0},
                }
            },
            "hash2": {
                "index": {
                    "file": "valid/path.py",
                    "nums": {"n_statements": 50, "n_missing": 0},
                }
            },
        }

        result = _convert_status_json(files)

        assert len(result) == 1
        assert "valid/path.py" in result

    def test_convert_zero_statements(self):
        """Should handle zero statements without division error."""
        files = {
            "hash": {
                "index": {
                    "file": "empty.py",
                    "nums": {"n_statements": 0, "n_missing": 0},
                }
            }
        }

        result = _convert_status_json(files)

        assert result["empty.py"]["summary"]["percent_covered"] == 0.0

    def test_convert_missing_nums(self):
        """Should handle missing nums with defaults."""
        files = {
            "hash": {
                "index": {
                    "file": "file.py",
                    "nums": {},
                }
            }
        }

        result = _convert_status_json(files)

        summary = result["file.py"]["summary"]
        assert summary["num_statements"] == 0
        assert summary["covered_lines"] == 0
        assert summary["percent_covered"] == 0.0


class TestMapModuleToCommand:
    """Tests for map_module_to_command function."""

    def test_map_run_command(self):
        """Should map run.py to 'dokumen run'."""
        result = map_module_to_command("dokumen/cli/commands/run.py")
        assert result == "dokumen run"

    def test_map_validate_command(self):
        """Should map validate.py to 'dokumen validate'."""
        result = map_module_to_command("dokumen/cli/commands/validate.py")
        assert result == "dokumen validate"

    def test_map_coverage_command(self):
        """Should map coverage.py to 'dokumen coverage'."""
        result = map_module_to_command("dokumen/cli/commands/coverage.py")
        assert result == "dokumen coverage"

    def test_map_list_cmd_special_case(self):
        """Should map list_cmd.py to 'dokumen list' (special case)."""
        result = map_module_to_command("dokumen/cli/commands/list_cmd.py")
        assert result == "dokumen list"

    def test_map_init_returns_none(self):
        """Should return None for __init__.py files."""
        result = map_module_to_command("dokumen/cli/commands/__init__.py")
        assert result is None

    def test_map_non_command_module_returns_none(self):
        """Should return None for non-command modules."""
        assert map_module_to_command("dokumen/cli/helpers.py") is None
        assert map_module_to_command("dokumen/agent_object.py") is None
        assert map_module_to_command("dokumen/providers/anthropic.py") is None

    def test_map_with_backslash_path(self):
        """Should handle Windows-style paths."""
        result = map_module_to_command("dokumen\\cli\\commands\\run.py")
        assert result == "dokumen run"


class TestExtractCliCoverage:
    """Tests for extract_cli_coverage function."""

    def test_extract_all_commands(self):
        """Should extract all CLI commands from coverage data."""
        coverage_data = {
            "dokumen/cli/commands/run.py": {
                "summary": {"percent_covered": 90.0, "covered_lines": 90, "num_statements": 100}
            },
            "dokumen/cli/commands/list_cmd.py": {
                "summary": {"percent_covered": 60.0, "covered_lines": 30, "num_statements": 50}
            },
            "dokumen/agent_object.py": {  # Not a command, should be filtered
                "summary": {"percent_covered": 80.0, "covered_lines": 80, "num_statements": 100}
            },
        }

        all_cmds, low_cmds = extract_cli_coverage(coverage_data)

        assert len(all_cmds) == 2
        commands = [c.command for c in all_cmds]
        assert "dokumen run" in commands
        assert "dokumen list" in commands

    def test_extract_filters_by_threshold(self):
        """Should filter commands below threshold."""
        coverage_data = {
            "dokumen/cli/commands/run.py": {
                "summary": {"percent_covered": 90.0, "covered_lines": 90, "num_statements": 100}
            },
            "dokumen/cli/commands/list_cmd.py": {
                "summary": {"percent_covered": 60.0, "covered_lines": 30, "num_statements": 50}
            },
        }

        _, low_cmds = extract_cli_coverage(coverage_data, threshold=70.0)

        assert len(low_cmds) == 1
        assert low_cmds[0].command == "dokumen list"
        assert low_cmds[0].percent_covered == 60.0

    def test_extract_sorts_by_coverage(self):
        """Should sort commands by coverage (lowest first)."""
        coverage_data = {
            "dokumen/cli/commands/run.py": {
                "summary": {"percent_covered": 90.0, "covered_lines": 90, "num_statements": 100}
            },
            "dokumen/cli/commands/list_cmd.py": {
                "summary": {"percent_covered": 60.0, "covered_lines": 30, "num_statements": 50}
            },
            "dokumen/cli/commands/validate.py": {
                "summary": {"percent_covered": 75.0, "covered_lines": 75, "num_statements": 100}
            },
        }

        all_cmds, _ = extract_cli_coverage(coverage_data)

        assert all_cmds[0].percent_covered == 60.0  # Lowest first
        assert all_cmds[1].percent_covered == 75.0
        assert all_cmds[2].percent_covered == 90.0  # Highest last

    def test_extract_empty_data(self):
        """Should handle empty coverage data."""
        all_cmds, low_cmds = extract_cli_coverage({})

        assert all_cmds == []
        assert low_cmds == []

    def test_extract_with_custom_threshold(self):
        """Should respect custom threshold."""
        coverage_data = {
            "dokumen/cli/commands/run.py": {
                "summary": {"percent_covered": 85.0, "covered_lines": 85, "num_statements": 100}
            },
        }

        # With default threshold (70%), run is above
        _, low_70 = extract_cli_coverage(coverage_data, threshold=70.0)
        assert len(low_70) == 0

        # With higher threshold (90%), run is below
        _, low_90 = extract_cli_coverage(coverage_data, threshold=90.0)
        assert len(low_90) == 1


class TestFormatCoveragePromptInjection:
    """Tests for format_coverage_prompt_injection function."""

    def test_format_empty_list(self):
        """Should return empty string for no low-coverage commands."""
        result = format_coverage_prompt_injection([])
        assert result == ""

    def test_format_single_command(self):
        """Should format single low-coverage command."""
        low_cmds = [
            CommandCoverage(
                command="dokumen run",
                module_path="dokumen/cli/commands/run.py",
                percent_covered=50.0,
                covered_lines=50,
                total_lines=100,
            )
        ]

        result = format_coverage_prompt_injection(low_cmds, threshold=70.0)

        assert "PRIORITY TESTING" in result
        assert "less than 70%" in result
        assert "dokumen run: 50.0% coverage" in result
        assert "(50/100 lines)" in result

    def test_format_multiple_commands(self):
        """Should format multiple low-coverage commands."""
        low_cmds = [
            CommandCoverage(
                command="dokumen run",
                module_path="dokumen/cli/commands/run.py",
                percent_covered=50.0,
                covered_lines=50,
                total_lines=100,
            ),
            CommandCoverage(
                command="dokumen list",
                module_path="dokumen/cli/commands/list_cmd.py",
                percent_covered=40.0,
                covered_lines=20,
                total_lines=50,
            ),
        ]

        result = format_coverage_prompt_injection(low_cmds)

        assert "dokumen run" in result
        assert "dokumen list" in result
        assert "50.0% coverage" in result
        assert "40.0% coverage" in result

    def test_format_includes_guidance(self):
        """Should include testing guidance text."""
        low_cmds = [
            CommandCoverage(
                command="dokumen run",
                module_path="dokumen/cli/commands/run.py",
                percent_covered=50.0,
                covered_lines=50,
                total_lines=100,
            )
        ]

        result = format_coverage_prompt_injection(low_cmds)

        assert "edge cases" in result
        assert "error conditions" in result
        assert "untested code paths" in result
