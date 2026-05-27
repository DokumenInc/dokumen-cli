"""
Tests for the `dokumen list tests` command.

TDD: These tests are written first, before implementation.
"""
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner


def extract_json_from_output(output: str) -> dict:
    """Extract JSON from CLI output, ignoring log lines."""
    # Find the first '{' which starts JSON
    start_idx = output.find('{')
    if start_idx == -1:
        return json.loads(output)

    # Try to parse from the start position
    decoder = json.JSONDecoder()
    try:
        result, end_idx = decoder.raw_decode(output[start_idx:])
        return result
    except json.JSONDecodeError:
        return json.loads(output)


# Import the module directly to avoid __init__.py list_cmd command shadowing
list_cmd_module = importlib.import_module("dokumen.cli.commands.list_cmd")


@pytest.fixture
def runner():
    """Click CLI test runner with separate stderr to avoid log contamination."""
    return CliRunner()


@pytest.fixture
def project_with_tests(tmp_path: Path, valid_config_path: Path, valid_minimal_scaffold_path: Path, valid_complete_scaffold_path: Path):
    """Create project with config and multiple test scaffolds."""
    # Copy config
    config = tmp_path / "dokumen.yaml"
    config.write_text(valid_config_path.read_text())

    # Create tests directory with scaffolds
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    # Add minimal scaffold
    scaffold1 = tests_dir / "minimal.test.yaml"
    scaffold1.write_text(valid_minimal_scaffold_path.read_text())

    # Add complete scaffold
    scaffold2 = tests_dir / "complete.test.yaml"
    scaffold2.write_text(valid_complete_scaffold_path.read_text())

    return tmp_path


@pytest.fixture
def empty_project(tmp_path: Path, valid_config_path: Path):
    """Create project with config but no test scaffolds."""
    config = tmp_path / "dokumen.yaml"
    config.write_text(valid_config_path.read_text())

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    return tmp_path


@pytest.fixture
def no_config_project(tmp_path: Path):
    """Create project without dokumen.yaml."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    return tmp_path


class TestListTestsSuccess:
    """Tests for successful list tests scenarios."""

    def test_list_tests_shows_all_tests(self, runner: CliRunner, project_with_tests: Path):
        """List all tests shows their names."""
        from dokumen.cli import cli

        os.chdir(project_with_tests)
        result = runner.invoke(cli, ["list", "tests"])

        assert result.exit_code == 0
        # Should show the test names from the scaffolds
        assert "minimal-test" in result.output or "my-test" in result.output
        assert "Tests" in result.output or "test" in result.output.lower()

    def test_list_tests_empty_project(self, runner: CliRunner, empty_project: Path):
        """List tests with no scaffolds shows appropriate message."""
        from dokumen.cli import cli

        os.chdir(empty_project)
        result = runner.invoke(cli, ["list", "tests"])

        assert result.exit_code == 0
        assert "no test scaffolds" in result.output.lower() or "no tests" in result.output.lower()


class TestListTestsJson:
    """Tests for --json output format."""

    def test_list_tests_json_valid(self, runner: CliRunner, project_with_tests: Path):
        """--json returns valid JSON."""
        from dokumen.cli import cli

        os.chdir(project_with_tests)
        result = runner.invoke(cli, ["list", "tests", "--json"])

        assert result.exit_code == 0
        data = extract_json_from_output(result.output)
        assert "tests" in data

    def test_list_tests_json_includes_metadata(self, runner: CliRunner, project_with_tests: Path):
        """JSON includes test name, file path, and reason."""
        from dokumen.cli import cli

        os.chdir(project_with_tests)
        result = runner.invoke(cli, ["list", "tests", "--json"])

        assert result.exit_code == 0
        data = extract_json_from_output(result.output)
        assert "tests" in data
        assert len(data["tests"]) >= 1

        # Check first test has required fields
        test = data["tests"][0]
        assert "name" in test
        assert "file" in test

    def test_list_tests_json_empty_project(self, runner: CliRunner, empty_project: Path):
        """JSON with no tests returns empty array."""
        from dokumen.cli import cli

        os.chdir(empty_project)
        result = runner.invoke(cli, ["list", "tests", "--json"])

        assert result.exit_code == 0
        data = extract_json_from_output(result.output)
        assert "tests" in data
        assert data["tests"] == []


class TestListTestsVerbose:
    """Tests for verbose output."""

    def test_list_tests_verbose_shows_details(self, runner: CliRunner, project_with_tests: Path):
        """Verbose mode shows additional details."""
        from dokumen.cli import cli

        os.chdir(project_with_tests)
        result = runner.invoke(cli, ["list", "tests", "--verbose"])

        assert result.exit_code == 0
        # Should show file path and potentially reason
        assert "Path:" in result.output or ".test.yaml" in result.output


class TestListTestsErrors:
    """Tests for error handling in list tests."""

    def test_list_tests_handles_invalid_yaml(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """list tests handles invalid YAML gracefully."""
        from dokumen.cli import cli

        # Create project with invalid YAML scaffold
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        invalid_scaffold = tests_dir / "invalid.test.yaml"
        invalid_scaffold.write_text("invalid: yaml: [")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["list", "tests"])

        # Should not crash, may show error message
        assert result.exit_code == 0

    def test_list_tests_json_handles_invalid_yaml(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """list tests --json handles invalid YAML gracefully."""
        from dokumen.cli import cli

        # Create project with invalid YAML scaffold
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        invalid_scaffold = tests_dir / "invalid.test.yaml"
        invalid_scaffold.write_text("invalid: yaml: [")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["list", "tests", "--json"])

        assert result.exit_code == 0
        data = extract_json_from_output(result.output)
        assert "tests" in data
        # Invalid file should have error field
        if data["tests"]:
            assert any("error" in t for t in data["tests"])


class TestListFilesCommand:
    """Tests for the list files command."""

    def test_list_files_basic(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path, monkeypatch):
        """list files shows tracked files."""
        from dokumen.cli import cli
        from unittest.mock import patch

        # Create project
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "api.md").write_text("# API")

        monkeypatch.chdir(tmp_path)

        with patch.object(list_cmd_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 1,
                "passed": 0,
                "percentage": 0.0,
                "covered_files": [],
                "failed_files": [],
                "uncovered_files": ["docs/api.md"],
            }

            result = runner.invoke(cli, ["list", "files"])

            assert result.exit_code == 0

    def test_list_files_empty(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path, monkeypatch):
        """list files with no files shows appropriate message."""
        from dokumen.cli import cli
        from unittest.mock import patch

        # Create project without docs
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        monkeypatch.chdir(tmp_path)

        with patch.object(list_cmd_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 0,
                "passed": 0,
                "percentage": 0.0,
                "covered_files": [],
                "failed_files": [],
                "uncovered_files": [],
            }

            result = runner.invoke(cli, ["list", "files"])

            assert result.exit_code == 0
            assert "no documentation files" in result.output.lower() or "dokumen.yaml" in result.output.lower()

    def test_list_files_with_metrics(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path, monkeypatch):
        """list files --metrics shows coverage percentage."""
        from dokumen.cli import cli
        from unittest.mock import patch

        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        monkeypatch.chdir(tmp_path)

        with patch.object(list_cmd_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 2,
                "passed": 1,
                "percentage": 50.0,
                "covered_files": ["docs/api.md"],
                "failed_files": [],
                "uncovered_files": ["docs/guide.md"],
            }

            result = runner.invoke(cli, ["list", "files", "--metrics"])

            assert result.exit_code == 0
            # Should show percentage
            assert "50%" in result.output or "Summary" in result.output

    def test_list_files_high_coverage_color(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path, monkeypatch):
        """list files --metrics shows green for high coverage."""
        from dokumen.cli import cli
        from unittest.mock import patch

        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        monkeypatch.chdir(tmp_path)

        with patch.object(list_cmd_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 9,
                "percentage": 90.0,
                "covered_files": ["docs/api.md"],
                "failed_files": [],
                "uncovered_files": [],
            }

            result = runner.invoke(cli, ["list", "files", "--metrics"])

            assert result.exit_code == 0
            # Should show high percentage
            assert "90%" in result.output

    def test_list_files_low_coverage_color(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path, monkeypatch):
        """list files --metrics shows red for low coverage."""
        from dokumen.cli import cli
        from unittest.mock import patch

        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        monkeypatch.chdir(tmp_path)

        with patch.object(list_cmd_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 3,
                "percentage": 30.0,
                "covered_files": [],
                "failed_files": [],
                "uncovered_files": ["a.md", "b.md"],
            }

            result = runner.invoke(cli, ["list", "files", "--metrics"])

            assert result.exit_code == 0


class TestListToolsCommand:
    """Tests for the list tools command."""

    def test_list_tools_shows_builtins(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path, monkeypatch):
        """list tools shows built-in tools."""
        from dokumen.cli import cli

        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        monkeypatch.chdir(tmp_path)

        result = runner.invoke(cli, ["list", "tools"])

        assert result.exit_code == 0
        assert "Available Tools" in result.output or "Built-in" in result.output

    def test_list_tools_shows_tool_names(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path, monkeypatch):
        """list tools shows tool names and descriptions."""
        from dokumen.cli import cli

        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        monkeypatch.chdir(tmp_path)

        result = runner.invoke(cli, ["list", "tools"])

        assert result.exit_code == 0
        # Should list some built-in tools
        output_lower = result.output.lower()
        assert "read" in output_lower or "list" in output_lower or "file" in output_lower


class TestListTestsTreeOutput:
    """Tests for --tree output format."""

    def test_list_tests_tree_groups_by_folder(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """--tree groups tests by folder."""
        from dokumen.cli import cli

        # Create project with tests in folders
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create test in api subfolder
        api_dir = tests_dir / "api"
        api_dir.mkdir()
        (api_dir / "auth.test.yaml").write_text("name: auth-test\nreason: test\nfiles:\n  - docs/api.md")

        # Create test in root
        (tests_dir / "smoke.test.yaml").write_text("name: smoke-test\nreason: test\nfiles: []")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["list", "tests", "--tree"])

        assert result.exit_code == 0
        assert "api/" in result.output
        assert "(root)" in result.output
        assert "auth-test" in result.output
        assert "smoke-test" in result.output

    def test_list_tests_tree_shows_file_count(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """--tree shows file count for each test."""
        from dokumen.cli import cli

        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "my.test.yaml").write_text("name: my-test\nreason: test\nfiles:\n  - a.md\n  - b.md")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["list", "tests", "--tree"])

        assert result.exit_code == 0
        assert "2 file(s)" in result.output

    def test_list_tests_tree_verbose(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """--tree --verbose shows full details."""
        from dokumen.cli import cli

        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "verbose.test.yaml").write_text("name: verbose-test\nreason: test\nfiles:\n  - a.md")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["list", "tests", "--tree", "--verbose"])

        assert result.exit_code == 0
        assert "verbose-test" in result.output
        assert "Path:" in result.output
        assert "Files:" in result.output

    def test_list_tests_tree_skips_invalid_yaml(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """--tree silently skips invalid YAML files."""
        from dokumen.cli import cli

        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "valid.test.yaml").write_text("name: valid-test\nreason: test\nfiles: []")
        (tests_dir / "invalid.test.yaml").write_text("invalid: yaml: [")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["list", "tests", "--tree"])

        assert result.exit_code == 0
        assert "valid-test" in result.output
        # Invalid file should be silently skipped

    def test_list_tests_tree_sorts_alphabetically(self, runner: CliRunner, tmp_path: Path, valid_config_path: Path):
        """--tree sorts folders and tests alphabetically."""
        from dokumen.cli import cli

        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create folders in non-alphabetical order
        (tests_dir / "zebra").mkdir()
        (tests_dir / "alpha").mkdir()
        (tests_dir / "zebra" / "z.test.yaml").write_text("name: z-test\nreason: test\nfiles: []")
        (tests_dir / "alpha" / "a.test.yaml").write_text("name: a-test\nreason: test\nfiles: []")

        os.chdir(tmp_path)
        result = runner.invoke(cli, ["list", "tests", "--tree"])

        assert result.exit_code == 0
        # alpha/ should appear before zebra/
        alpha_pos = result.output.find("alpha/")
        zebra_pos = result.output.find("zebra/")
        assert alpha_pos < zebra_pos


class TestListCmdGroup:
    """Tests for the list command group."""

    def test_list_cmd_help(self, runner: CliRunner):
        """list --help shows subcommands."""
        from dokumen.cli import cli

        result = runner.invoke(cli, ["list", "--help"])

        assert result.exit_code == 0
        assert "tests" in result.output
        assert "files" in result.output
        assert "tools" in result.output

    def test_list_cmd_without_subcommand(self, runner: CliRunner):
        """list without subcommand shows help."""
        from dokumen.cli import cli

        result = runner.invoke(cli, ["list"])

        # Should show help or error about missing subcommand
        assert result.exit_code == 0 or result.exit_code == 2
