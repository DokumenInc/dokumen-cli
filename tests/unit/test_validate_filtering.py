"""
Tests for DOKUMEN_TESTS filtering - ensures validate and run commands
only process selected tests when the env var is set.

TDD: These tests verify that malformed unrelated tests don't cause
pipeline failures when running specific tests.
"""
import os
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mixed_valid_invalid_project(tmp_path: Path, valid_config_path: Path):
    """Create project with one valid and one malformed test scaffold."""
    config = tmp_path / "dokumen.yaml"
    config.write_text(valid_config_path.read_text())

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text("# Test\n\nTest content.")

    # Valid scaffold
    valid_scaffold = tests_dir / "valid-test.test.yaml"
    valid_scaffold.write_text("""
name: valid-test
reason: A valid test
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Validate the document
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate it
""")

    # Malformed scaffold (missing executor entirely)
    malformed_scaffold = tests_dir / "malformed-test.test.yaml"
    malformed_scaffold.write_text("""
name: malformed-test
reason: This test is broken on purpose
files:
  - path: docs/test.md
judges:
  - name: judge
    system_prompt: Evaluate it
""")

    return tmp_path


class TestValidateIgnoresMalformedWhenFiltered:
    """Validate only processes selected tests, ignoring malformed ones."""

    def test_validate_passes_when_dokumen_tests_selects_valid_test(
        self, runner: CliRunner, mixed_valid_invalid_project: Path, monkeypatch
    ):
        """Setting DOKUMEN_TESTS to a valid test skips malformed tests entirely."""
        from dokumen.cli import cli

        os.chdir(mixed_valid_invalid_project)
        monkeypatch.setenv("DOKUMEN_TESTS", "valid-test")
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0, f"Expected pass, got: {result.output}"

    def test_validate_fails_without_filtering(
        self, runner: CliRunner, mixed_valid_invalid_project: Path
    ):
        """Without DOKUMEN_TESTS, validate catches the malformed test."""
        from dokumen.cli import cli

        os.chdir(mixed_valid_invalid_project)
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 2, f"Expected failure, got: {result.output}"

    def test_validate_comma_separated_selects_only_listed(
        self, runner: CliRunner, mixed_valid_invalid_project: Path, monkeypatch
    ):
        """DOKUMEN_TESTS=valid-test selects only that test, not malformed-test."""
        from dokumen.cli import cli

        os.chdir(mixed_valid_invalid_project)
        # Only select the valid test via comma-separated list
        monkeypatch.setenv("DOKUMEN_TESTS", "valid-test")
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0, f"Expected pass, got: {result.output}"
        # Should not mention malformed-test at all
        assert "malformed-test" not in result.output


class TestValidateLogsFiltering:
    """Validate logs a message when DOKUMEN_TESTS filtering is active."""

    def test_validate_logs_filtering_message(
        self, runner: CliRunner, mixed_valid_invalid_project: Path, monkeypatch
    ):
        """Output contains filtering message when DOKUMEN_TESTS is set."""
        from dokumen.cli import cli

        os.chdir(mixed_valid_invalid_project)
        monkeypatch.setenv("DOKUMEN_TESTS", "valid-test")
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0
        assert "Filtering by DOKUMEN_TESTS" in result.output
        assert "1 test(s) selected" in result.output

    def test_validate_no_filtering_message_without_env(
        self, runner: CliRunner, mixed_valid_invalid_project: Path
    ):
        """No filtering message when DOKUMEN_TESTS is not set."""
        from dokumen.cli import cli

        os.chdir(mixed_valid_invalid_project)
        # Don't set DOKUMEN_TESTS - ensure it's unset
        os.environ.pop("DOKUMEN_TESTS", None)
        result = runner.invoke(cli, ["validate"])

        assert "Filtering by DOKUMEN_TESTS" not in result.output

    def test_validate_logs_count_for_multiple_tests(
        self, runner: CliRunner, mixed_valid_invalid_project: Path, monkeypatch
    ):
        """Filtering message shows correct count for multiple selected tests."""
        from dokumen.cli import cli

        # Add a second valid test
        tests_dir = mixed_valid_invalid_project / "tests"
        second_valid = tests_dir / "another-valid.test.yaml"
        second_valid.write_text("""
name: another-valid
reason: Another valid test
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Validate again
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate it
""")

        os.chdir(mixed_valid_invalid_project)
        monkeypatch.setenv("DOKUMEN_TESTS", "valid-test,another-valid")
        result = runner.invoke(cli, ["validate"])

        assert result.exit_code == 0
        assert "Filtering by DOKUMEN_TESTS" in result.output
        assert "2 test(s) selected" in result.output


class TestRunSkipsMalformedYaml:
    """Run command skips files with YAML syntax errors instead of crashing."""

    @pytest.fixture
    def yaml_error_project(self, tmp_path: Path, valid_config_path: Path):
        """Create project with a valid test and one with YAML syntax errors."""
        config = tmp_path / "dokumen.yaml"
        config.write_text(valid_config_path.read_text())

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\n\nTest content.")

        # Valid scaffold
        valid_scaffold = tests_dir / "valid-test.test.yaml"
        valid_scaffold.write_text("""
name: valid-test
reason: A valid test
files:
  - path: docs/test.md
executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: Validate the document
  tools:
    - read_file
judges:
  - name: judge
    system_prompt: Evaluate it
""")

        # YAML syntax error scaffold (invalid YAML, not just missing fields)
        broken_yaml = tests_dir / "broken-yaml.test.yaml"
        broken_yaml.write_text("""
name: broken-yaml
  bad_indent: this is invalid
? unexpected_key
  another: problem
""")

        return tmp_path

    def test_run_dry_run_skips_yaml_errors(
        self, runner: CliRunner, yaml_error_project: Path, monkeypatch
    ):
        """dokumen run --dry-run succeeds even when some YAML files are malformed."""
        from dokumen.cli import cli

        os.chdir(yaml_error_project)
        monkeypatch.setenv("DOKUMEN_TESTS", "valid-test")
        result = runner.invoke(cli, ["run", "--dry-run"])

        assert result.exit_code == 0, f"Expected pass, got: {result.output}"
        assert "valid-test" in result.output

    def test_load_all_scaffolds_skips_yaml_errors(self, yaml_error_project: Path):
        """load_all_scaffolds skips files with YAML syntax errors."""
        from dokumen.loader import load_all_scaffolds

        os.chdir(yaml_error_project)
        tests, load_errors = load_all_scaffolds(tests_dir="tests")

        # Should load the valid test and skip the broken YAML
        assert len(tests) == 1
        assert tests[0].id == "valid-test"
        # The broken YAML should appear in load_errors
        assert len(load_errors) > 0


class TestRunRespectsFiltering:
    """Run command respects DOKUMEN_TESTS with --dry-run."""

    def test_run_dry_run_only_lists_selected_test(
        self, runner: CliRunner, mixed_valid_invalid_project: Path, monkeypatch
    ):
        """dokumen run --dry-run with DOKUMEN_TESTS only lists selected tests."""
        from dokumen.cli import cli

        os.chdir(mixed_valid_invalid_project)
        monkeypatch.setenv("DOKUMEN_TESTS", "valid-test")
        result = runner.invoke(cli, ["run", "--dry-run"])

        assert result.exit_code == 0, f"Expected pass, got: {result.output}"
        assert "valid-test" in result.output
        # The "Would run" section should only list the selected test
        lines_after_would_run = []
        capture = False
        for line in result.output.splitlines():
            if "Would run" in line:
                capture = True
                continue
            if capture and line.strip().startswith("- "):
                lines_after_would_run.append(line.strip())
        assert len(lines_after_would_run) == 1
        assert "valid-test" in lines_after_would_run[0]

    def test_run_dry_run_shows_filtering_message(
        self, runner: CliRunner, mixed_valid_invalid_project: Path, monkeypatch
    ):
        """dokumen run --dry-run shows DOKUMEN_TESTS filtering message."""
        from dokumen.cli import cli

        os.chdir(mixed_valid_invalid_project)
        monkeypatch.setenv("DOKUMEN_TESTS", "valid-test")
        result = runner.invoke(cli, ["run", "--dry-run"])

        assert result.exit_code == 0
        assert "Filtering by DOKUMEN_TESTS" in result.output
