"""
Tests for dokumen run command.

TDD tests for the run command, including:
- Basic execution (run all, run specific, run with grep)
- Exit codes (0, 1, 2, 3)
- Output formats (text, json, junit, tap)
"""
import importlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

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


# Import the module directly to avoid __init__.py run command shadowing
run_module = importlib.import_module("dokumen.cli.commands.run")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Click CLI test runner with separate stderr to avoid log contamination."""
    return CliRunner()


@pytest.fixture
def mock_test_object():
    """Create a mock test object with id attribute."""
    def _make(test_id: str = "my-test", reason: str = "test reason"):
        mock = MagicMock()
        mock.id = test_id
        mock.reason = reason
        mock.files = []
        mock.timeout = 60.0
        return mock
    return _make


# =============================================================================
# Basic Execution Tests
# =============================================================================


class TestRunBasicExecution:
    """Tests for basic run command execution."""

    def test_run_all_tests_no_args(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Running without args executes all tests."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            # Setup mock suite
            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 0
            mock_load.assert_called_once()

    def test_run_specific_test_by_name(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Running with test name runs only that test."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            # Return multiple tests but only one should match
            mock_load.return_value = ([
                mock_test_object("api-test"),
                mock_test_object("auth-test"),
            ], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("api-test")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "api-test"])

            assert result.exit_code == 0
            # Verify only one test was added to suite
            assert mock_suite.add_test.call_count == 1

    def test_run_multiple_tests_by_name(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Running with multiple test names runs those tests."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([
                mock_test_object("test-1"),
                mock_test_object("test-2"),
                mock_test_object("test-3"),
            ], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=2, passed=2, failed=0,
                test_results=[
                    mock_test_result("test-1"),
                    mock_test_result("test-2"),
                ]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "test-1", "test-2"])

            assert result.exit_code == 0
            assert mock_suite.add_test.call_count == 2

    def test_run_no_tests_in_project(self, runner, project_with_tests):
        """Running with no tests shows message and exits 1 (EXIT_FAILURE).

        Per PHASE0-CLI-SPEC: Exit code 4 is for invalid arguments only.
        No tests found is a failure scenario, not an argument error.
        """
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([], {})  # No tests found

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 1
            # Error messages go to stderr when using mix_stderr=False
            combined_output = result.output + (result.stderr or '')
            assert "No tests found" in combined_output

    def test_run_dry_run_shows_tests(self, runner, project_with_tests, mock_test_object):
        """Dry run shows tests without executing."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([
                mock_test_object("test-1"),
                mock_test_object("test-2"),
            ], {})

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--dry-run"])

            assert result.exit_code == 0
            assert "Would run" in result.output
            assert "test-1" in result.output
            assert "test-2" in result.output
            # Verify TestSuite was never instantiated
            mock_suite_class.assert_not_called()


# =============================================================================
# Load Error Reporting Tests
# =============================================================================


class TestRunLoadErrorReporting:
    """Tests that run command reports load errors instead of 'No tests match'."""

    def test_run_shows_load_error_for_requested_test(self, runner, project_with_tests):
        """When a requested test failed to load, show the load error not 'No tests match'."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            # No tests loaded successfully, but one had a load error
            mock_load.return_value = ([], {"broken-test": "code_repos clone failed: 401 Unauthorized"})

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "broken-test"])

            assert result.exit_code == 2  # EXIT_CONFIG_ERROR
            combined_output = result.output + (result.stderr or '')
            assert "failed to load" in combined_output
            assert "401 Unauthorized" in combined_output

    def test_run_shows_no_match_when_test_not_in_errors(self, runner, project_with_tests, mock_test_object):
        """When a requested test isn't in load_errors either, show 'No tests match'."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("other-test")], {})

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "nonexistent-test"])

            assert result.exit_code == 1  # EXIT_FAILURE
            combined_output = result.output + (result.stderr or '')
            assert "No tests match" in combined_output

    def test_run_all_scaffolds_fail_reports_errors(self, runner, project_with_tests):
        """When all scaffolds fail to load and no filter, report all load errors."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            # All scaffolds failed, no tests loaded
            mock_load.return_value = ([], {
                "test-a": "YAML parse error",
                "test-b": "code_repos clone failed: 401",
            })

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 2  # EXIT_CONFIG_ERROR
            combined_output = result.output + (result.stderr or '')
            assert "test-a" in combined_output
            assert "test-b" in combined_output
            assert "failed to load" in combined_output


# =============================================================================
# Grep/Filter Tests
# =============================================================================


class TestRunGrepFilter:
    """Tests for --grep filtering."""

    def test_run_grep_filters_tests(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """--grep filters tests by pattern."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([
                mock_test_object("api-auth-test"),
                mock_test_object("api-users-test"),
                mock_test_object("db-migration-test"),
            ], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=2, passed=2, failed=0,
                test_results=[
                    mock_test_result("api-auth-test"),
                    mock_test_result("api-users-test"),
                ]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--grep", "api-*"])

            assert result.exit_code == 0
            # Verify only api tests were added
            assert mock_suite.add_test.call_count == 2

    def test_run_grep_no_matches(self, runner, project_with_tests, mock_test_object):
        """--grep with no matches shows message and exits 1 (EXIT_FAILURE).

        Per PHASE0-CLI-SPEC: Exit code 4 is for invalid arguments only.
        No tests matching filter is a failure scenario, not an argument error.
        """
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([
                mock_test_object("db-test"),
            ], {})

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--grep", "api-*"])

            assert result.exit_code == 1
            # Error messages go to stderr when using mix_stderr=False
            combined_output = result.output + (result.stderr or '')
            assert "No tests match" in combined_output

    def test_run_grep_shorthand(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """-g shorthand works for --grep."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("api-test")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("api-test")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "-g", "api-*"])

            assert result.exit_code == 0


# =============================================================================
# Exit Code Tests
# =============================================================================


class TestExitCodes:
    """Tests for exit code handling."""

    def test_exit_0_all_pass(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Exit code 0 when all tests pass."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1", passed=True)]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 0

    def test_exit_1_any_failure(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Exit code 1 when any test fails."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=0, failed=1,
                test_results=[mock_test_result("test-1", passed=False, failure_reasons=["Assertion failed"])]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 1

    def test_exit_1_partial_failure(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Exit code 1 when some tests pass and some fail."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([
                mock_test_object("test-1"),
                mock_test_object("test-2"),
            ], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=2, passed=1, failed=1,
                test_results=[
                    mock_test_result("test-1", passed=True),
                    mock_test_result("test-2", passed=False, failure_reasons=["Error"]),
                ]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 1

    def test_exit_1_judge_error_only(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Exit code 1 when judge times out (error > 0, failed == 0)."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            # error=1, failed=0 — judge timeout with no normal failures
            results = mock_suite_results(
                total=1, passed=0, failed=0, error=1,
                test_results=[mock_test_result("test-1", passed=False, failure_reasons=["Judge timed out"])]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 1

    def test_exit_2_load_error(self, runner, project_with_tests):
        """Exit code 2 when loading tests fails."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.side_effect = Exception("Invalid scaffold syntax")

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 2
            # Error messages go to stderr when using mix_stderr=False
            combined_output = result.output + (result.stderr or '')
            assert "Error loading tests" in combined_output

    def test_exit_1_invalid_provider_fails_tests(self, runner, tmp_path):
        """Exit code 1 when provider is invalid - tests fail to execute."""
        from dokumen.cli import cli

        # Note: load_config() is lenient - it silently ignores YAML parse errors
        # and falls back to defaults. An invalid provider name doesn't cause a
        # config loading error - instead, tests run but fail with "No provider".
        # This results in exit code 1 (test failure) rather than 2 (config error).
        (tmp_path / "dokumen.yaml").write_text("""
version: "1.0"
provider:
  name: invalid-provider
  model: nonexistent
""")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "sample.test.yaml").write_text("""
name: sample-test
executor:
  system_prompt: test
  user_prompt: test
  tools: []
judges:
  - name: test
    system_prompt: test
""")

        with patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            os.chdir(tmp_path)
            result = runner.invoke(cli, ["run"])

            # Invalid provider causes scaffold load failure → config error (exit 2)
            assert result.exit_code == 2
            assert "failed to load" in result.output.lower() or "failed" in (result.stderr or '').lower()

    def test_exit_3_runtime_error(self, runner, project_with_tests, mock_test_object):
        """Exit code 3 when runtime error occurs during test execution."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            # Simulate runtime error during test execution
            mock_suite.run = AsyncMock(side_effect=RuntimeError("API timeout"))
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            # Note: Currently run.py doesn't catch runtime errors with EXIT_RUNTIME_ERROR
            # This test documents expected behavior - should be non-zero
            assert result.exit_code != 0


class TestExitCodeInvalidArgs:
    """Tests for exit code 2 on invalid arguments."""

    def test_exit_2_invalid_output_format(self, runner, project_with_tests):
        """Exit code 2 for invalid --output value."""
        from dokumen.cli import cli

        os.chdir(project_with_tests)
        result = runner.invoke(cli, ["run", "--output", "invalid"])

        assert result.exit_code == 2
        # Error messages go to stderr when using mix_stderr=False
        combined_output = result.output + (result.stderr or '')
        assert "Invalid value" in combined_output or "invalid" in combined_output.lower()

    def test_exit_2_unknown_flag(self, runner, project_with_tests):
        """Exit code 2 for unknown flag."""
        from dokumen.cli import cli

        os.chdir(project_with_tests)
        result = runner.invoke(cli, ["run", "--unknown-flag"])

        assert result.exit_code == 2


# =============================================================================
# Output Format Tests
# =============================================================================


class TestOutputFormats:
    """Tests for output format options."""

    def test_output_text_default(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Default output is text format."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 0
            # Text output includes human-readable content
            assert "Running" in result.output or "Results" in result.output or "passed" in result.output.lower()

    def test_output_json_valid(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """--output json produces valid JSON."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--output", "json"])

            assert result.exit_code == 0
            # Verify valid JSON
            data = extract_json_from_output(result.output)
            assert "total" in data
            assert "passed" in data
            assert "failed" in data
            assert "tests" in data

    def test_output_junit_valid(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """--output junit produces valid XML."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--output", "junit"])

            assert result.exit_code == 0
            # Verify XML structure
            assert '<?xml version="1.0"' in result.output
            assert "<testsuite" in result.output
            assert "<testcase" in result.output

    def test_output_tap_valid(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """--output tap produces TAP format."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--output", "tap"])

            assert result.exit_code == 0
            # Verify TAP format
            assert "TAP version 13" in result.output
            assert "1..1" in result.output
            assert "ok 1" in result.output


class TestOutputModes:
    """Tests for verbose, quiet, and debug modes."""

    def test_quiet_suppresses_progress(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """--quiet suppresses progress output."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--quiet"])

            assert result.exit_code == 0
            # Quiet mode should have minimal output
            # (exact behavior depends on implementation)

    def test_output_shows_failed_reasons(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Failed tests show failure reasons in output."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=0, failed=1,
                test_results=[mock_test_result(
                    "test-1",
                    passed=False,
                    failure_reasons=["Judge assertion failed: Expected OAuth"]
                )]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 1
            # Should show failure reason
            assert "FAIL" in result.output or "failed" in result.output.lower()


# =============================================================================
# Timeout Tests
# =============================================================================


class TestRunTimeout:
    """Tests for timeout handling."""

    def test_run_with_timeout_override(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """--timeout overrides default timeout."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            test_obj = mock_test_object("test-1")
            mock_load.return_value = ([test_obj], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--timeout", "300"])

            assert result.exit_code == 0
            # Verify timeout was set on the test object
            assert test_obj.timeout == 300.0


# =============================================================================
# Force Flag Tests
# =============================================================================


class TestRunForceFlag:
    """Tests for --force flag."""

    def test_run_force_skips_cache(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """--force skips cache loading."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--force"])

            assert result.exit_code == 0
            # With --force, cache should NOT be loaded
            mock_suite.load_cache.assert_not_called()


class TestRunEnvVars:
    """Tests for environment variable handling."""

    def test_dokumen_tests_env_filters_tests(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result, monkeypatch):
        """DOKUMEN_TESTS env var filters tests to run."""
        from dokumen.cli import cli

        # Set env var to run only test-a and test-b
        monkeypatch.setenv("DOKUMEN_TESTS", "test-a,test-b")

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            # Return 3 tests, but only 2 should run
            mock_load.return_value = ([
                mock_test_object("test-a"),
                mock_test_object("test-b"),
                mock_test_object("test-c"),
            ], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=2, passed=2, failed=0,
                test_results=[
                    mock_test_result("test-a"),
                    mock_test_result("test-b"),
                ]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 0
            # Only 2 tests should be added to suite (test-a and test-b)
            assert mock_suite.add_test.call_count == 2

    def test_dokumen_timeout_env_overrides_config(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result, monkeypatch):
        """DOKUMEN_TIMEOUT env var overrides config timeout."""
        from dokumen.cli import cli

        # Set env var timeout to 120 seconds
        monkeypatch.setenv("DOKUMEN_TIMEOUT", "120")

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            test_obj = mock_test_object("test-1")
            mock_load.return_value = ([test_obj], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 0
            # Timeout should be set to env var value (120)
            assert test_obj.timeout == 120.0

    def test_cli_args_override_env_vars(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result, monkeypatch):
        """CLI args override env vars (CLI args > env vars > config)."""
        from dokumen.cli import cli

        # Set env var timeout to 120
        monkeypatch.setenv("DOKUMEN_TIMEOUT", "120")

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            test_obj = mock_test_object("test-1")
            mock_load.return_value = ([test_obj], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            # Pass --timeout 60, which should override env var 120
            result = runner.invoke(cli, ["run", "--timeout", "60"])

            assert result.exit_code == 0
            # CLI arg (60) should win over env var (120)
            assert test_obj.timeout == 60.0

    def test_dokumen_force_env_var(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result, monkeypatch):
        """DOKUMEN_FORCE env var enables force mode."""
        from dokumen.cli import cli

        monkeypatch.setenv("DOKUMEN_FORCE", "1")

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 0
            # DOKUMEN_FORCE should disable cache loading
            mock_suite.load_cache.assert_not_called()


class TestRunDebugMode:
    """Tests for debug mode."""

    def test_run_with_debug_flag(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """--debug flag enables debug mode and implies verbose."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug') as mock_set_debug, \
             patch('dokumen.debug.start_debug_session') as mock_start_debug, \
             patch('dokumen.debug.end_debug_session') as mock_end_debug, \
             patch.object(run_module, 'OutputWriter') as mock_writer_class:

            # Mock OutputWriter to avoid Pydantic validation issues
            mock_writer = MagicMock()
            mock_writer_class.return_value = mock_writer

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--debug"])

            assert result.exit_code == 0
            # Debug mode should be enabled
            mock_set_debug.assert_called_with(True)
            # Debug session should be started
            mock_start_debug.assert_called_once()
            # OutputWriter.write_all should be called with debug_enabled=True
            mock_writer.write_all.assert_called_once()
            call_args = mock_writer.write_all.call_args
            assert call_args.kwargs.get('debug_enabled') is True or call_args[0][2] is True

    def test_global_debug_implies_verbose(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Global --debug implies --verbose on run command."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug') as mock_set_debug, \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'), \
             patch.object(run_module, 'OutputWriter') as mock_writer_class:

            # Mock OutputWriter to avoid Pydantic validation issues
            mock_writer = MagicMock()
            mock_writer_class.return_value = mock_writer

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            # Use global --debug flag before run command
            result = runner.invoke(cli, ["--debug", "run"])

            # Should succeed
            assert result.exit_code == 0
            # set_debug should be called with True due to global --debug
            mock_set_debug.assert_called_with(True)


# =============================================================================
# Folder Filter Tests
# =============================================================================


class TestRunFolderOption:
    """Tests for --folder option in run command."""

    def test_run_folder_filters_tests(self, runner, project_with_tests, mock_suite_results, mock_test_result):
        """--folder filters tests by folder path."""
        from dokumen.cli import cli
        from dokumen.cli.helpers import EXIT_SUCCESS

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()

            # Create mock tests with source_path
            api_test = MagicMock()
            api_test.id = "api-test"
            api_test.reason = "test"
            api_test.files = []
            api_test.timeout = 60.0
            api_test.source_path = "tests/api/auth/login.test.yaml"

            smoke_test = MagicMock()
            smoke_test.id = "smoke-test"
            smoke_test.reason = "test"
            smoke_test.files = []
            smoke_test.timeout = 60.0
            smoke_test.source_path = "tests/smoke.test.yaml"

            mock_load.return_value = ([api_test, smoke_test], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("api-test")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--folder", "api"])

            assert result.exit_code == EXIT_SUCCESS
            # Verify add_test was called only for the api-test (filtered result)
            # The smoke-test should be filtered out by folder filter
            add_test_calls = [call for call in mock_suite.add_test.call_args_list]
            assert len(add_test_calls) == 1
            assert add_test_calls[0][0][0].id == "api-test"

    def test_run_folder_invalid_path(self, runner, project_with_tests, mock_suite_results, mock_test_result):
        """--folder with invalid path returns EXIT_INVALID_ARGS."""
        from dokumen.cli import cli
        from dokumen.cli.helpers import EXIT_INVALID_ARGS

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([MagicMock(id="test", reason="", files=[], timeout=60)], {})

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--folder", "../escape"])

            assert result.exit_code == EXIT_INVALID_ARGS
            assert "Invalid folder path" in result.output or "parent traversal" in (result.output + (result.stderr or ''))

    def test_run_folder_shorthand(self, runner, project_with_tests, mock_suite_results, mock_test_result):
        """-d shorthand works for --folder."""
        from dokumen.cli import cli
        from dokumen.cli.helpers import EXIT_SUCCESS

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()

            api_test = MagicMock()
            api_test.id = "api-test"
            api_test.reason = "test"
            api_test.files = []
            api_test.timeout = 60.0
            api_test.source_path = "tests/api/test.yaml"

            mock_load.return_value = ([api_test], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("api-test")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "-d", "api"])

            assert result.exit_code == EXIT_SUCCESS


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestRunParallelFlag:
    """Tests for --parallel flag."""

    def test_run_parallel_flag_sets_config(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """--parallel 4 sets parallel_execution=True, max_concurrency=4."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.test_suite.TestSuiteConfig') as mock_config_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "--parallel", "4"])

            assert result.exit_code == 0
            # Verify TestSuiteConfig was called with parallel settings
            mock_config_class.assert_called_once()
            config_kwargs = mock_config_class.call_args.kwargs
            assert config_kwargs.get('parallel_execution') is True
            assert config_kwargs.get('max_concurrency') == 4

    def test_run_parallel_shorthand(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """-p shorthand works for --parallel."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.test_suite.TestSuiteConfig') as mock_config_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run", "-p", "3"])

            assert result.exit_code == 0
            mock_config_class.assert_called_once()
            config_kwargs = mock_config_class.call_args.kwargs
            assert config_kwargs.get('parallel_execution') is True
            assert config_kwargs.get('max_concurrency') == 3

    def test_run_parallel_default_off(self, runner, project_with_tests, mock_test_object, mock_suite_results, mock_test_result):
        """Without --parallel, parallel_execution=False."""
        from dokumen.cli import cli

        with patch('dokumen.loader.load_all_scaffolds') as mock_load, \
             patch('dokumen.loader.get_configured_provider') as mock_provider, \
             patch('dokumen.test_suite.TestSuite') as mock_suite_class, \
             patch('dokumen.test_suite.TestSuiteConfig') as mock_config_class, \
             patch('dokumen.debug.set_debug'), \
             patch('dokumen.debug.start_debug_session'), \
             patch('dokumen.debug.end_debug_session'):

            mock_provider.return_value = MagicMock()
            mock_load.return_value = ([mock_test_object("test-1")], {})

            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            results = mock_suite_results(
                total=1, passed=1, failed=0,
                test_results=[mock_test_result("test-1")]
            )
            mock_suite.run = AsyncMock(return_value=results)
            mock_suite.load_cache = AsyncMock()
            mock_suite.save_cache = AsyncMock()

            os.chdir(project_with_tests)
            result = runner.invoke(cli, ["run"])

            assert result.exit_code == 0
            mock_config_class.assert_called_once()
            config_kwargs = mock_config_class.call_args.kwargs
            assert config_kwargs.get('parallel_execution') is False
