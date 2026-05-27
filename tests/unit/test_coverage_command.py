"""Tests for coverage and status CLI commands."""

import importlib
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

# Import the module directly to avoid __init__.py coverage command shadowing
coverage_module = importlib.import_module("dokumen.cli.commands.coverage")


class TestCoverageCommand:
    """Tests for the coverage command."""

    def test_coverage_command_exists(self):
        """coverage command exists."""
        from dokumen.cli.commands.coverage import coverage

        assert coverage is not None

    def test_coverage_default_output(self, tmp_path, monkeypatch):
        """coverage shows text output by default."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        # Create minimal config
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "api.md").write_text("# API")

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 1,
                "passed": 0,
                "failed": 0,
                "percentage": 0.0,
                "covered_files": [],
                "failed_files": [],
                "uncovered_files": ["docs/api.md"],
                "files_detail": {},
            }

            result = runner.invoke(cli, ['coverage'])

            assert result.exit_code == 0

    def test_coverage_json_output(self, tmp_path, monkeypatch):
        """coverage --output json returns JSON."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 2,
                "passed": 1,
                "percentage": 50.0,
            }

            result = runner.invoke(cli, ['coverage', '--output', 'json'])

            assert result.exit_code == 0
            assert '"total": 2' in result.output
            assert '"passed": 1' in result.output

    def test_coverage_with_path_filter(self, tmp_path, monkeypatch):
        """coverage filters by path argument."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 2,
                "passed": 1,
                "percentage": 50.0,
                "files_detail": {
                    "docs/api.md": {"status": "passed"},
                    "docs/guide.md": {"status": "uncovered"},
                },
                "covered_files": [],
                "failed_files": [],
                "uncovered_files": [],
            }

            result = runner.invoke(cli, ['coverage', 'docs/'])

            assert result.exit_code == 0

    def test_coverage_min_threshold_pass(self, tmp_path, monkeypatch):
        """coverage --min passes when threshold met."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 9,
                "percentage": 90.0,
                "covered_files": [],
                "failed_files": [],
                "uncovered_files": [],
                "files_detail": {},
            }

            result = runner.invoke(cli, ['coverage', '--min', '80'])

            assert result.exit_code == 0

    def test_coverage_min_threshold_fail(self, tmp_path, monkeypatch):
        """coverage --min fails when below threshold."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 5,
                "percentage": 50.0,
                "covered_files": [],
                "failed_files": [],
                "uncovered_files": [],
                "files_detail": {},
            }

            result = runner.invoke(cli, ['coverage', '--min', '80'])

            assert result.exit_code == 1
            assert 'below' in result.output.lower() or '80%' in result.output

    def test_coverage_files_flag(self, tmp_path, monkeypatch):
        """coverage --files shows file details."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 2,
                "passed": 1,
                "percentage": 50.0,
                "covered_files": ["docs/api.md"],
                "failed_files": [],
                "uncovered_files": ["docs/guide.md"],
                "files_detail": {
                    "docs/api.md": {"status": "passed", "test_count": 2},
                    "docs/guide.md": {"status": "uncovered", "test_count": 0},
                },
            }

            result = runner.invoke(cli, ['coverage', '--files'])

            assert result.exit_code == 0

    def test_coverage_uncovered_flag(self, tmp_path, monkeypatch):
        """coverage --uncovered shows only uncovered files."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 3,
                "passed": 1,
                "percentage": 33.0,
                "covered_files": ["docs/api.md"],
                "failed_files": [],
                "uncovered_files": ["docs/guide.md", "docs/readme.md"],
                "files_detail": {},
            }

            result = runner.invoke(cli, ['coverage', '--uncovered'])

            assert result.exit_code == 0


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_command_exists(self):
        """status command exists."""
        from dokumen.cli.commands.coverage import status

        assert status is not None

    def test_status_default_output(self, tmp_path, monkeypatch):
        """status shows text output by default."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 8,
                "percentage": 80.0,
            }

            result = runner.invoke(cli, ['status'])

            assert result.exit_code == 0
            assert '80%' in result.output

    def test_status_json_output(self, tmp_path, monkeypatch):
        """status --json returns JSON."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 5,
                "passed": 4,
                "percentage": 80.0,
            }

            result = runner.invoke(cli, ['status', '--json'])

            assert result.exit_code == 0
            assert '"coverage"' in result.output
            assert '"passed"' in result.output
            assert '"total"' in result.output

    def test_status_min_threshold_pass(self, tmp_path, monkeypatch):
        """status --min passes when threshold met."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 9,
                "percentage": 90.0,
            }

            result = runner.invoke(cli, ['status', '--min', '80'])

            assert result.exit_code == 0

    def test_status_min_threshold_fail(self, tmp_path, monkeypatch):
        """status --min fails when below threshold."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 5,
                "percentage": 50.0,
            }

            result = runner.invoke(cli, ['status', '--min', '80'])

            assert result.exit_code == 1

    def test_status_low_coverage_warning(self, tmp_path, monkeypatch):
        """status shows warning for low coverage."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 2,
                "percentage": 20.0,
            }

            result = runner.invoke(cli, ['status'])

            assert result.exit_code == 0
            # Should show LOW status

    def test_status_medium_coverage(self, tmp_path, monkeypatch):
        """status shows OK for medium coverage."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 6,
                "percentage": 60.0,
            }

            result = runner.invoke(cli, ['status'])

            assert result.exit_code == 0

    def test_status_high_coverage(self, tmp_path, monkeypatch):
        """status shows OK for high coverage."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 9,
                "percentage": 90.0,
            }

            result = runner.invoke(cli, ['status'])

            assert result.exit_code == 0
            # Should show OK with green

    def test_status_with_uncovered_files(self, tmp_path, monkeypatch):
        """status shows uncovered file count."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 7,
                "percentage": 70.0,
            }

            result = runner.invoke(cli, ['status'])

            assert result.exit_code == 0
            # Should mention uncovered files
            assert '3' in result.output or 'uncovered' in result.output.lower()

    def test_status_json_with_threshold(self, tmp_path, monkeypatch):
        """status --json includes threshold info."""
        from dokumen.cli import cli
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch.object(coverage_module, '_get_coverage_stats') as mock_stats:
            mock_stats.return_value = {
                "total": 10,
                "passed": 9,
                "percentage": 90.0,
            }

            result = runner.invoke(cli, ['status', '--json', '--min', '80'])

            assert result.exit_code == 0
            assert '"threshold"' in result.output
            assert '"threshold_passed"' in result.output


class TestGetCoverageStats:
    """Tests for _get_coverage_stats wrapper."""

    def test_wrapper_calls_get_coverage_stats(self):
        """_get_coverage_stats calls dokumen.cli.get_coverage_stats."""
        from dokumen.cli.commands.coverage import _get_coverage_stats

        with patch('dokumen.cli.get_coverage_stats') as mock_stats:
            mock_stats.return_value = {"total": 5}

            result = _get_coverage_stats({"key": "value"})

            assert result == {"total": 5}
            mock_stats.assert_called_once()
