"""Tests for CLI __init__ module."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock


class TestDokumenGroup:
    """Tests for DokumenGroup click group class."""

    def test_dokumen_group_exists(self):
        """DokumenGroup class exists."""
        from dokumen.cli import DokumenGroup

        assert DokumenGroup is not None

    def test_dokumen_group_main_commands(self):
        """DokumenGroup has MAIN_COMMANDS defined."""
        from dokumen.cli import DokumenGroup

        assert 'run' in DokumenGroup.MAIN_COMMANDS
        assert 'validate' in DokumenGroup.MAIN_COMMANDS
        assert 'list' in DokumenGroup.MAIN_COMMANDS
        assert 'coverage' in DokumenGroup.MAIN_COMMANDS
        assert 'status' in DokumenGroup.MAIN_COMMANDS

    def test_dokumen_group_init(self):
        """DokumenGroup initializes with empty command order."""
        from dokumen.cli import DokumenGroup

        group = DokumenGroup()

        assert group._command_order == []

    def test_add_command_tracks_order(self):
        """add_command tracks insertion order."""
        from dokumen.cli import DokumenGroup
        import click

        group = DokumenGroup()

        @click.command()
        def cmd1():
            pass

        @click.command()
        def cmd2():
            pass

        group.add_command(cmd1)
        group.add_command(cmd2)

        assert group._command_order == ['cmd1', 'cmd2']

    def test_add_command_with_custom_name(self):
        """add_command uses custom name when provided."""
        from dokumen.cli import DokumenGroup
        import click

        group = DokumenGroup()

        @click.command()
        def original():
            pass

        group.add_command(original, name='custom')

        assert 'custom' in group._command_order
        assert 'original' not in group._command_order

    def test_list_commands_returns_order(self):
        """list_commands returns commands in insertion order."""
        from dokumen.cli import DokumenGroup
        import click

        group = DokumenGroup()

        @click.command()
        def alpha():
            pass

        @click.command()
        def zeta():
            pass

        @click.command()
        def beta():
            pass

        group.add_command(zeta)
        group.add_command(alpha)
        group.add_command(beta)

        commands = group.list_commands(None)

        assert commands == ['zeta', 'alpha', 'beta']

    def test_add_command_no_duplicates(self):
        """add_command doesn't add duplicates."""
        from dokumen.cli import DokumenGroup
        import click

        group = DokumenGroup()

        @click.command()
        def cmd():
            pass

        group.add_command(cmd)
        group.add_command(cmd)

        assert group._command_order.count('cmd') == 1


class TestCliGroup:
    """Tests for the main cli group."""

    def test_cli_exists(self):
        """cli group exists."""
        from dokumen.cli import cli

        assert cli is not None

    def test_cli_has_version(self):
        """cli has version option."""
        from dokumen.cli import cli
        runner = CliRunner()

        result = runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert 'dokumen' in result.output.lower() or 'version' in result.output.lower()

    def test_cli_help(self):
        """cli shows help."""
        from dokumen.cli import cli
        runner = CliRunner()

        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'run' in result.output.lower()
        assert 'validate' in result.output.lower()

    def test_cli_banner_in_help(self):
        """cli help shows banner."""
        from dokumen.cli import cli
        runner = CliRunner()

        result = runner.invoke(cli, ['--help'])

        # Banner contains "Dokumen" text
        assert 'dokumen' in result.output.lower() or 'Documentation' in result.output


class TestCliCommands:
    """Tests for CLI command registration."""

    def test_run_command_registered(self):
        """run command is registered."""
        from dokumen.cli import cli

        command = cli.get_command(None, 'run')

        assert command is not None
        assert command.name == 'run'

    def test_validate_command_registered(self):
        """validate command is registered."""
        from dokumen.cli import cli

        command = cli.get_command(None, 'validate')

        assert command is not None
        assert command.name == 'validate'

    def test_list_command_registered(self):
        """list command is registered."""
        from dokumen.cli import cli

        command = cli.get_command(None, 'list')

        assert command is not None
        assert command.name == 'list'

    def test_coverage_command_registered(self):
        """coverage command is registered."""
        from dokumen.cli import cli

        command = cli.get_command(None, 'coverage')

        assert command is not None
        assert command.name == 'coverage'

    def test_status_command_registered(self):
        """status command is registered."""
        from dokumen.cli import cli

        command = cli.get_command(None, 'status')

        assert command is not None
        assert command.name == 'status'


class TestCliReexports:
    """Tests for re-exported modules."""

    def test_exit_codes_exported(self):
        """Exit codes are exported."""
        from dokumen.cli import (
            EXIT_SUCCESS,
            EXIT_FAILURE,
            EXIT_CONFIG_ERROR,
            EXIT_RUNTIME_ERROR,
            EXIT_INVALID_ARGS,
        )

        assert EXIT_SUCCESS == 0
        assert EXIT_FAILURE == 1
        assert EXIT_CONFIG_ERROR == 2
        assert EXIT_RUNTIME_ERROR == 3
        assert EXIT_INVALID_ARGS == 4

    def test_run_async_exported(self):
        """run_async is exported."""
        from dokumen.cli import run_async

        assert callable(run_async)

    def test_load_config_exported(self):
        """load_config is exported."""
        from dokumen.cli import load_config

        assert callable(load_config)

    def test_deep_merge_exported(self):
        """deep_merge is exported."""
        from dokumen.cli import deep_merge

        assert callable(deep_merge)

    def test_normalize_path_exported(self):
        """normalize_path is exported."""
        from dokumen.cli import normalize_path

        assert callable(normalize_path)

    def test_filter_tests_exported(self):
        """filter_tests is exported."""
        from dokumen.cli import filter_tests

        assert callable(filter_tests)

    def test_backward_compat_aliases(self):
        """Backward compatibility aliases exist."""
        from dokumen.cli import _load_config, _deep_merge, _discover_doc_files, _filter_tests

        assert callable(_load_config)
        assert callable(_deep_merge)
        assert callable(_discover_doc_files)
        assert callable(_filter_tests)


class TestCLIWithDebugFlag:
    """Tests for CLI debug flag."""

    def test_cli_debug_option(self):
        """CLI has debug option."""
        from dokumen.cli import cli

        # Check the cli group has debug parameter
        params = {p.name for p in cli.params}
        assert 'debug' in params

    def test_cli_config_option(self):
        """CLI has config option."""
        from dokumen.cli import cli

        params = {p.name for p in cli.params}
        assert 'config' in params


class TestFormatCommands:
    """Tests for DokumenGroup.format_commands method."""

    def test_format_commands_splits_main_and_other(self):
        """format_commands splits commands into Main and Other sections."""
        from dokumen.cli import DokumenGroup
        import click

        group = DokumenGroup()

        @click.command()
        def run():
            """Run tests."""
            pass

        @click.command()
        def custom():
            """Custom command."""
            pass

        group.add_command(run)
        group.add_command(custom)

        # Create mock context and formatter
        ctx = click.Context(group)
        formatter = click.HelpFormatter()

        group.format_commands(ctx, formatter)

        # Check output contains both sections
        output = formatter.getvalue()
        # main commands like 'run' should be present
        assert 'run' in output.lower()

    def test_format_commands_empty_group(self):
        """format_commands handles empty group."""
        from dokumen.cli import DokumenGroup
        import click

        group = DokumenGroup()

        ctx = click.Context(group)
        formatter = click.HelpFormatter()

        # Should not raise
        group.format_commands(ctx, formatter)

    def test_format_commands_hidden_command(self):
        """format_commands skips hidden commands."""
        from dokumen.cli import DokumenGroup
        import click

        group = DokumenGroup()

        @click.command(hidden=True)
        def hidden_cmd():
            """Hidden command."""
            pass

        group.add_command(hidden_cmd)

        ctx = click.Context(group)
        formatter = click.HelpFormatter()

        group.format_commands(ctx, formatter)

        output = formatter.getvalue()
        assert 'hidden_cmd' not in output


class TestFormatOptions:
    """Tests for DokumenGroup.format_options method."""

    def test_format_options_basic(self):
        """format_options formats options."""
        from dokumen.cli import DokumenGroup
        import click

        @click.command(cls=DokumenGroup)
        @click.option('--test', help='Test option')
        def cmd():
            pass

        ctx = click.Context(cmd)
        formatter = click.HelpFormatter()

        cmd.format_options(ctx, formatter)

        output = formatter.getvalue()
        # Options should be present
        assert 'test' in output.lower() or 'Options' in output


class TestFormatHelp:
    """Tests for DokumenGroup.format_help method."""

    def test_format_help_includes_banner(self):
        """format_help includes banner."""
        from dokumen.cli import DokumenGroup, BANNER
        import click

        group = DokumenGroup()
        ctx = click.Context(group)
        formatter = click.HelpFormatter()

        group.format_help(ctx, formatter)

        output = formatter.getvalue()
        # Banner should be in output
        assert 'Dokumen' in output or 'Documentation' in output


class TestBANNER:
    """Tests for CLI banner."""

    def test_banner_exists(self):
        """BANNER constant exists."""
        from dokumen.cli import BANNER

        assert BANNER is not None
        assert isinstance(BANNER, str)

    def test_banner_contains_text(self):
        """BANNER contains descriptive text."""
        from dokumen.cli import BANNER

        assert 'Documentation' in BANNER or 'Test' in BANNER


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG export."""

    def test_default_config_exported(self):
        """DEFAULT_CONFIG is exported."""
        from dokumen.cli import DEFAULT_CONFIG

        assert isinstance(DEFAULT_CONFIG, dict)

    def test_default_config_has_version(self):
        """DEFAULT_CONFIG has version."""
        from dokumen.cli import DEFAULT_CONFIG

        assert 'version' in DEFAULT_CONFIG

    def test_default_config_has_provider(self):
        """DEFAULT_CONFIG has provider."""
        from dokumen.cli import DEFAULT_CONFIG

        assert 'provider' in DEFAULT_CONFIG


class TestCliDebugMode:
    """Tests for CLI debug flag behavior."""

    def test_debug_flag_sets_verbose(self):
        """--debug flag sets verbose in context."""
        from dokumen.cli import cli
        runner = CliRunner()

        # Invoke with debug flag - just check it doesn't crash
        result = runner.invoke(cli, ['--debug', '--help'])

        assert result.exit_code == 0


class TestBackwardCompatFunctions:
    """Tests for backward compatibility helper functions."""

    def test_get_failed_tests_returns_empty(self):
        """get_failed_tests returns empty list."""
        from dokumen.cli import get_failed_tests

        result = get_failed_tests()

        assert result == []

    def test_find_tests_for_file_returns_list(self, tmp_path, monkeypatch):
        """find_tests_for_file returns list of test IDs."""
        from dokumen.cli import find_tests_for_file
        monkeypatch.chdir(tmp_path)

        # Create minimal config
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        result = find_tests_for_file("docs/api.md")

        assert isinstance(result, list)

    def test_run_test_by_id_returns_dict(self, tmp_path, monkeypatch):
        """run_test_by_id returns dict with passed key."""
        from dokumen.cli import run_test_by_id
        monkeypatch.chdir(tmp_path)

        # Create minimal config
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Non-existent test should return passed=False
        result = run_test_by_id("nonexistent-test")

        assert isinstance(result, dict)
        assert 'passed' in result

    def test_run_test_suite_callable(self, tmp_path, monkeypatch):
        """run_test_suite is callable."""
        from dokumen.cli import run_test_suite
        from unittest.mock import patch, AsyncMock
        monkeypatch.chdir(tmp_path)

        # Create minimal config
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        with patch('dokumen.cli._run_tests_impl', new_callable=AsyncMock) as mock_run:
            from dokumen.test_suite import TestSuiteResults
            mock_run.return_value = TestSuiteResults(
                passed=0,
                failed=0,
                test_results=[],
                duration=0.0,
                total_tests=0,
                skipped=0,
                cached_results={}
            )

            result = run_test_suite()

            assert result is not None


class TestRunTestsImpl:
    """Tests for _run_tests_impl async function."""

    @pytest.mark.asyncio
    async def test_run_tests_impl_empty(self, tmp_path, monkeypatch):
        """_run_tests_impl handles empty test list."""
        from dokumen.cli import _run_tests_impl
        from unittest.mock import patch, AsyncMock
        monkeypatch.chdir(tmp_path)

        # Create minimal config
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        results = await _run_tests_impl([])

        assert results.passed == 0
        assert results.failed == 0

    @pytest.mark.asyncio
    async def test_run_tests_impl_with_timeout(self, tmp_path, monkeypatch):
        """_run_tests_impl sets timeout on tests."""
        from dokumen.cli import _run_tests_impl
        from dokumen.test_object import TestObject
        # AgentObject removed
        from dokumen.file_object import FileObject
        from unittest.mock import patch, AsyncMock, MagicMock
        monkeypatch.chdir(tmp_path)

        # Create minimal config
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        # Create a mock test object
        mock_test = MagicMock(spec=TestObject)
        mock_test.id = "test-1"
        mock_test.timeout = 60
        mock_test.files = []

        with patch('dokumen.test_suite.TestSuite.run', new_callable=AsyncMock) as mock_run:
            from dokumen.test_suite import TestSuiteResults
            mock_run.return_value = TestSuiteResults(
                passed=0, failed=0, test_results=[], duration=0.0,
                total_tests=0, skipped=0, cached_results={}
            )

            results = await _run_tests_impl([mock_test], timeout=120, use_cache=False)

            # Timeout should be set on test
            assert mock_test.timeout == 120.0

    @pytest.mark.asyncio
    async def test_run_tests_impl_with_cache(self, tmp_path, monkeypatch):
        """_run_tests_impl uses cache when enabled."""
        from dokumen.cli import _run_tests_impl
        from unittest.mock import patch, AsyncMock
        monkeypatch.chdir(tmp_path)

        # Create minimal config
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        with patch('dokumen.test_suite.TestSuite.load_cache', new_callable=AsyncMock) as mock_load:
            with patch('dokumen.test_suite.TestSuite.save_cache', new_callable=AsyncMock) as mock_save:
                with patch('dokumen.test_suite.TestSuite.run', new_callable=AsyncMock) as mock_run:
                    from dokumen.test_suite import TestSuiteResults
                    mock_run.return_value = TestSuiteResults(
                        passed=0, failed=0, test_results=[], duration=0.0,
                        total_tests=0, skipped=0, cached_results={}
                    )

                    await _run_tests_impl([], use_cache=True)

                    mock_load.assert_called_once()
                    mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_tests_impl_no_cache(self, tmp_path, monkeypatch):
        """_run_tests_impl skips cache when disabled."""
        from dokumen.cli import _run_tests_impl
        from unittest.mock import patch, AsyncMock
        monkeypatch.chdir(tmp_path)

        # Create minimal config
        (tmp_path / "dokumen.yaml").write_text("version: '1.0'")

        with patch('dokumen.test_suite.TestSuite.load_cache', new_callable=AsyncMock) as mock_load:
            with patch('dokumen.test_suite.TestSuite.save_cache', new_callable=AsyncMock) as mock_save:
                with patch('dokumen.test_suite.TestSuite.run', new_callable=AsyncMock) as mock_run:
                    from dokumen.test_suite import TestSuiteResults
                    mock_run.return_value = TestSuiteResults(
                        passed=0, failed=0, test_results=[], duration=0.0,
                        total_tests=0, skipped=0, cached_results={}
                    )

                    await _run_tests_impl([], use_cache=False)

                    mock_load.assert_not_called()
                    mock_save.assert_not_called()
