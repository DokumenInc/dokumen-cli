"""
CLI for the Dokumen Documentation Unit Test Framework.

Phase 0 Commands:
    dokumen run               Run documentation tests
    dokumen validate          Validate config and test scaffolds
    dokumen list tests|files  List resources
    dokumen coverage          View documentation coverage (file-level)
    dokumen status            Quick coverage status for CI/CD
"""
from pathlib import Path
from typing import Optional, List
from importlib.metadata import version as get_version

import click
from dotenv import load_dotenv

from ..logging_config import LogConfig, setup_logging, get_logger
from ..sentry_config import init_sentry

# Load .env file from current directory
load_dotenv()

# Initialize Sentry early (before any commands run)
init_sentry()

# Initialize module-level logger (configured in cli() function)
logger = get_logger(__name__)

# Import Phase 0 commands only
from .commands import run, coverage, status, list_cmd, validate
from .commands.explore import explore
from .commands.ask import ask
from .commands.create import create
from .commands.summarize import summarize
from .commands.mimick import mimick
from .commands.config_cmd import config

# Import _run_tests for backward compatibility (tests patch this)
from .commands.run import _run_tests

# Import helpers for backward compatibility
from .helpers import (
    EXIT_SUCCESS,
    EXIT_FAILURE,
    EXIT_CONFIG_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_INVALID_ARGS,
    run_async,
    DEFAULT_CONFIG,
    load_config,
    deep_merge,
    normalize_path,
    get_coverage_stats,
    discover_doc_files,
    get_uncovered_files,
    filter_tests,
)

# Re-export with old names for backward compatibility
_load_config = load_config
_deep_merge = deep_merge
_discover_doc_files = discover_doc_files
_filter_tests = filter_tests


# =============================================================================
# Main CLI Group
# =============================================================================

BANNER = r"""
    ____        __
   / __ \____  / /____  ______ ___  ___  ____
  / / / / __ \/ //_/ / / / __ `__ \/ _ \/ __ \
 / /_/ / /_/ / ,< / /_/ / / / / / /  __/ / / /
/_____/\____/_/|_|\__,_/_/ /_/ /_/\___/_/ /_/

       Documentation Unit Test Framework
"""


class DokumenGroup(click.Group):
    """Custom group that preserves command insertion order and groups commands."""

    # Define which commands are "main" vs "other"
    MAIN_COMMANDS = {'run', 'validate', 'list', 'coverage', 'status', 'explore', 'ask', 'create', 'summarize', 'mimick', 'config'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track insertion order
        self._command_order = []

    def add_command(self, cmd, name=None):
        super().add_command(cmd, name)
        cmd_name = name or cmd.name
        if cmd_name not in self._command_order:
            self._command_order.append(cmd_name)

    def list_commands(self, ctx):
        # Return commands in insertion order instead of alphabetical
        return self._command_order

    def format_commands(self, ctx, formatter):
        """Override to split commands into Main and Other sections."""
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            commands.append((subcommand, cmd))

        if not commands:
            return

        # Split into main and other
        main_cmds = [(name, cmd) for name, cmd in commands if name in self.MAIN_COMMANDS]
        other_cmds = [(name, cmd) for name, cmd in commands if name not in self.MAIN_COMMANDS]

        # Calculate max width
        limit = formatter.width - 6 - max(len(name) for name, _ in commands)

        # Write main commands
        if main_cmds:
            with formatter.section("Main Commands"):
                rows = []
                for subcommand, cmd in main_cmds:
                    help_text = cmd.get_short_help_str(limit=limit)
                    rows.append((subcommand, help_text))
                formatter.write_dl(rows)

        # Write other commands
        if other_cmds:
            with formatter.section("Other Commands"):
                rows = []
                for subcommand, cmd in other_cmds:
                    help_text = cmd.get_short_help_str(limit=limit)
                    rows.append((subcommand, help_text))
                formatter.write_dl(rows)

    def format_options(self, ctx, formatter):
        """Override to NOT call format_commands (we handle that separately)."""
        opts = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                opts.append(rv)
        if opts:
            with formatter.section("Options"):
                formatter.write_dl(opts)

    def format_help(self, ctx, formatter):
        formatter.write(click.style(BANNER, fg='blue', bold=True))
        formatter.write("\n")
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_commands(ctx, formatter)
        self.format_epilog(ctx, formatter)


@click.group(cls=DokumenGroup)
@click.version_option(version=get_version("dokumen"), prog_name="dokumen")
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@click.option('--debug', is_flag=True, help='Debug mode with trace file output to .dokumen-cache/debug-traces/')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
              default='INFO', help='Logging level')
@click.option('--log-file', type=click.Path(), help='Log file path for persistent logging')
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool, log_level: str, log_file: Optional[str]):
    """Test your documentation with AI agents.

    Dokumen verifies that your docs are accurate by having an executor
    agent read them and perform tasks, then judge agents evaluate the results.
    """
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['debug'] = debug
    if debug:
        ctx.obj['verbose'] = True  # --debug implies --verbose
        log_level = 'DEBUG'  # --debug implies DEBUG log level

    # Configure logging
    log_config = LogConfig(
        level=log_level.upper(),
        log_file=Path(log_file) if log_file else None,
        json_format=False,  # Use text format for CLI output
    )
    setup_logging(log_config)

    logger.info("cli.start", version=get_version("dokumen"), log_level=log_level)


# Register Phase 0 commands only
cli.add_command(run)
cli.add_command(validate)
cli.add_command(list_cmd)
cli.add_command(coverage)
cli.add_command(status)
cli.add_command(explore)
cli.add_command(ask)
cli.add_command(create)
cli.add_command(summarize)
cli.add_command(mimick)
cli.add_command(config)


# =============================================================================
# Helper functions for tests (backward compatibility)
# =============================================================================

def run_test_suite(tests=None, **kwargs):
    """Run test suite - callable by tests."""
    from ..loader import load_all_scaffolds, get_configured_provider

    provider = get_configured_provider()
    all_tests, _load_errors = load_all_scaffolds(provider=provider)

    filtered = filter_tests(all_tests, test_ids=tests)
    return run_async(_run_tests_impl(filtered, **kwargs))


def run_test_by_id(test_id: str) -> dict:
    """Run single test by ID - callable by tests."""
    result = run_test_suite(tests=(test_id,))
    if result.test_results:
        return {'passed': result.test_results[0].passed}
    return {'passed': False}


def find_tests_for_file(file_path: str) -> List[str]:
    """Find tests that cover a specific file."""
    from ..loader import load_all_scaffolds

    tests, _load_errors = load_all_scaffolds()
    return [t.id for t in tests if file_path in [f.path for f in t.files]]


def get_failed_tests() -> List[str]:
    """Get list of previously failed test IDs from cache."""
    return []


async def _run_tests_impl(tests, timeout=None, bail=False, use_cache=True, config=None, quiet=False):
    """Async test runner implementation (sequential only in Phase 0)."""
    from ..test_suite import TestSuite, TestSuiteConfig
    from ..loader import get_configured_provider
    from .formatters import make_progress_callback

    provider = get_configured_provider()

    suite = TestSuite(TestSuiteConfig(
        name="cli-run",
        parallel_execution=False,  # Phase 0: sequential only
        max_concurrency=1,
    ))

    for test in tests:
        if timeout:
            test.timeout = float(timeout)
        suite.add_test(test)

    if use_cache:
        await suite.load_cache()

    cached_tests = set()
    progress_callback = make_progress_callback(quiet, cached_tests)
    results = await suite.run(on_progress=progress_callback)

    # Attach cached test IDs to results for display
    results.cached_test_ids = cached_tests

    if use_cache:
        await suite.save_cache()

    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    cli()
