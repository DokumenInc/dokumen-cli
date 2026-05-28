"""
CLI for testing whether agents follow business SOPs with LLM judges.

Core Commands:
    dokumen run               Run agent SOP tests
    dokumen validate          Validate config and test scaffolds
    dokumen list tests|files  List resources
"""

from pathlib import Path
from typing import Optional, List
from importlib.metadata import version as get_version

import click
from dotenv import load_dotenv

from ..logging_config import LogConfig, setup_logging, get_logger
from .commands import run, coverage, status, list_cmd, validate
from .commands.config_cmd import config
from .commands.explore import explore
from .commands.run import _run_tests as _run_tests
from .commands.summarize import summarize
from .helpers import (
    DEFAULT_CONFIG as DEFAULT_CONFIG,
    EXIT_CONFIG_ERROR as EXIT_CONFIG_ERROR,
    EXIT_FAILURE as EXIT_FAILURE,
    EXIT_INVALID_ARGS as EXIT_INVALID_ARGS,
    EXIT_RUNTIME_ERROR as EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS as EXIT_SUCCESS,
    deep_merge,
    discover_doc_files,
    filter_tests,
    get_coverage_stats as get_coverage_stats,
    get_uncovered_files as get_uncovered_files,
    load_config,
    normalize_path as normalize_path,
    run_async,
)

# Load .env file from current directory
load_dotenv()

# Initialize module-level logger (configured in cli() function)
logger = get_logger(__name__)

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

       Business SOP Agent Test CLI
"""


class DokumenGroup(click.Group):
    """Custom group that preserves command insertion order and groups commands."""

    COMMAND_GROUPS = [
        ("Core Commands", {"run", "validate", "list"}),
        ("Supporting Commands", {"help", "explore", "summarize", "config"}),
        ("Experimental Commands", {"coverage", "status"}),
    ]

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

        limit = formatter.width - 6 - max(len(name) for name, _ in commands)
        rendered = set()

        for section_name, command_names in self.COMMAND_GROUPS:
            section_cmds = [(name, cmd) for name, cmd in commands if name in command_names]
            if not section_cmds:
                continue

            with formatter.section(section_name):
                formatter.write_dl(
                    [
                        (subcommand, cmd.get_short_help_str(limit=limit))
                        for subcommand, cmd in section_cmds
                    ]
                )
            rendered.update(name for name, _ in section_cmds)

        other_cmds = [(name, cmd) for name, cmd in commands if name not in rendered]
        if other_cmds:
            with formatter.section("Other Commands"):
                formatter.write_dl(
                    [
                        (subcommand, cmd.get_short_help_str(limit=limit))
                        for subcommand, cmd in other_cmds
                    ]
                )

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
        formatter.write(click.style(BANNER, fg="blue", bold=True))
        formatter.write("\n")
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_commands(ctx, formatter)
        self.format_epilog(ctx, formatter)


@click.group(cls=DokumenGroup)
@click.version_option(version=get_version("dokumen"), prog_name="dokumen")
@click.option("--config", "-c", type=click.Path(), help="Configuration file path")
@click.option(
    "--debug",
    is_flag=True,
    help="Debug mode with trace file output to .dokumen-cache/debug-traces/",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="WARNING",
    help="Logging level",
)
@click.option("--log-file", type=click.Path(), help="Log file path for persistent logging")
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool, log_level: str, log_file: Optional[str]):
    """Test whether agents follow business SOPs with LLM judges.

    Dokumen runs an agent attempt with allowed tools, then judge agents evaluate
    whether it followed the configured procedure and success criteria.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["debug"] = debug
    if debug:
        ctx.obj["verbose"] = True  # --debug implies --verbose
        log_level = "DEBUG"  # --debug implies DEBUG log level

    # Configure logging
    log_config = LogConfig(
        level=log_level.upper(),
        log_file=Path(log_file) if log_file else None,
        json_format=False,  # Use text format for CLI output
    )
    setup_logging(log_config)

    logger.info("cli.start", version=get_version("dokumen"), log_level=log_level)


@click.command(name="help", context_settings={"ignore_unknown_options": True})
@click.argument("command_path", nargs=-1)
@click.pass_context
def help_command(ctx, command_path: tuple[str, ...]):
    """Show help for Dokumen or a command."""
    root_ctx = ctx.parent
    root_command = root_ctx.command

    if not command_path:
        click.echo(root_command.get_help(root_ctx))
        return

    current_command = root_command
    current_ctx = root_ctx
    for command_name in command_path:
        if not isinstance(current_command, click.Group):
            raise click.ClickException(f"Command has no subcommands: {' '.join(command_path)}")

        next_command = current_command.get_command(current_ctx, command_name)
        if next_command is None:
            raise click.ClickException(f"No such command: {' '.join(command_path)}")

        current_ctx = click.Context(next_command, info_name=command_name, parent=current_ctx)
        current_command = next_command

    click.echo(current_command.get_help(current_ctx))


# Register Phase 0 commands only
cli.add_command(run)
cli.add_command(validate)
cli.add_command(list_cmd)
cli.add_command(help_command)
cli.add_command(coverage)
cli.add_command(status)
cli.add_command(explore)
cli.add_command(summarize)
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
        return {"passed": result.test_results[0].passed}
    return {"passed": False}


def find_tests_for_file(file_path: str) -> List[str]:
    """Find tests that cover a specific file."""
    from ..loader import load_all_scaffolds

    tests, _load_errors = load_all_scaffolds()
    return [t.id for t in tests if file_path in [f.path for f in t.files]]


def get_failed_tests() -> List[str]:
    """Get list of previously failed test IDs from cache."""
    return []


async def _run_tests_impl(
    tests, timeout=None, bail=False, use_cache=True, config=None, quiet=False
):
    """Async test runner implementation (sequential only in Phase 0)."""
    from ..test_suite import TestSuite, TestSuiteConfig
    from .formatters import make_progress_callback

    suite = TestSuite(
        TestSuiteConfig(
            name="cli-run",
            parallel_execution=False,  # Phase 0: sequential only
            max_concurrency=1,
        )
    )

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

if __name__ == "__main__":
    cli()
