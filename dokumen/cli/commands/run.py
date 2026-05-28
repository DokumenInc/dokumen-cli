"""
Run command for dokumen CLI - Phase 0.
"""

import json
import os
import sys
from typing import Optional, Tuple

import click

from ..helpers import (
    EXIT_SUCCESS,
    EXIT_FAILURE,
    EXIT_CONFIG_ERROR,
    EXIT_INVALID_ARGS,
    load_config,
    filter_tests,
    run_async,
    validate_folder_path,
)
from ..formatters import (
    results_to_dict,
    results_to_junit,
    results_to_tap,
    print_results_text,
    print_run_settings,
    make_progress_callback,
    make_tool_call_callback,
    make_conversation_callback,
    make_executor_complete_callback,
    make_judge_complete_callback,
)
from ..output import OutputWriter
from ..helpers import get_coverage_stats


@click.command()
@click.argument("tests", nargs=-1)
@click.option("--grep", "-g", help="Filter tests by pattern")
@click.option("--file", "for_file", help="Run tests for specific file")
@click.option(
    "--folder",
    "-d",
    default=None,
    help='Run tests in folder and subfolders. Use "." for root-level only.',
)
@click.option("--timeout", "-t", type=int, help="Override timeout (seconds)")
@click.option("--bail", "-b", is_flag=True, help="Stop on first failure")
@click.option(
    "--force", "-f", is_flag=True, envvar="DOKUMEN_FORCE", help="Force run skipped/cached tests"
)
@click.option("--dry-run", is_flag=True, help="Show tests without running")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json", "junit", "tap"]),
    default="text",
    help="Output format",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output including tool calls")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output, suppress progress")
@click.option(
    "--parallel",
    "-p",
    type=int,
    default=None,
    help="Run tests in parallel with N concurrent tests (e.g. --parallel 4)",
)
@click.option("--debug", is_flag=True, help="Show internal debug output")
@click.pass_context
def run(
    ctx,
    tests: Tuple[str],
    grep: Optional[str],
    for_file: Optional[str],
    folder: Optional[str],
    timeout: Optional[int],
    bail: bool,
    force: bool,
    dry_run: bool,
    output: str,
    verbose: bool,
    quiet: bool,
    parallel: Optional[int],
    debug: bool,
):
    """Run agent SOP tests.

    Executes tests and reports results.

    Examples:

        dokumen run                  Run all tests

        dokumen run api-auth-test    Run specific test

        dokumen run --grep "api-*"   Run tests matching pattern

        dokumen run --folder api     Run tests in api/ folder

        dokumen run --dry-run        Show what would run
    """
    from ...loader import load_all_scaffolds, get_configured_providers
    from ...debug import set_debug, start_debug_session, end_debug_session

    # Check for global --debug from parent context, merge with local flag
    global_debug = ctx.obj.get("debug", False)
    debug = debug or global_debug

    # Check DOKUMEN_FORCE env var explicitly
    if os.environ.get("DOKUMEN_FORCE", "").lower() in ("1", "true", "yes", "on"):
        force = True

    # Read DOKUMEN_TESTS env var (CLI args override env vars)
    env_tests = os.environ.get("DOKUMEN_TESTS", "")
    if env_tests and not tests:  # Only use env var if no CLI tests specified
        tests = tuple(t.strip() for t in env_tests.split(",") if t.strip())
        if not quiet:
            click.echo(f"Filtering by DOKUMEN_TESTS: {len(tests)} test(s) selected")

    # Read DOKUMEN_TIMEOUT env var (CLI args override env vars)
    env_timeout = os.environ.get("DOKUMEN_TIMEOUT", "")
    if not timeout and env_timeout:
        try:
            timeout = int(env_timeout)
        except ValueError:
            pass  # Ignore invalid timeout values

    # --debug implies --verbose
    if debug:
        verbose = True

    # Also check for global verbose from --debug
    if ctx.obj.get("verbose", False):
        verbose = True

    # Enable debug mode if requested
    set_debug(debug)

    config = load_config(ctx.obj.get("config_path"))

    # Start debug session if enabled
    if debug:
        provider_name = config.get("provider", {}).get("name", "unknown")
        model_name = config.get("provider", {}).get("model", "default")
        start_debug_session(
            command="run",
            meta={
                "provider": provider_name,
                "model": model_name,
                "tests_requested": list(tests) if tests else "all",
            },
        )

    try:
        # Load all tests
        try:
            config_path = ctx.obj.get("config_path")
            # Get separate providers for executor and judge (supports different models)
            providers = get_configured_providers(config_path)
            all_tests, load_errors = load_all_scaffolds(
                tests_dir="tests",
                provider=providers["default"],
                config_path=config_path,
                executor_provider=providers["executor"],
                judge_provider=providers["judge"],
            )
        except Exception as e:
            click.echo(f"Error loading tests: {e}", err=True)
            sys.exit(EXIT_CONFIG_ERROR)

        if not all_tests and not load_errors:
            click.echo("No tests found in tests/ directory.", err=True)
            sys.exit(EXIT_FAILURE)

        # All scaffolds failed to load — report errors regardless of filters
        if not all_tests and load_errors:
            for name, err in load_errors.items():
                click.echo(f"Error: Test '{name}' failed to load: {err}", err=True)
            sys.exit(EXIT_CONFIG_ERROR)

        # Validate folder path early
        if folder is not None:
            try:
                folder = validate_folder_path(folder)
            except ValueError as e:
                click.echo(f"Invalid folder path: {e}", err=True)
                sys.exit(EXIT_INVALID_ARGS)

        # Filter tests
        filtered_tests = filter_tests(all_tests, tests, grep, for_file, folder)

        if not filtered_tests:
            if tests:
                # Check if any requested tests failed to load
                failed_to_load = {t: load_errors[t] for t in tests if t in load_errors}
                if failed_to_load:
                    for name, err in failed_to_load.items():
                        click.echo(f"Error: Test '{name}' failed to load: {err}", err=True)
                    sys.exit(EXIT_CONFIG_ERROR)
                click.echo(f"No tests match: {', '.join(tests)}", err=True)
            elif grep:
                click.echo(f"No tests match pattern: {grep}", err=True)
            elif folder is not None:
                click.echo(f"No tests found in folder: {folder or '(root)'}", err=True)
            else:
                click.echo("No tests match the specified criteria.", err=True)
            sys.exit(EXIT_FAILURE)

        # Dry run mode
        if dry_run:
            click.echo(f"Would run {len(filtered_tests)} test(s):")
            for test in filtered_tests:
                click.echo(f"  - {test.id}")
            sys.exit(EXIT_SUCCESS)

        # Suppress progress for non-text output formats
        suppress_progress = quiet or output != "text"

        if output == "text" and not quiet:
            print_run_settings(
                config=config,
                test_count=len(filtered_tests),
                verbose=verbose,
                debug=debug,
                force=force,
                bail=bail,
                timeout_override=timeout,
            )

        # Run tests (sequential or parallel)
        result = run_async(
            _run_tests(
                filtered_tests,
                timeout=timeout,
                bail=bail,
                use_cache=not force,
                config=config,
                quiet=suppress_progress,
                verbose=verbose,
                parallel=parallel,
            )
        )

        # Write output files to .dokumen-cache/
        cache_path = config.get("cache", {}).get("path", ".dokumen-cache")
        coverage_stats = get_coverage_stats(config=config)
        writer = OutputWriter(cache_dir=cache_path)
        writer.write_all(result, coverage_stats, debug_enabled=debug)

        # Output results
        if output == "json":
            click.echo(json.dumps(results_to_dict(result), indent=2))
            sys.stdout.flush()
        elif output == "junit":
            click.echo(results_to_junit(result))
            sys.stdout.flush()
        elif output == "tap":
            click.echo(results_to_tap(result))
            sys.stdout.flush()
        else:
            print_results_text(result, verbose=verbose)

        if result.failed > 0 or result.error > 0:
            sys.exit(EXIT_FAILURE)

    finally:
        # Write debug trace file if debug enabled
        if debug:
            output_path = end_debug_session()
            if output_path:
                click.echo(f"\nDebug trace: {output_path}")


async def _run_tests(
    tests,
    timeout=None,
    bail=False,
    use_cache=True,
    config=None,
    quiet=False,
    verbose=False,
    parallel=None,
):
    """Async test runner implementation."""
    from ...test_suite import TestSuite, TestSuiteConfig

    suite = TestSuite(
        TestSuiteConfig(
            name="cli-run",
            parallel_execution=parallel is not None,
            max_concurrency=parallel or 1,
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
    tool_call_callback = make_tool_call_callback(quiet, verbose) if verbose else None
    conversation_callback = make_conversation_callback(quiet, verbose) if verbose else None
    executor_complete_callback = make_executor_complete_callback(quiet, verbose)
    judge_complete_callback = make_judge_complete_callback(quiet, verbose)

    results = await suite.run(
        on_progress=progress_callback,
        on_tool_call=tool_call_callback,
        on_conversation_message=conversation_callback,
        on_executor_complete=executor_complete_callback,
        on_judge_complete=judge_complete_callback,
    )

    # Attach cached test IDs to results for display
    results.cached_test_ids = cached_tests

    # Always save cache
    await suite.save_cache()

    return results
