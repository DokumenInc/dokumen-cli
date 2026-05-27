"""
Output formatters for CLI commands.
"""

import logging
from typing import Dict, List, Optional, Tuple

import click

logger = logging.getLogger(__name__)


# =============================================================================
# Run Settings Banner
# =============================================================================


def print_run_settings(
    config: dict,
    test_count: int,
    verbose: bool = False,
    debug: bool = False,
    force: bool = False,
    bail: bool = False,
    timeout_override: Optional[int] = None,
) -> None:
    """Print resolved CLI settings at the start of a dokumen run.

    Args:
        config: The merged config dict from load_config().
        test_count: Number of tests that will be executed.
        verbose: Whether --verbose flag is active.
        debug: Whether --debug flag is active.
        force: Whether --force flag is active (disables cache).
        bail: Whether --bail flag is active.
        timeout_override: CLI --timeout override value, if provided.
    """
    logger.info(
        "Printing run settings",
        extra={
            "test_count": test_count,
            "force": force,
            "bail": bail,
            "debug": debug,
            "verbose": verbose,
            "timeout_override": timeout_override,
        },
    )

    # Provider & models
    provider_cfg = config.get("provider", {})
    provider_name = provider_cfg.get("name", "unknown")
    default_model = provider_cfg.get("model", "default")
    executor_model = config.get("executor_model", default_model)
    judge_model = config.get("judge_model", default_model)

    click.echo(f"\nProvider: {provider_name}")
    click.echo(f"  Executor model: {executor_model}")
    click.echo(f"  Judge model: {judge_model}")

    # Explore
    explore_cfg = config.get("explore", {})
    explore_enabled = explore_cfg.get("enabled", True) if explore_cfg else True
    if explore_enabled and explore_cfg:
        explore_model = explore_cfg.get("model", default_model)
        max_files = explore_cfg.get("max_files", 20)
        explore_timeout = explore_cfg.get("timeout", 60)
        click.echo(f"  Explore model: {explore_model}")
        click.echo(f"  Explore: max_files={max_files}, timeout={explore_timeout}s")
    else:
        click.echo("  Explore: disabled")

    # Execution
    exec_cfg = config.get("execution", {})
    exec_timeout = exec_cfg.get("timeout", 60)
    retries = exec_cfg.get("retries", 0)
    click.echo(f"Execution: timeout={exec_timeout}s, retries={retries}")

    # Coverage
    cov_cfg = config.get("coverage", {})
    include = cov_cfg.get("include", [])
    min_threshold = cov_cfg.get("min_threshold")
    cov_parts = [f"include={include}"]
    if min_threshold is not None:
        cov_parts.append(f"min={min_threshold}%")
    click.echo(f"Coverage: {', '.join(cov_parts)}")

    # Tools
    tools_cfg = config.get("tools", {})
    if tools_cfg:
        defaults = tools_cfg.get("defaults")
        allowed = tools_cfg.get("allowed")
        if defaults:
            click.echo(f"Tools defaults: {defaults}")
        if allowed:
            click.echo(f"Tools allowed: {allowed}")
        blocked = tools_cfg.get("blocked")
        if blocked:
            click.echo(f"Tools blocked: {blocked}")

    # Cache
    if force:
        click.echo("Cache: disabled (force)")
    else:
        cache_cfg = config.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True)
        cache_path = cache_cfg.get("path", ".dokumen-cache")
        if cache_enabled:
            click.echo(f"Cache: enabled ({cache_path})")
        else:
            click.echo("Cache: disabled")

    # Flags (only shown when at least one is active)
    flags = []
    if force:
        flags.append("force")
    if bail:
        flags.append("bail")
    if debug:
        flags.append("debug")
    if verbose:
        flags.append("verbose")
    if timeout_override is not None:
        flags.append(f"timeout={timeout_override}s")
    if flags:
        click.echo(f"Flags: {', '.join(flags)}")

    # Test count
    click.echo(f"Running {test_count} test(s)...\n")
    import sys

    sys.stdout.flush()

    logger.debug("Run settings printed successfully")


# =============================================================================
# Coverage Formatters
# =============================================================================


def print_coverage_text(
    stats: dict,
    files: bool = False,
    uncovered: bool = False,
    tree: bool = False,
    verbose: bool = False,
    line_stats: dict = None,
    quiet: bool = False,
):
    """Print coverage in text format with bar-per-state display."""
    total = stats["total"]
    by_state = stats.get(
        "by_state",
        {
            "passed": stats.get("passed", 0),
            "failed": stats.get("failed", 0),
            "uncovered": len(stats.get("uncovered_files", [])),
        },
    )
    test_counts = stats.get("test_counts", {})
    total_tests = sum(test_counts.values())

    # Quiet mode: only show summary line
    if quiet:
        pct = stats["percentage"]
        passed = stats["passed"]
        line_pct = line_stats.get("percentage", 0) if line_stats else 0
        line_total = line_stats.get("total_lines", 0) if line_stats else 0
        line_covered = line_stats.get("covered_lines", 0) if line_stats else 0
        if line_stats and line_total > 0:
            click.echo(
                f"Experimental coverage: {pct:.0f}% files ({passed}/{total}), {line_pct:.1f}% lines ({line_covered}/{line_total})"
            )
        else:
            click.echo(f"Experimental coverage: {pct:.0f}% ({passed}/{total} files)")
        return

    click.echo("\nExperimental Source Coverage")
    click.echo("=" * 28)

    # Files section with bar per state
    click.echo(f"\nFiles: {total} total, {total_tests} tests")
    click.echo("-" * 44)
    _print_state_bars(by_state, total)

    # Lines section with bar per state
    if line_stats:
        line_total = line_stats.get("total_lines", 0)
        line_by_state = line_stats.get(
            "by_state",
            {
                "passed": line_stats.get("covered_lines", 0),
                "failed": line_stats.get("failed_lines", 0),
                "uncovered": 0,
            },
        )
        if line_total > 0:
            click.echo(f"\nLines: {line_total} total")
            click.echo("-" * 44)
            _print_state_bars(line_by_state, line_total)

    # Show per-file details with --files flag
    if files:
        _print_files_table(stats, line_stats)

    # Show failed files (always visible if any)
    if stats.get("failed_files") and not files:
        click.echo(
            f"\n{click.style('Failed Files', fg='red', bold=True)} ({len(stats['failed_files'])})"
        )
        click.echo("-" * 40)
        for f in stats["failed_files"]:
            status = click.style("[X]", fg="red")
            click.echo(f"  {status} {f}")

    # Show uncovered files with --uncovered flag
    if uncovered and stats.get("uncovered_files") and not files:
        click.echo(f"\nUncovered Files ({len(stats['uncovered_files'])})")
        click.echo("-" * 40)
        for f in stats["uncovered_files"]:
            status = click.style("-", fg="white", dim=True)
            click.echo(f"  {status} {f}")

    if tree:
        print_coverage_tree(stats)


def _print_state_bars(by_state: dict, total: int):
    """Print progress bars for each state."""
    width = 24
    states = [("passed", "green", "[+]"), ("failed", "red", "[X]"), ("uncovered", "white", "[-]")]

    for state, color, icon in states:
        count = by_state.get(state, 0)
        pct = (count / total * 100) if total > 0 else 0
        filled = int(width * pct / 100)
        bar_filled = click.style("#" * filled, fg=color)
        bar_empty = "-" * (width - filled)
        label = f"{state.capitalize()}:".ljust(11)
        icon_styled = click.style(icon, fg=color)
        click.echo(f"  {label} [{bar_filled}{bar_empty}] {count:4} ({pct:5.1f}%)  {icon_styled}")


def _print_files_table(stats: dict, line_stats: dict = None):
    """Print per-file coverage table."""
    files_detail = stats.get("files_detail", {})
    line_files = line_stats.get("files", {}) if line_stats else {}

    click.echo("\nPer-File Coverage")
    click.echo("-" * 70)
    click.echo(f"  {'File':<40} {'Tests':>5}  {'Status':>8}  {'Lines':>8}")
    click.echo("-" * 70)

    for file_path in sorted(files_detail.keys()):
        detail = files_detail[file_path]
        test_count = detail.get("test_count", 0)
        status = detail.get("status", "uncovered")

        # Get line coverage percentage
        line_pct = detail.get("line_coverage_pct")
        if line_pct is None and file_path in line_files:
            line_pct = line_files[file_path].get("percentage", 0.0)

        # Format status with icon and color
        if status == "passed":
            status_str = click.style("[+] pass", fg="green")
        elif status == "failed":
            status_str = click.style("[X] fail", fg="red")
        else:
            status_str = click.style("- none", fg="white", dim=True)

        # Format line coverage
        if line_pct is not None:
            line_str = f"{line_pct:5.1f}%"
        else:
            line_str = "    -"

        # Truncate long file paths
        display_path = file_path if len(file_path) <= 40 else "..." + file_path[-37:]

        click.echo(f"  {display_path:<40} {test_count:>5}  {status_str}  {line_str:>8}")

    click.echo("-" * 70)


def print_line_coverage_text(stats: dict, detailed: bool = False):
    """Print line-level coverage in text format.

    Args:
        stats: Line coverage stats from get_line_coverage_stats()
        detailed: If True, show per-file breakdown with line ranges
    """
    click.echo("\nLine-Level Coverage")
    click.echo("=" * 20)

    total_lines = stats.get("total_lines", 0)
    covered_lines = stats.get("covered_lines", 0)
    failed_lines_count = stats.get("failed_lines", 0)
    pct = stats.get("percentage", 0.0)

    if total_lines == 0:
        click.echo("\nNo line coverage data available.")
        click.echo("Run tests with a provider configured to collect line coverage.")
        return

    # Determine color based on percentage and failures
    if failed_lines_count > 0:
        pct_color = "red"
    elif pct >= 80:
        pct_color = "green"
    elif pct >= 50:
        pct_color = "yellow"
    else:
        pct_color = "red"

    pct_str = click.style(f"{pct:.1f}%", fg=pct_color, bold=True)
    summary = f"\nOverall: {pct_str} ({covered_lines}/{total_lines} lines)"
    if failed_lines_count > 0:
        failed_str = click.style(f"{failed_lines_count} failed", fg="red", bold=True)
        summary += f", {failed_str}"
    click.echo(summary)

    # Progress bar with color
    width = 28
    filled = int(width * pct / 100)
    bar_filled = click.style("#" * filled, fg=pct_color)
    bar_empty = "-" * (width - filled)
    click.echo(f"\n  [{bar_filled}{bar_empty}] {pct:.1f}%")

    if detailed and stats.get("files"):
        click.echo("\nPer-File Line Coverage")
        click.echo("-" * 40)

        for file_path in sorted(stats["files"].keys()):
            data = stats["files"][file_path]
            file_pct = data.get("percentage", 0.0)
            file_total = data.get("total_lines", 0)
            file_count = data.get("covered_count", 0)
            file_failed = data.get("failed_count", 0)
            file_status = data.get("status", "uncovered")

            # Color based on status and percentage
            if file_status == "failed":
                file_color = "red"
                status_icon = click.style("[X]", fg="red")
            elif file_pct >= 80:
                file_color = "green"
                status_icon = click.style("[+]", fg="green")
            elif file_pct >= 50:
                file_color = "yellow"
                status_icon = click.style("[*]", fg="yellow")
            else:
                file_color = "red"
                status_icon = click.style("[*]", fg="red")

            file_pct_str = click.style(f"{file_pct:.1f}%", fg=file_color)
            click.echo(f"\n  {status_icon} {file_path}")
            coverage_summary = f"    Coverage: {file_pct_str} ({file_count}/{file_total} lines)"
            if file_failed > 0:
                failed_str = click.style(f"{file_failed} failed", fg="red")
                coverage_summary += f", {failed_str}"
            click.echo(coverage_summary)

            # Show covered line ranges
            covered = data.get("covered_lines", [])
            if covered:
                ranges = _compress_line_ranges(covered)
                click.echo(f"    Covered: {ranges}")

            # Show failed line ranges
            failed = data.get("failed_lines", [])
            if failed:
                ranges = _compress_line_ranges(failed)
                failed_ranges = click.style(ranges, fg="red")
                click.echo(f"    Failed: {failed_ranges}")

            # Show incorrect lines with reasons
            incorrect = data.get("incorrect_lines", [])
            if incorrect:
                click.echo(
                    f"    {click.style('Potentially Incorrect Lines:', fg='red', bold=True)}"
                )
                for item in incorrect:
                    line_num = item.get("line_number", 0)
                    reason = item.get("reason", "Unknown")
                    line_str = click.style(f"Line {line_num}", fg="red", bold=True)
                    click.echo(f"      {line_str}: {reason}")

    # Show failure analysis summary if available
    failure_analysis = stats.get("failure_analysis", {})
    if failure_analysis:
        click.echo(f"\n{click.style('Failure Analysis', fg='red', bold=True)}")
        click.echo("-" * 40)
        for file_path, analyses in failure_analysis.items():
            click.echo(f"\n  {click.style(file_path, bold=True)}")
            for test_id, analysis in analyses.items():
                click.echo(f"    Test: {test_id}")
                analysis_text = analysis.get("analysis", "No analysis")
                click.echo(f"    Analysis: {analysis_text}")
                incorrect = analysis.get("incorrect_lines", [])
                if incorrect:
                    for item in incorrect:
                        line_num = item.get("line_number", 0)
                        reason = item.get("reason", "Unknown")
                        line_str = click.style(f"Line {line_num}", fg="red")
                        click.echo(f"      {line_str}: {reason}")


def _compress_line_ranges(lines: List[int]) -> str:
    """Compress list of lines into ranges like '1-5, 10, 15-20'.

    Args:
        lines: List of line numbers (1-indexed)

    Returns:
        Formatted string with compressed ranges
    """
    if not lines:
        return "none"

    lines = sorted(lines)
    ranges = []
    start = lines[0]
    end = lines[0]

    for line in lines[1:]:
        if line == end + 1:
            end = line
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = line
            end = line

    # Add the last range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ", ".join(ranges)


def coverage_to_lcov(stats: dict) -> str:
    """Convert line coverage stats to LCOV format for IDE integration.

    Args:
        stats: Line coverage stats from get_line_coverage_stats()

    Returns:
        LCOV formatted string
    """
    lines = []

    for file_path, data in stats.get("files", {}).items():
        lines.append(f"SF:{file_path}")

        covered_set = set(data.get("covered_lines", []))
        total = data.get("total_lines", 0)

        for line_num in range(1, total + 1):
            hit = 1 if line_num in covered_set else 0
            lines.append(f"DA:{line_num},{hit}")

        lines.append(f"LH:{len(covered_set)}")
        lines.append(f"LF:{total}")
        lines.append("end_of_record")

    return "\n".join(lines)


def print_file_with_coverage(file_path: str, lines: List[str], coverage_data: dict):
    """Display file content with line-by-line coverage highlighting.

    Args:
        file_path: Path to the file being displayed
        lines: List of file lines (content)
        coverage_data: Per-file coverage data from get_line_coverage_stats()['files'][path]
            Expected keys: covered_lines, failed_lines, incorrect_lines, total_lines,
                          covered_count, failed_count, percentage
    """
    covered = set(coverage_data.get("covered_lines", []))
    failed = set(coverage_data.get("failed_lines", []))
    incorrect = {item["line_number"]: item for item in coverage_data.get("incorrect_lines", [])}

    failed_count = coverage_data.get("failed_count", len(failed))
    percentage = coverage_data.get("percentage", 0.0)

    # Header with stats
    header = f"{file_path} ({percentage:.0f}% covered"
    if failed_count > 0:
        header += f", {failed_count} lines failed"
    header += ")"
    click.echo(f"\n{header}")
    click.echo("─" * min(len(header), 60))

    # Calculate line number width for alignment
    line_num_width = len(str(len(lines)))

    # Print each line with status indicator
    for i, line_content in enumerate(lines, 1):
        line_num_str = str(i).rjust(line_num_width)

        # Determine line status (priority: incorrect > failed > covered > blank > uncovered)
        if i in incorrect:
            # Incorrect line - show with annotation
            indicator = click.style("[!]", fg="yellow", bold=True)
            annotation = click.style("  <- incorrect", fg="yellow")
            click.echo(f"{line_num_str} {indicator} {line_content.rstrip()}{annotation}")
        elif i in failed:
            # Failed line
            indicator = click.style("[X]", fg="red")
            click.echo(f"{line_num_str} {indicator} {line_content.rstrip()}")
        elif i in covered:
            # Passed line
            indicator = click.style("[+]", fg="green")
            click.echo(f"{line_num_str} {indicator} {line_content.rstrip()}")
        elif not line_content.strip():
            # Blank/whitespace line
            indicator = "[ ]"
            click.echo(f"{line_num_str} {indicator} {line_content.rstrip()}")
        else:
            # Uncovered line
            indicator = click.style("[-]", fg="white", dim=True)
            line_styled = click.style(line_content.rstrip(), dim=True)
            click.echo(f"{line_num_str} {indicator} {line_styled}")

    # Legend
    click.echo()
    legend_parts = [
        click.style("[+]", fg="green") + " passed",
        click.style("[X]", fg="red") + " failed",
        click.style("[!]", fg="yellow") + " incorrect",
        click.style("[-]", dim=True) + " uncovered",
        "[ ] blank",
    ]
    click.echo("Legend: " + "  ".join(legend_parts))


def file_coverage_to_dict(file_path: str, lines: List[str], coverage_data: dict) -> dict:
    """Convert file coverage to JSON-serializable dict.

    Args:
        file_path: Path to the file
        lines: List of file lines (content)
        coverage_data: Per-file coverage data from get_line_coverage_stats()['files'][path]

    Returns:
        Dictionary with line-by-line coverage data
    """
    covered = set(coverage_data.get("covered_lines", []))
    failed = set(coverage_data.get("failed_lines", []))
    incorrect = {item["line_number"]: item for item in coverage_data.get("incorrect_lines", [])}

    line_data = []
    for i, line_content in enumerate(lines, 1):
        status = "uncovered"
        if i in incorrect:
            status = "incorrect"
        elif i in failed:
            status = "failed"
        elif i in covered:
            status = "passed"
        elif not line_content.strip():
            status = "blank"

        entry = {"line_number": i, "content": line_content.rstrip(), "status": status}

        if i in incorrect:
            entry["reason"] = incorrect[i].get("reason", "")

        line_data.append(entry)

    return {
        "file_path": file_path,
        "total_lines": len(lines),
        "covered_count": coverage_data.get("covered_count", 0),
        "failed_count": coverage_data.get("failed_count", 0),
        "percentage": coverage_data.get("percentage", 0.0),
        "lines": line_data,
    }


def print_coverage_tree(stats: dict):
    """Print coverage as directory tree with status (passed/failed/uncovered)."""
    all_files = (
        stats.get("covered_files", [])
        + stats.get("failed_files", [])
        + stats.get("uncovered_files", [])
    )
    covered_set = set(stats.get("covered_files", []))
    failed_set = set(stats.get("failed_files", []))

    if not all_files:
        return

    click.echo("\nDirectory Tree")
    click.echo("-" * 40)

    # Group by directory with status
    dirs: Dict[str, List[Tuple[str, str]]] = {}  # file_name, status
    for f in sorted(set(all_files)):  # Use set to avoid duplicates
        parts = f.split("/")
        if len(parts) > 1:
            dir_name = "/".join(parts[:-1])
            file_name = parts[-1]
        else:
            dir_name = "."
            file_name = f

        if dir_name not in dirs:
            dirs[dir_name] = []

        if f in failed_set:
            status = "failed"
        elif f in covered_set:
            status = "passed"
        else:
            status = "uncovered"

        dirs[dir_name].append((file_name, status))

    for dir_name in sorted(dirs.keys()):
        files_in_dir = dirs[dir_name]
        covered_in_dir = sum(1 for _, s in files_in_dir if s == "passed")
        failed_in_dir = sum(1 for _, s in files_in_dir if s == "failed")
        total_in_dir = len(files_in_dir)

        dir_summary = f"  {dir_name}/ ({covered_in_dir}/{total_in_dir}"
        if failed_in_dir > 0:
            dir_summary += click.style(f", {failed_in_dir} failed", fg="red")
        dir_summary += ")"
        click.echo(dir_summary)

        for file_name, file_status in files_in_dir:
            if file_status == "failed":
                status = click.style("[X]", fg="red")
            elif file_status == "passed":
                status = click.style("[+]", fg="green")
            else:
                status = click.style("[-]", fg="white", dim=True)
            click.echo(f"    {status} {file_name}")


# =============================================================================
# Test Results Formatters
# =============================================================================


def results_to_dict(results) -> dict:
    """Convert TestSuiteResults to dict for JSON output."""
    tests_output = []
    for r in results.test_results:
        test_data = {
            "id": r.test_id,
            "passed": r.passed,
            "duration": round(r.duration, 2) if hasattr(r, "duration") else 0,
        }
        # Include failure reasons if test failed
        if not r.passed and hasattr(r, "failure_reasons"):
            test_data["failure_reasons"] = r.failure_reasons
        # Include failure analysis if available
        if hasattr(r, "failure_analysis") and r.failure_analysis:
            test_data["failure_analysis"] = {
                file_path: {
                    "referenced_lines": analysis.referenced_lines,
                    "incorrect_lines": [
                        {"line_number": il.line_number, "reason": il.reason}
                        for il in analysis.incorrect_lines
                    ],
                    "analysis": analysis.analysis,
                }
                for file_path, analysis in r.failure_analysis.items()
            }
        tests_output.append(test_data)

    return {
        "total": results.total_tests,
        "passed": results.passed,
        "failed": results.failed,
        "skipped": results.skipped,
        "duration": round(results.duration, 2),
        "cached": results.cached_results,
        "tests": tests_output,
    }


def results_to_junit(results) -> str:
    """Convert TestSuiteResults to JUnit XML format."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuite name="dokumen" tests="{results.total_tests}" '
        f'failures="{results.failed + results.error}" time="{results.duration:.2f}">',
    ]

    for r in results.test_results:
        duration = r.duration if hasattr(r, "duration") else 0
        if r.passed:
            lines.append(f'  <testcase name="{r.test_id}" time="{duration:.2f}"/>')
        else:
            lines.append(f'  <testcase name="{r.test_id}" time="{duration:.2f}">')
            reason = getattr(r, "failure_reasons", ["Unknown failure"])
            lines.append(
                f'    <failure message="{"; ".join(reason) if isinstance(reason, list) else reason}"/>'
            )
            lines.append("  </testcase>")

    lines.append("</testsuite>")
    return "\n".join(lines)


def results_to_tap(results) -> str:
    """Convert TestSuiteResults to TAP format."""
    lines = ["TAP version 13", f"1..{results.total_tests}"]

    for i, r in enumerate(results.test_results, 1):
        status = "ok" if r.passed else "not ok"
        lines.append(f"{status} {i} - {r.test_id}")

    return "\n".join(lines)


def print_results_text(results, verbose=False):
    """Print test results in text format."""
    click.echo()
    click.echo(click.style("Skill Tests", bold=True))
    click.echo("=" * 11)

    # Summary
    total = results.total_tests
    passed = results.passed
    failed = results.failed
    cached = results.cached_results
    # Adjust passed count to exclude cached (they weren't actually run)
    actually_passed = passed - cached

    passed_str = click.style(
        f"{actually_passed} passed", fg="green" if actually_passed > 0 else "white"
    )
    failed_str = click.style(f"{failed} failed", fg="red" if failed > 0 else "white")
    skipped_str = click.style(f"{cached} skipped", fg="yellow" if cached > 0 else "white")
    click.echo(f"\nResults: {passed_str}, {failed_str}, {skipped_str}, {total} total")
    click.echo(f"Duration: {results.duration:.1f}s")

    # Get cached test IDs if available
    cached_test_ids = getattr(results, "cached_test_ids", set())

    # Individual results
    if verbose or failed > 0:
        click.echo("\nTest Results:")
        click.echo("-" * 40)
        for r in results.test_results:
            if r.test_id in cached_test_ids:
                status = click.style("SKIP", fg="yellow", bold=True)
                click.echo(f"  [{status}] {r.test_id} (cached)")
            elif r.passed:
                status = click.style("PASS", fg="green", bold=True)
                click.echo(f"  [{status}] {r.test_id}")
            else:
                status = click.style("FAIL", fg="red", bold=True)
                click.echo(f"  [{status}] {r.test_id}")

            if not r.passed and hasattr(r, "failure_reasons"):
                for reason in r.failure_reasons or []:
                    reason_str = click.style(reason, fg="red")
                    click.echo(f"         {reason_str}")

            # Show failure analysis for failed tests
            if not r.passed and hasattr(r, "failure_analysis") and r.failure_analysis:
                for file_path, analysis in r.failure_analysis.items():
                    click.echo(
                        f"\n         {click.style('Failure Analysis:', fg='red', bold=True)} {file_path}"
                    )
                    click.echo(f"         {analysis.analysis}")
                    if analysis.incorrect_lines:
                        click.echo(
                            f"         {click.style('Potentially Incorrect Lines:', fg='red')}"
                        )
                        for il in analysis.incorrect_lines:
                            line_str = click.style(f"Line {il.line_number}", fg="red", bold=True)
                            click.echo(f"           {line_str}: {il.reason}")

            # Show verbose details
            if verbose and hasattr(r, "executor_output") and r.executor_output:
                eo = r.executor_output
                tool_count = click.style(f"{len(eo.tool_calls)}", fg="cyan")
                click.echo(f"         Tool calls: {tool_count}")
                for tc in eo.tool_calls:
                    if isinstance(tc, dict):
                        raw_name = tc.get("tool_name") or tc.get("name") or "unknown"
                        raw_params = (
                            tc.get("parameters") or tc.get("tool_input") or tc.get("input") or {}
                        )
                    else:
                        raw_name = getattr(tc, "tool_name", "unknown")
                        raw_params = getattr(tc, "parameters", {})
                    tool_name = click.style(str(raw_name), fg="blue")
                    click.echo(f"           - {tool_name}({raw_params})")
                if eo.final_response:
                    click.echo()
                    click.echo(
                        click.style("         --- Executor Response ---", fg="cyan", bold=True)
                    )
                    for line in eo.final_response.strip().split("\n"):
                        click.echo(f"         {line}")
                    click.echo(click.style("         -------------------------", fg="cyan"))

            # Show judge results in verbose mode
            if verbose and hasattr(r, "judge_results") and r.judge_results:
                for jr in r.judge_results:
                    judge_status = (
                        click.style("PASS", fg="green")
                        if jr.passed
                        else click.style("FAIL", fg="red")
                    )
                    click.echo(f"         Judge [{jr.judge_id}]: {judge_status}")
                    if jr.response:
                        click.echo()
                        click.echo(
                            click.style("         --- Judge Response ---", fg="magenta", bold=True)
                        )
                        for line in jr.response.strip().split("\n"):
                            click.echo(f"         {line}")
                        click.echo(click.style("         ----------------------", fg="magenta"))


# =============================================================================
# Progress Callback
# =============================================================================


def _print_tool_provenance(provenance: dict) -> None:
    """Print tool provenance grouped by source.

    Args:
        provenance: Dict with executor_tools, judge_tools, explore_tools,
                    overrides_active, and removed_tools.
    """
    executor_tools = provenance.get("executor_tools", {})
    judge_tools = provenance.get("judge_tools", {})
    explore_tools = provenance.get("explore_tools", {})
    overrides_active = provenance.get("overrides_active", False)
    removed_tools = provenance.get("removed_tools", [])

    # Group executor tools by source
    if executor_tools:
        groups: Dict[str, List[str]] = {}
        for tool_name, source in executor_tools.items():
            groups.setdefault(source, []).append(tool_name)

        click.echo("         Executor tools:")
        for source, tools in groups.items():
            tool_list = ", ".join(sorted(tools))
            source_styled = click.style(f"({source})", dim=True)
            click.echo(f"           {tool_list} {source_styled}")

    # Print judge tools
    for judge_name, tools in judge_tools.items():
        if tools:
            tool_list = ", ".join(sorted(tools.keys()))
            sources = sorted(set(tools.values()))
            source_str = ", ".join(sources)
            source_styled = click.style(f"({source_str})", dim=True)
            click.echo(f"         Judge [{judge_name}]: [{tool_list}] {source_styled}")

    # Print explore tools
    if explore_tools:
        tool_list = ", ".join(sorted(explore_tools.keys()))
        sources = sorted(set(explore_tools.values()))
        source_str = ", ".join(sources)
        source_styled = click.style(f"({source_str})", dim=True)
        click.echo(f"         Explore: [{tool_list}] {source_styled}")

    # Show overrides indicator
    if overrides_active:
        click.echo("         " + click.style("Overrides: active", dim=True))

    # Show filtered-out tools
    if removed_tools:
        removed_list = ", ".join(sorted(removed_tools))
        click.echo("         " + click.style(f"Filtered out: [{removed_list}]", dim=True))


def make_progress_callback(quiet: bool = False, cached_tests: set = None, verbose: bool = False):
    """Create a progress callback for test execution.

    Flushes stdout after each echo to ensure immediate output visibility during
    long-running tests, preventing the appearance of "no output" when tests
    execute without cache.

    Args:
        quiet: If True, suppress all progress output.
        cached_tests: Set to track cached test IDs.
        verbose: If True, print tool usage after each test completes (legacy, use make_tool_call_callback for streaming).
    """
    import sys

    def on_progress(event: str, test_id: str, data):
        if event == "cached" and cached_tests is not None:
            cached_tests.add(test_id)
        if quiet:
            return
        # Flush stdout after echo to ensure immediate output during async test execution
        if event == "start":
            click.echo(click.style("  RUN  ", fg="cyan", bold=True) + f" {test_id}")
            if data and isinstance(data, dict):
                provenance = data.get("tool_provenance")
                if provenance:
                    _print_tool_provenance(provenance)
                elif data.get("tools"):
                    tool_list = ", ".join(data["tools"])
                    click.echo(f"         Tools: [{tool_list}]")
            sys.stdout.flush()
        elif event == "complete":
            if data and data.passed:
                click.echo(click.style("  PASS ", fg="green", bold=True) + f" {test_id}")
            else:
                click.echo(click.style("  FAIL ", fg="red", bold=True) + f" {test_id}")
            sys.stdout.flush()
        elif event == "cached":
            click.echo(click.style("  SKIP ", fg="yellow", bold=True) + f" {test_id} (cached)")
            sys.stdout.flush()

    return on_progress


def make_tool_call_callback(quiet: bool = False, verbose: bool = False):
    """Create a callback that prints tool calls as they are executed.

    This callback is fired in real-time as each tool is called during test execution,
    allowing users to see tool usage as it happens (streaming).

    Args:
        quiet: If True, suppress all output (unless verbose overrides).
        verbose: If True, overrides quiet to show output.

    Returns:
        Callback function with signature (tool_name: str, params: dict, result: Any) -> None
    """
    import sys

    def on_tool_call(tool_name: str, params: dict, result):
        # verbose overrides quiet for streaming output
        if quiet and not verbose:
            return
        # Format parameters as a concise string
        if isinstance(params, dict):
            params_str = ", ".join(f"{v}" for v in params.values())
        else:
            params_str = str(params) if params else ""
        click.echo(click.style("         - ", fg="cyan") + f"{tool_name}({params_str})")
        sys.stdout.flush()

    return on_tool_call


def make_conversation_callback(quiet: bool = False, verbose: bool = False):
    """Create a callback that prints conversation messages as they occur.

    This callback streams system prompts, user prompts, and model responses
    in real-time during executor and judge execution.

    Args:
        quiet: If True, suppress all output.
        verbose: If True, show conversation messages. If False, suppress output.

    Returns:
        Callback function with signature:
        (agent_type: str, message_type: str, content: str) -> None

        agent_type: "executor" or "judge"
        message_type: "system", "user", "assistant", "thinking"
        content: The message content
    """
    import sys

    def on_conversation_message(agent_type: str, message_type: str, content: str):
        # verbose overrides quiet for streaming output
        if not verbose:
            return

        # Format header based on agent type and message type
        if message_type == "system":
            header = click.style(f"[{agent_type.upper()}] System Prompt:", fg="cyan", bold=True)
        elif message_type == "user":
            header = click.style(f"[{agent_type.upper()}] User Prompt:", fg="cyan", bold=True)
        elif message_type == "assistant":
            header = click.style(f"[{agent_type.upper()}] Response:", fg="green", bold=True)
        elif message_type == "thinking":
            header = click.style(f"[{agent_type.upper()}] Thinking:", fg="yellow")
        else:
            header = click.style(f"[{agent_type.upper()}] {message_type}:", fg="white")

        click.echo(f"\n{header}")
        # Indent content for readability
        for line in content.strip().split("\n"):
            click.echo(f"    {line}")
        sys.stdout.flush()

    return on_conversation_message


def make_executor_complete_callback(quiet: bool = False, verbose: bool = False):
    """Create a callback that prints executor completion summary.

    Args:
        quiet: If True, suppress all output.
        verbose: If True, show full response. If False, show truncated summary.

    Returns:
        Callback function with signature:
        (test_id: str, executor_output: ExecutorOutput) -> None
    """
    import sys

    def on_executor_complete(test_id: str, executor_output):
        # verbose overrides quiet for streaming output
        if quiet and not verbose:
            return

        header = click.style("--- Executor Complete ---", fg="cyan", bold=True)
        click.echo(f"\n{header}")

        if executor_output.success:
            status = click.style("SUCCESS", fg="green", bold=True)
        else:
            status = click.style("FAILED", fg="red", bold=True)

        click.echo(f"    Status: {status}")

        if not executor_output.success and executor_output.error:
            click.echo(f"    Error: {executor_output.error}")

        click.echo(f"    Tool calls: {len(executor_output.tool_calls)}")

        if verbose and executor_output.final_response:
            click.echo(click.style("    Final Response:", fg="cyan"))
            # Show full response in verbose mode
            for line in executor_output.final_response.strip().split("\n"):
                click.echo(f"      {line}")
        elif executor_output.final_response:
            # Show truncated response
            response = executor_output.final_response.strip()
            if len(response) > 200:
                response = response[:200] + "..."
            click.echo(f"    Response: {response}")

        sys.stdout.flush()

    return on_executor_complete


def make_judge_complete_callback(quiet: bool = False, verbose: bool = False):
    """Create a callback that prints judge completion summary.

    Args:
        quiet: If True, suppress all output.
        verbose: If True, show full reasoning. If False, show summary.

    Returns:
        Callback function with signature:
        (test_id: str, judge_result: JudgeResult) -> None
    """
    import sys

    def on_judge_complete(test_id: str, judge_result):
        # verbose overrides quiet for streaming output
        if quiet and not verbose:
            return

        if judge_result.passed:
            verdict = click.style("PASS", fg="green", bold=True)
        else:
            verdict = click.style("FAIL", fg="red", bold=True)

        header = click.style(f"--- Judge [{judge_result.judge_id}] ---", fg="magenta", bold=True)
        click.echo(f"\n{header}")
        click.echo(f"    Verdict: {verdict}")

        if not judge_result.passed and judge_result.failure_reason:
            reason = click.style(judge_result.failure_reason, fg="red")
            click.echo(f"    Reason: {reason}")

        if verbose and judge_result.response:
            click.echo(click.style("    Full Response:", fg="magenta"))
            for line in judge_result.response.strip().split("\n"):
                click.echo(f"      {line}")

        sys.stdout.flush()

    return on_judge_complete


# =============================================================================
# Explore Formatters
# =============================================================================


def make_explore_callback(quiet: bool = False):
    """Create a callback that prints explore events as they occur.

    This callback is fired during the explore phase to show progress.

    Args:
        quiet: If True, suppress all output.

    Returns:
        Callback function with signature (event_type: str, data: dict) -> None
        Events: 'start', 'file_found', 'complete'
    """
    import sys

    def on_explore_event(event_type: str, data: dict):
        if quiet:
            return

        if event_type == "start":
            goal = data.get("goal", "documentation")
            click.echo(click.style("  EXPLORE", fg="magenta", bold=True) + f" Finding {goal}...")
            sys.stdout.flush()

        elif event_type == "file_found":
            path = data.get("path", "")
            click.echo(click.style("    Found:", fg="magenta") + f" {path}")
            sys.stdout.flush()

        elif event_type == "complete":
            files_found = data.get("files_found", 0)
            duration = data.get("duration", 0)
            file_word = "file" if files_found == 1 else "files"
            click.echo(
                click.style("  EXPLORE", fg="magenta", bold=True)
                + f" Complete ({files_found} {file_word}, {duration:.1f}s)"
            )
            sys.stdout.flush()

    return on_explore_event


def explore_to_dict(result) -> dict:
    """Convert ExploreResult to dict for JSON output.

    Args:
        result: ExploreResult instance

    Returns:
        Dictionary with explore results
    """
    output = {
        "success": result.success,
        "duration": result.duration,
        "tool_calls_count": result.tool_calls_count,
        "files": [
            {"path": f.path, "summary": f.summary, "relevance": f.relevance} for f in result.files
        ],
    }

    if result.error:
        output["error"] = result.error

    return output


def explore_event_to_json(event_type: str, test_id: str, data: dict) -> dict:
    """Format explore event as JSON dict for streaming output.

    Args:
        event_type: Type of event ('start', 'file_found', 'complete')
        test_id: ID of the test being explored
        data: Event-specific data

    Returns:
        Dictionary ready for JSON serialization
    """
    if event_type == "start":
        return {"event": "explore_start", "test_id": test_id, "goal": data.get("goal", "")}

    elif event_type == "file_found":
        return {
            "event": "explore_file",
            "test_id": test_id,
            "path": data.get("path", ""),
            "summary": data.get("summary", ""),
        }

    elif event_type == "complete":
        return {
            "event": "explore_complete",
            "test_id": test_id,
            "files_found": data.get("files_found", 0),
            "duration_ms": int(data.get("duration", 0) * 1000),
        }

    return {"event": event_type, "test_id": test_id, **data}
