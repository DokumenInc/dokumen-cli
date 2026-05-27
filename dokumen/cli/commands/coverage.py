"""
Coverage and status commands for dokumen CLI - Phase 0 (file-level only).
"""

import json
import sys
from typing import Optional

import click

from ..helpers import EXIT_FAILURE, load_config, filter_stats_by_path


def _get_coverage_stats(config):
    """Wrapper to allow patching at dokumen.cli level."""
    import dokumen.cli

    return dokumen.cli.get_coverage_stats(config=config)


@click.command()
@click.option("--files", "-f", is_flag=True, help="Show per-file coverage details")
@click.option("--uncovered", "-u", is_flag=True, help="Show only uncovered files")
@click.option("--tree", "-t", is_flag=True, help="Show directory tree with coverage")
@click.option(
    "--min",
    "-m",
    "min_threshold",
    type=int,
    help="Minimum file coverage threshold (exit 1 if below)",
)
@click.option(
    "--output", "-o", type=click.Choice(["text", "json"]), default="text", help="Output format"
)
@click.option("--verbose", "-v", is_flag=True, help="Show extra details")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
@click.argument("path", required=False)
@click.pass_context
def coverage(
    ctx,
    files: bool,
    uncovered: bool,
    tree: bool,
    min_threshold: Optional[int],
    output: str,
    verbose: bool,
    quiet: bool,
    path: Optional[str],
):
    """View source coverage (file-level).

    The primary command for understanding your test coverage.

    Examples:

        dokumen coverage              Show overall coverage

        dokumen coverage docs/        Show coverage for docs/ directory

        dokumen coverage docs/api.md  Show coverage for specific file

        dokumen coverage --uncovered  List files without tests

        dokumen coverage --min 80     Fail if coverage < 80%
    """
    from ..formatters import print_coverage_text

    config = load_config(ctx.obj.get("config_path"))
    stats = _get_coverage_stats(config=config)

    # Filter stats by path if provided
    if path:
        stats = filter_stats_by_path(stats, path)

    if output == "json":
        click.echo(json.dumps(stats, indent=2))
    else:
        print_coverage_text(
            stats, files=files, uncovered=uncovered, tree=tree, verbose=verbose, quiet=quiet
        )

    # Check file coverage threshold
    if min_threshold is not None:
        if stats["percentage"] < min_threshold:
            if output == "text":
                click.echo(
                    f"\nFile coverage {stats['percentage']:.0f}% is below threshold {min_threshold}%",
                    err=True,
                )
            sys.exit(EXIT_FAILURE)


@click.command()
@click.option("--min", "-m", "min_threshold", type=int, help="Minimum coverage threshold")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx, min_threshold: Optional[int], as_json: bool):
    """Quick coverage status for CI/CD.

    Returns exit code 1 if coverage is below threshold.

    Examples:

        dokumen status            Show quick status

        dokumen status --min 80   Fail if coverage < 80%
    """
    config = load_config(ctx.obj.get("config_path"))
    stats = _get_coverage_stats(config=config)

    pct = stats["percentage"]
    passed = stats["passed"]
    total = stats["total"]
    uncovered_count = total - passed

    if as_json:
        click.echo(
            json.dumps(
                {
                    "coverage": pct,
                    "passed": passed,
                    "total": total,
                    "uncovered": uncovered_count,
                    "threshold": min_threshold,
                    "threshold_passed": min_threshold is None or pct >= min_threshold,
                },
                indent=2,
            )
        )
    else:
        # Determine color based on coverage
        if min_threshold and pct < min_threshold:
            status_icon = click.style("[X] FAIL", fg="red", bold=True)
            pct_color = "red"
        elif pct >= 80:
            status_icon = click.style("[+] OK", fg="green", bold=True)
            pct_color = "green"
        elif pct >= 50:
            status_icon = click.style("[*] OK", fg="yellow", bold=True)
            pct_color = "yellow"
        else:
            status_icon = click.style("[*] LOW", fg="red", bold=True)
            pct_color = "red"

        pct_str = click.style(f"{pct:.0f}%", fg=pct_color, bold=True)

        if min_threshold and pct < min_threshold:
            click.echo(f"{status_icon} Coverage: {pct_str} (below {min_threshold}% threshold)")
        else:
            click.echo(f"{status_icon} Coverage: {pct_str} ({passed}/{total} files)")

        if uncovered_count > 0:
            uncovered_str = click.style(f"{uncovered_count}", fg="yellow")
            click.echo(f"     Uncovered: {uncovered_str} file(s) need tests")

    if min_threshold and pct < min_threshold:
        sys.exit(EXIT_FAILURE)
