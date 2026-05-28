"""
List command group for dokumen CLI.
"""

import json as json_module
from pathlib import Path

import click
import yaml

from ..helpers import load_config


def _get_coverage_stats(config):
    """Wrapper to allow patching at dokumen.cli level."""
    import dokumen.cli

    return dokumen.cli.get_coverage_stats(config=config)


@click.group(name="list")
@click.pass_context
def list_cmd(ctx):
    """List resources (tests, files, tools).

    Examples:

        dokumen list tests    Show all test scaffolds

        dokumen list files    Show tracked source files
    """
    pass


def _normalize_folder_path_for_list(file_path: str) -> str:
    """Extract canonical folder_path from a test file path for list command.

    Simplified version that works with local paths.
    """
    path = file_path.replace("\\", "/")

    # Strip tests/ prefix
    if path.startswith("tests/"):
        path = path[6:]

    # Get directory part
    if "/" in path:
        return path.rsplit("/", 1)[0]
    return ""


@list_cmd.command("tests")
@click.option("--verbose", "-v", is_flag=True, help="Show details")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.option("--tree", "tree_output", is_flag=True, help="Show tests grouped by folder")
@click.pass_context
def list_tests(ctx, verbose: bool, json_output: bool, tree_output: bool):
    """List all test scaffolds."""
    from ...scaffold import discover_scaffolds

    scaffolds = discover_scaffolds("tests")

    if json_output:
        # JSON output mode - always flat list with folder_path field
        tests_data = []
        for scaffold_path in sorted(scaffolds):
            try:
                with open(scaffold_path) as f:
                    data = yaml.safe_load(f) or {}

                name = data.get("name", Path(scaffold_path).stem)
                reason = data.get("reason", "")
                files_covered = len(data.get("files", []))
                folder_path = _normalize_folder_path_for_list(scaffold_path)

                tests_data.append(
                    {
                        "name": name,
                        "file": scaffold_path,
                        "folder_path": folder_path,
                        "reason": reason.strip() if reason else None,
                        "files_count": files_covered,
                    }
                )
            except (IOError, yaml.YAMLError):
                tests_data.append(
                    {
                        "name": Path(scaffold_path).stem,
                        "file": scaffold_path,
                        "folder_path": _normalize_folder_path_for_list(scaffold_path),
                        "error": "Failed to parse",
                    }
                )

        click.echo(json_module.dumps({"tests": tests_data}, indent=2))
        return

    # Text output mode
    if not scaffolds:
        click.echo("No test scaffolds found in tests/")
        click.echo("Use the repo-local authoring skill at .claude/skills/dokumen-test-author.")
        return

    if tree_output:
        # Tree output mode - group tests by folder
        click.echo(f"\nTests ({len(scaffolds)})")
        click.echo("=" * 40)

        # Group tests by folder
        folder_tests: dict[str, list[tuple[str, str, int]]] = {}
        for scaffold_path in scaffolds:
            try:
                with open(scaffold_path) as f:
                    data = yaml.safe_load(f) or {}
                name = data.get("name", Path(scaffold_path).stem)
                files_covered = len(data.get("files", []))
                folder = _normalize_folder_path_for_list(scaffold_path)
                if folder not in folder_tests:
                    folder_tests[folder] = []
                folder_tests[folder].append((name, scaffold_path, files_covered))
            except (IOError, yaml.YAMLError):
                pass

        # Sort folders alphabetically
        for folder in sorted(folder_tests.keys()):
            if folder:
                click.echo(f"\n  {folder}/")
            else:
                click.echo("\n  (root)")

            # Sort tests within folder alphabetically
            for name, path, files_count in sorted(folder_tests[folder], key=lambda x: x[0]):
                if verbose:
                    click.echo(f"    {name}")
                    click.echo(f"      Path: {path}")
                    click.echo(f"      Files: {files_count}")
                else:
                    click.echo(f"    - {name} ({files_count} file(s))")
        return

    # Standard flat output
    click.echo(f"\nTests ({len(scaffolds)})")
    click.echo("=" * 40)

    for scaffold_path in sorted(scaffolds):
        try:
            with open(scaffold_path) as f:
                data = yaml.safe_load(f) or {}

            name = data.get("name", Path(scaffold_path).stem)
            files_covered = len(data.get("files", []))

            if verbose:
                click.echo(f"\n  {name}")
                click.echo(f"    Path: {scaffold_path}")
                click.echo(f"    Files: {files_covered}")
                if data.get("reason"):
                    click.echo(f"    Reason: {data.get('reason')}")
            else:
                click.echo(f"  - {name} ({files_covered} file(s))")

        except (IOError, yaml.YAMLError) as e:
            click.echo(f"  - {scaffold_path} (error: {e})", err=True)


@list_cmd.command("files")
@click.option("--metrics", "-m", is_flag=True, help="Show coverage metrics")
@click.pass_context
def list_files(ctx, metrics: bool):
    """List tracked source files."""
    from ..formatters import print_coverage_tree

    config = load_config(ctx.obj.get("config_path"))
    stats = _get_coverage_stats(config=config)

    all_files = (
        stats.get("covered_files", [])
        + stats.get("failed_files", [])
        + stats.get("uncovered_files", [])
    )

    if not all_files:
        click.echo("No source files found.")
        click.echo("Update 'coverage.include' in dokumen.yaml.")
        return

    # Display as directory tree
    print_coverage_tree(stats)

    if metrics:
        covered_set = set(stats.get("covered_files", []))
        pct = stats["percentage"]
        if pct >= 80:
            pct_color = "green"
        elif pct >= 50:
            pct_color = "yellow"
        else:
            pct_color = "red"
        pct_str = click.style(f"{pct:.0f}%", fg=pct_color, bold=True)
        click.echo(f"\nSummary: {len(covered_set)}/{len(all_files)} files covered ({pct_str})")


@list_cmd.command("tools")
@click.pass_context
def list_tools(ctx):
    """List available tools for test scaffolds."""
    from ...tools_object import BUILTIN_TOOLS
    from ...playwright_tools import get_browser_tool_names

    core_tools = {
        "run_shell_command": "Run shell commands through the SDK Bash tool.",
        "search_file_content": "Search file contents through the SDK Grep tool.",
        "web_fetch": "Fetch and summarize web pages.",
        "web_search": "Search the web through the SDK WebSearch tool.",
    }
    agent_tools = {
        "explore": "Run a focused workspace exploration sub-agent.",
    }

    click.echo("\nAvailable Tools")
    click.echo("=" * 40)

    click.echo("\nBuilt-in Tools:")
    for name, tool_factory in BUILTIN_TOOLS.items():
        # Create a local definition to display the tool description.
        tool_def = tool_factory(".")
        desc = tool_def.description
        click.echo(f"  - {name}: {desc}")

    click.echo("\nSDK Core Tools:")
    for name, desc in core_tools.items():
        click.echo(f"  - {name}: {desc}")

    click.echo("\nBrowser Tools (Playwright MCP):")
    for name in get_browser_tool_names():
        click.echo(f"  - {name}")

    click.echo("\nOptional Local Agent Tools:")
    for name, desc in agent_tools.items():
        click.echo(f"  - {name}: {desc}")
