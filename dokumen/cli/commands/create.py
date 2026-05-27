"""
Create command for dokumen CLI.

Generates test scaffolds from natural language goals.

Supports three modes:
1. Direct mode: dokumen create --goal "Verify refund policy"
2. Stdin mode: dokumen create --stdin (for backend integration)
3. Output to file: dokumen create --goal "..." --output tests/new.test.yaml

Stdin mode reads JSON input and outputs NDJSON events:
  Input: {"goal": "...", "files": [...], "existing_tests": [...]}
  Output: {"event": "create_start", ...}, {"event": "done", "scaffold": {...}}
"""
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from ..helpers import load_config, run_async

logger = logging.getLogger(__name__)


def _emit_event(event_type: str, data: Dict[str, Any]) -> None:
    """Emit a streaming event as NDJSON.

    Args:
        event_type: Event type.
        data: Event data.
    """
    event = {"event": event_type, **data}
    print(json.dumps(event), flush=True)


def _discover_existing_tests(tests_dir: str = "tests") -> List[str]:
    """Discover existing test names in the tests directory.

    Args:
        tests_dir: Directory containing test scaffolds.

    Returns:
        List of existing test names.
    """
    existing = []
    tests_path = Path(tests_dir)

    if not tests_path.exists():
        return existing

    for yaml_file in tests_path.glob("*.test.yaml"):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if content and "name" in content:
                    existing.append(content["name"])
        except Exception as e:
            logger.warning(f"Failed to read test file {yaml_file}: {e}")

    return existing


def _create_agent(config: Optional[dict], timeout: float):
    """Create and return a CreateAgent instance.

    Args:
        config: Optional config dict.
        timeout: Timeout in seconds.

    Returns:
        CreateAgent instance.
    """
    from dokumen.create_agent import CreateAgent
    from dokumen.loader import get_configured_provider
    from dokumen.tools_object import BUILTIN_TOOLS, create_bash_tool, create_grep_tool

    # Get provider
    provider = get_configured_provider()
    logger.info(f"[CREATE_CLI] Provider initialized: {type(provider).__name__}")

    # Create tools
    base_dir = "."
    tools = []
    for name, factory in BUILTIN_TOOLS.items():
        tools.append(factory(base_dir))

    # Add bash and grep tools
    tools.append(create_bash_tool(sandbox=None, timeout=30.0, base_dir=base_dir))
    tools.append(create_grep_tool(sandbox=None))

    # Create agent
    agent = CreateAgent(
        provider=provider,
        base_dir=base_dir,
        timeout=timeout,
        tools=tools,
    )

    return agent


def _create_progress_callback(stream: bool):
    """Create a progress callback for streaming events.

    Args:
        stream: Whether streaming is enabled.

    Returns:
        Progress callback function or None.
    """
    if not stream:
        return None

    def on_progress(event_type: str, data: dict) -> None:
        """Progress callback that emits streaming events."""
        _emit_event(event_type, data)

    return on_progress


async def _run_create(
    goal: str,
    files: Optional[List[str]] = None,
    existing_tests: Optional[List[str]] = None,
    timeout: float = 120.0,
    config: Optional[dict] = None,
    stream: bool = False,
    test_type: str = "standard",
) -> "CreateResult":
    """Run the create agent to generate a test scaffold.

    Args:
        goal: What the test should validate.
        files: Optional list of files to test.
        existing_tests: List of existing test names.
        timeout: Timeout in seconds.
        config: Optional config dict.
        stream: If True, emit NDJSON events.
        test_type: Type of test to generate ('standard' or 'browser').

    Returns:
        CreateResult with the generated scaffold.
    """
    logger.info(f"[CREATE_CLI] Starting create for goal: {goal[:100]} (type={test_type})")

    # Create agent
    agent = _create_agent(config, timeout)
    on_progress = _create_progress_callback(stream)

    # Run create
    result = await agent.create(
        goal=goal,
        files=files,
        existing_tests=existing_tests,
        on_progress=on_progress,
        test_type=test_type,
    )

    return result


async def _run_stdin_session(
    timeout: float,
    config: Optional[dict],
) -> None:
    """Run stdin-based create session for backend integration.

    Reads JSON from stdin, outputs NDJSON events.

    Args:
        timeout: Timeout in seconds.
        config: Optional config dict.
    """
    # Read JSON from stdin
    input_data = sys.stdin.read().strip()

    try:
        params = json.loads(input_data)
    except json.JSONDecodeError as e:
        _emit_event("error", {"message": f"Invalid JSON input: {e}"})
        return

    goal = params.get("goal")
    if not goal:
        _emit_event("error", {"message": "Missing required 'goal' field"})
        return

    files = params.get("files", [])
    existing_tests = params.get("existing_tests", [])

    # Read and validate test type
    test_type = params.get("type", "standard")
    if test_type not in ("standard", "browser"):
        _emit_event("error", {"message": f"Invalid type: '{test_type}'. Must be 'standard' or 'browser'"})
        return

    # Run create with streaming
    result = await _run_create(
        goal=goal,
        files=files if files else None,
        existing_tests=existing_tests if existing_tests else None,
        timeout=timeout,
        config=config,
        stream=True,
        test_type=test_type,
    )

    # Emit final result
    _emit_event("done", {
        "success": result.success,
        "name": result.name,
        "scaffold_yaml": result.scaffold_yaml,
        "scaffold_dict": result.scaffold_dict,
        "discovered_files": result.discovered_files,
        "duration": result.duration,
        "error": result.error,
        "test_type": result.test_type,
    })


@click.command()
@click.option(
    "--goal",
    "-g",
    type=str,
    help="What the test should validate (required unless --stdin)",
)
@click.option(
    "--name",
    "-n",
    type=str,
    help="Test name in kebab-case (auto-generated if not provided)",
)
@click.option(
    "--files",
    "-f",
    type=str,
    help="Comma-separated file paths to test (auto-discovered if not provided)",
)
@click.option(
    "--existing-tests",
    type=str,
    help="Comma-separated existing test names (for conflict avoidance)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (stdout if not provided)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format (yaml or json)",
)
@click.option(
    "--timeout",
    "-t",
    type=float,
    default=120.0,
    help="Timeout in seconds",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview scaffold without writing to file",
)
@click.option(
    "--type",
    "test_type",
    type=click.Choice(["standard", "browser"]),
    default="standard",
    help="Test type: standard (default) or browser",
)
@click.option(
    "--stdin",
    "stdin_mode",
    is_flag=True,
    help="Read JSON input from stdin (for backend integration)",
)
@click.pass_context
def create(
    ctx,
    goal: Optional[str],
    name: Optional[str],
    files: Optional[str],
    existing_tests: Optional[str],
    output: Optional[str],
    output_format: str,
    timeout: float,
    dry_run: bool,
    test_type: str,
    stdin_mode: bool,
):
    """Generate a test scaffold from a natural language goal.

    Creates a new documentation test by:
    1. Exploring documentation to find relevant files
    2. Generating executor prompts and judge criteria
    3. Outputting a valid test scaffold YAML

    \b
    Examples:

        # Generate scaffold and print to stdout
        dokumen create --goal "Verify refund policy handles 30-day returns"

        # Save to file
        dokumen create --goal "Verify API auth" --output tests/api-auth.test.yaml

        # Specify files to test
        dokumen create --goal "Verify margin docs" --files "docs/margin.md,docs/trading.md"

        # Backend integration (stdin mode)
        echo '{"goal": "Verify policy"}' | dokumen create --stdin
    """
    logger.info(f"[CREATE_CMD] Command invoked: goal={goal[:50] if goal else None!r}, type={test_type}, stdin={stdin_mode}")

    # Load config
    config = None
    try:
        config_path = ctx.obj.get("config_path") if ctx.obj else None
        config = load_config(config_path)
    except Exception as e:
        logger.warning(f"[CREATE_CMD] Failed to load config: {e}")

    # Stdin mode
    if stdin_mode:
        logger.info("[CREATE_CMD] Running in stdin mode")
        try:
            run_async(_run_stdin_session(timeout=timeout, config=config))
        except Exception as e:
            logger.error(f"[CREATE_CMD] Stdin session failed: {e}", exc_info=True)
            _emit_event("error", {"message": str(e)})
            sys.exit(1)
        return

    # Require goal for non-stdin mode
    if not goal:
        raise click.UsageError("--goal is required (or use --stdin for JSON input)")

    # Parse comma-separated options
    file_list = [f.strip() for f in files.split(",")] if files else None
    existing_list = [t.strip() for t in existing_tests.split(",")] if existing_tests else None

    # Auto-discover existing tests if not provided
    if not existing_list:
        existing_list = _discover_existing_tests()
        logger.info(f"[CREATE_CMD] Discovered {len(existing_list)} existing tests")

    # Run create
    try:
        result = run_async(
            _run_create(
                goal=goal,
                files=file_list,
                existing_tests=existing_list,
                timeout=timeout,
                config=config,
                stream=False,
                test_type=test_type,
            )
        )
    except Exception as e:
        logger.error(f"[CREATE_CMD] Create failed: {e}", exc_info=True)
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)

    # Handle failure
    if not result.success:
        if output_format == "json":
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            click.echo(click.style(f"Failed: {result.error}", fg="red"), err=True)
        sys.exit(1)

    # Format output
    if output_format == "json":
        output_content = json.dumps(result.to_dict(), indent=2)
    else:
        output_content = result.scaffold_yaml

    # Write to file or stdout
    if output and not dry_run:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_content)
        click.echo(click.style(f"Saved to: {output}", fg="green"))
        click.echo("")
        click.echo(output_content)
    else:
        click.echo(output_content)

    if dry_run:
        click.echo(click.style("\n(dry-run: no file written)", dim=True))

    logger.info("[CREATE_CMD] Command completed")
