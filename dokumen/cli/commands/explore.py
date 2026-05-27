"""
Explore command for dokumen CLI.

Discovers relevant documentation files using an AI agent via the Claude Agent SDK.
"""
import json
import logging
import os
import sys
import time
from typing import Optional

import click

from ..helpers import load_config, run_async

# Configure logger for explore command
logger = logging.getLogger(__name__)


async def _run_explore(
    topic: str,
    max_files: int = 20,
    timeout: int = 60,
    config: Optional[dict] = None,
) -> "ExploreResult":
    """Run the explore agent to discover relevant files.

    This function is the core exploration logic, separated for testability.
    Uses the SDK-based ExploreAgent with read-only tools (Read, Glob, Grep).

    Args:
        topic: The topic to explore
        max_files: Maximum number of files to return
        timeout: Timeout in seconds
        config: Optional config dict (if None, uses defaults)

    Returns:
        ExploreResult with discovered files
    """
    from dokumen.explore_agent import ExploreAgent
    from dokumen.config import DEFAULT_EXPLORE_TOOL_NAMES

    logger.info(f"[EXPLORE_CLI] Starting exploration for topic: {topic!r}")
    logger.info(f"[EXPLORE_CLI] Parameters: max_files={max_files}, timeout={timeout}")
    logger.debug(f"[EXPLORE_CLI] Working directory: {os.getcwd()}")

    start_time = time.time()

    # Get explore config from dokumen.yaml or use defaults
    explore_config = {}
    if config:
        explore_config = config.get("explore", {})
        logger.info(f"[EXPLORE_CLI] Loaded explore config from dokumen.yaml: {explore_config}")
    else:
        logger.info("[EXPLORE_CLI] No config provided, using defaults")

    # Override with CLI options if provided
    if max_files != 20:  # If not default, use CLI value
        explore_config["max_files"] = max_files
        logger.debug(f"[EXPLORE_CLI] Overriding max_files from CLI: {max_files}")
    elif "max_files" not in explore_config:
        explore_config["max_files"] = max_files

    if timeout != 60:  # If not default, use CLI value
        explore_config["timeout"] = timeout
        logger.debug(f"[EXPLORE_CLI] Overriding timeout from CLI: {timeout}")
    elif "timeout" not in explore_config:
        explore_config["timeout"] = timeout

    # Get the model from config
    model = explore_config.get("model")
    logger.info(f"[EXPLORE_CLI] Using model: {model}")

    # Create explore agent using SDK
    logger.info(
        f"[EXPLORE_CLI] Creating ExploreAgent with max_files={explore_config.get('max_files', 20)}, "
        f"max_turns={explore_config.get('max_iterations', 50)}, "
        f"timeout={explore_config.get('timeout', 60)}"
    )
    agent = ExploreAgent(
        base_dir=".",
        max_files=explore_config.get("max_files", 20),
        max_turns=explore_config.get("max_iterations", 50),
        timeout=float(explore_config.get("timeout", 60)),
        model=model,
    )

    # Run exploration
    logger.info(f"[EXPLORE_CLI] Starting agent exploration for: {topic!r}")
    result = await agent.explore(topic)

    elapsed = time.time() - start_time
    logger.info(f"[EXPLORE_CLI] Exploration completed in {elapsed:.2f}s")
    logger.info(f"[EXPLORE_CLI] Result: success={result.success}, files_count={len(result.files) if hasattr(result, 'files') else 0}, tool_calls={result.tool_calls_count}")
    if result.summary:
        logger.info(f"[EXPLORE_CLI] Summary preview: {result.summary[:200]}...")

    return result


def _format_text_output(result) -> str:
    """Format ExploreResult as human-readable text.

    Args:
        result: ExploreResult object

    Returns:
        Formatted text string
    """
    lines = []

    if result.success:
        lines.append(click.style("Exploration Complete", fg="green", bold=True))
    else:
        lines.append(click.style("Exploration Failed", fg="red", bold=True))
        if hasattr(result, "error") and result.error:
            lines.append(f"Error: {result.error}")

    lines.append("")

    # Summary
    if result.summary:
        lines.append(click.style("Summary:", bold=True))
        lines.append(result.summary)
        lines.append("")

    # Files found
    if hasattr(result, "files") and result.files:
        lines.append(click.style(f"Files Found ({len(result.files)}):", bold=True))
        for f in result.files:
            path = f.path if hasattr(f, "path") else f.get("path", "")
            summary = f.summary if hasattr(f, "summary") else f.get("summary", "")
            relevance = f.relevance if hasattr(f, "relevance") else f.get("relevance", 0)
            lines.append(f"  {click.style(path, fg='cyan')} ({relevance:.0%})")
            if summary:
                lines.append(f"    {summary}")
    else:
        lines.append(click.style("No files found.", fg="yellow"))

    lines.append("")

    # Stats
    lines.append(click.style("Stats:", bold=True))
    lines.append(f"  Duration: {result.duration:.2f}s")
    lines.append(f"  Tool calls: {result.tool_calls_count}")

    return "\n".join(lines)


@click.command()
@click.argument("topic")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format (text or json)",
)
@click.option(
    "--max-files",
    "-m",
    type=int,
    default=20,
    help="Maximum number of files to return",
)
@click.option(
    "--timeout",
    "-t",
    type=int,
    default=60,
    help="Timeout in seconds",
)
@click.pass_context
def explore(ctx, topic: str, output: str, max_files: int, timeout: int):
    """Discover relevant documentation files.

    Uses an AI agent to explore the codebase and find files relevant to TOPIC.

    Examples:

        dokumen explore "margin policy"

        dokumen explore "API authentication" --output json

        dokumen explore "user guide" --max-files 10 --timeout 30
    """
    logger.info(f"[EXPLORE_CMD] Command invoked: topic={topic!r}, output={output}, max_files={max_files}, timeout={timeout}")

    # Load config if available
    config = None
    try:
        config_path = ctx.obj.get("config_path") if ctx.obj else None
        logger.debug(f"[EXPLORE_CMD] Loading config from: {config_path}")
        config = load_config(config_path)
        logger.info(f"[EXPLORE_CMD] Config loaded successfully")
    except Exception as e:
        logger.warning(f"[EXPLORE_CMD] Failed to load config: {e}, continuing with defaults")

    # Run exploration
    try:
        logger.info(f"[EXPLORE_CMD] Starting async exploration...")
        result = run_async(
            _run_explore(
                topic=topic,
                max_files=max_files,
                timeout=timeout,
                config=config,
            )
        )
        logger.info(f"[EXPLORE_CMD] Exploration completed successfully")
    except Exception as e:
        logger.error(f"[EXPLORE_CMD] Exploration failed with error: {e}", exc_info=True)
        # Handle errors gracefully
        if output == "json":
            error_output = {
                "success": False,
                "error": str(e),
                "files": [],
                "summary": "",
                "duration": 0,
                "tool_calls_count": 0,
                "tool_history": [],
            }
            click.echo(json.dumps(error_output, indent=2))
        else:
            click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)

    # Output result
    if output == "json":
        logger.debug(f"[EXPLORE_CMD] Outputting JSON result")
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        logger.debug(f"[EXPLORE_CMD] Outputting text result")
        click.echo(_format_text_output(result))

    logger.info(f"[EXPLORE_CMD] Command completed")
