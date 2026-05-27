"""
Ask command for dokumen CLI.

Asks questions about documentation.

Supports three modes:
1. Single question: dokumen ask "What is X?"
2. Interactive REPL: dokumen ask (no argument)
3. Stdin mode: dokumen ask --stdin (for backend integration)

Streaming mode (--stream) outputs NDJSON events:
  {"event": "tool_start", "tool": "read_file", "params": {...}}
  {"event": "tool_end", "tool": "read_file", "result": "..."}
  {"event": "chunk", "content": "..."}
  {"event": "done", "success": true, "sources": [...]}
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from ..helpers import load_config, run_async

logger = logging.getLogger(__name__)


def _emit_event(event_type: str, data: Dict[str, Any]) -> None:
    """Emit a streaming event as NDJSON.

    Args:
        event_type: Event type (tool_start, tool_end, chunk, done).
        data: Event data.
    """
    event = {"event": event_type, **data}
    # Print to stdout, flush immediately for real-time streaming
    print(json.dumps(event), flush=True)


def _load_context_file(context_file: str) -> List[Dict[str, str]]:
    """Load conversation history from a JSON file.

    Args:
        context_file: Path to JSON file containing conversation history.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    try:
        path = Path(context_file)
        if not path.exists():
            logger.warning(f"[ASK_CLI] Context file not found: {context_file}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            context = json.load(f)

        if not isinstance(context, list):
            logger.warning("[ASK_CLI] Context file must contain a JSON array")
            return []

        logger.info(f"[ASK_CLI] Loaded {len(context)} messages from context file")
        return context
    except Exception as e:
        logger.warning(f"[ASK_CLI] Failed to load context file: {e}")
        return []


def _create_agent(config: Optional[dict], timeout: float, skip_tests: bool):
    """Create and return an AskAgent instance.

    Args:
        config: Optional config dict.
        timeout: Timeout in seconds.
        skip_tests: Whether to skip test matching phase.

    Returns:
        Tuple of (AskAgent, on_progress callback creator)
    """
    from dokumen.ask_agent import AskAgent
    from dokumen.loader import get_configured_provider
    from dokumen.tools_object import (
        BUILTIN_TOOLS,
        create_bash_tool,
        create_grep_tool,
        create_create_test_tool,
        create_chat_write_file_tool,
        create_chat_delete_file_tool,
        create_re_explore_tool,
    )

    # Get provider (uses config from dokumen.yaml or environment)
    provider = get_configured_provider()
    logger.info(f"[ASK_CLI] Provider initialized: {type(provider).__name__}")

    # Create tools (full executor toolset)
    base_dir = "."
    tools = []
    for name, factory in BUILTIN_TOOLS.items():
        tools.append(factory(base_dir))
        logger.debug(f"[ASK_CLI] Added tool: {name}")

    # Add bash and grep tools (sandbox=None runs without sandboxing)
    tools.append(create_bash_tool(sandbox=None, timeout=30.0, base_dir=base_dir))
    tools.append(create_grep_tool(sandbox=None))

    # Add create_test tool for generating test scaffolds
    tools.append(create_create_test_tool(base_dir=base_dir))
    logger.debug("[ASK_CLI] Added tool: create_test")

    # Add write_file tool for writing files to local clone (batch committed at end of turn)
    tools.append(create_chat_write_file_tool(base_dir=base_dir))
    logger.debug("[ASK_CLI] Added tool: write_file")

    # Add delete_file tool for deleting files from local clone (batch committed at end of turn)
    tools.append(create_chat_delete_file_tool(base_dir=base_dir))
    logger.debug("[ASK_CLI] Added tool: delete_file")

    # Add re_explore tool for re-exploring the codebase with a new topic
    tools.append(create_re_explore_tool())
    logger.debug("[ASK_CLI] Added tool: re_explore")

    logger.info(f"[ASK_CLI] Total tools available: {len(tools)}")

    # Determine tests directory
    tests_dir = "tests"
    if config:
        ask_config = config.get("ask", {})
        tests_dir = ask_config.get("tests_dir", "tests")

    # Create ask agent
    agent = AskAgent(
        provider=provider,
        base_dir=base_dir,
        timeout=timeout,
        tools=tools,
        tests_dir=tests_dir if not skip_tests else "__skip__",
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
        if event_type == "tool_start":
            _emit_event("tool_start", {
                "tool": data.get("tool", ""),
                "params": data.get("params", {}),
            })
        elif event_type == "tool_end":
            tool_name = data.get("tool", "")
            result = data.get("result", "")
            # Pass full result for create_test (backend needs it for auto-save)
            # Truncate other tools to avoid huge outputs in streaming
            if tool_name == "create_test":
                _emit_event("tool_end", {
                    "tool": tool_name,
                    "result": result,
                })
            else:
                _emit_event("tool_end", {
                    "tool": tool_name,
                    "result": result[:500] if len(result) > 500 else result,
                })
        elif event_type == "chunk":
            _emit_event("chunk", {"content": data.get("content", "")})
        elif event_type == "explore_start":
            logger.info("[ASK_CLI] Emitting explore_start event")
            _emit_event("explore_start", {"question": data.get("question", "")})
        elif event_type == "explore_end":
            summary = data.get("summary")
            logger.info(f"[ASK_CLI] Emitting explore_end event, summary={repr(summary)[:100]}")
            _emit_event("explore_end", {
                "files": data.get("files", 0),
                "tool_history": data.get("tool_history", []),
                "summary": summary or "",
            })
        elif event_type == "explore_tests_start":
            _emit_event("explore_tests_start", {})
        elif event_type == "explore_tests_end":
            _emit_event("explore_tests_end", {"matched": data.get("matched", 0)})

    return on_progress


async def _run_ask(
    question: str,
    timeout: float = 120.0,
    skip_tests: bool = False,
    config: Optional[dict] = None,
    stream: bool = False,
    context_file: Optional[str] = None,
) -> Any:
    """Run the ask agent to answer a question.

    Args:
        question: The user's question.
        timeout: Timeout in seconds.
        skip_tests: Whether to skip test matching phase.
        config: Optional config dict.
        stream: If True, emit NDJSON events to stdout.
        context_file: Optional path to JSON file with conversation history.

    Returns:
        AskResult with the answer.
    """
    logger.info(f"[ASK_CLI] Starting ask for question: {question[:100]!r}")
    logger.info(f"[ASK_CLI] Parameters: timeout={timeout}, skip_tests={skip_tests}, stream={stream}")

    start_time = time.time()

    # Load conversation history if provided
    conversation_history = []
    if context_file:
        conversation_history = _load_context_file(context_file)
        if stream:
            _emit_event("context_loaded", {"messages": len(conversation_history)})

    # Create agent and progress callback
    agent = _create_agent(config, timeout, skip_tests)
    on_progress = _create_progress_callback(stream)

    # Run ask with progress callback
    logger.info("[ASK_CLI] Running ask agent...")
    result = await agent.ask(
        question,
        on_progress=on_progress,
        conversation_history=conversation_history if conversation_history else None,
    )

    elapsed = time.time() - start_time
    logger.info(f"[ASK_CLI] Ask completed in {elapsed:.2f}s")
    logger.info(f"[ASK_CLI] Result: success={result.success}")

    return result


async def _run_interactive_session(
    timeout: float,
    skip_tests: bool,
    config: Optional[dict],
    stream: bool,
) -> None:
    """Run an interactive ask session.

    Explores documentation once at the start (using the first question as topic),
    then accepts follow-up questions in a REPL loop until the user exits.

    Args:
        timeout: Timeout in seconds per question.
        skip_tests: Whether to skip test matching phase.
        config: Optional config dict.
        stream: If True, emit NDJSON events to stdout.
    """
    # Create agent
    agent = _create_agent(config, timeout, skip_tests)
    on_progress = _create_progress_callback(stream)

    # Session will be initialized with the first question as the topic
    session_initialized = False

    if not stream:
        click.echo(click.style("Ready to explore documentation based on your first question.", fg="cyan"))
        click.echo(click.style("Type 'exit' or 'quit' to end the session.\n", dim=True))

    # Interactive loop
    while True:
        try:
            if stream:
                # In stream mode, emit ready event
                _emit_event("ready", {})

            # Get user input
            if stream:
                # Read from stdin
                line = sys.stdin.readline()
                if not line:
                    # EOF
                    break
                line = line.strip()
                if not line:
                    continue

                # Parse as JSON or plain text
                try:
                    msg = json.loads(line)
                    msg_type = msg.get("type", "question")

                    if msg_type == "quit":
                        break
                    elif msg_type == "re_explore":
                        # Handle re-explore request
                        topic = msg.get("topic", "")
                        logger.info(f"[ASK_CLI] Re-explore requested with topic: {topic[:100]!r}")

                        # Reset session and run fresh explore
                        agent.reset_session()
                        explore_result = await agent.initialize_session(
                            topic=topic,
                            on_progress=on_progress,
                        )
                        session_initialized = True

                        logger.info(f"[ASK_CLI] Re-explore complete: {len(explore_result.files)} files found")
                        # Continue waiting for next question
                        continue

                    question = msg.get("content", "")
                except json.JSONDecodeError:
                    question = line
            else:
                # Interactive prompt
                try:
                    question = click.prompt(click.style(">", fg="cyan"), prompt_suffix=" ")
                except click.Abort:
                    break

            # Check for exit commands
            if question.lower() in ("exit", "quit", "q"):
                break

            if not question.strip():
                continue

            # Initialize session with first question's topic (runs explore once)
            if not session_initialized:
                if not stream:
                    click.echo(click.style(f"Exploring documentation for: {question[:50]}...", fg="cyan"))
                explore_result = await agent.initialize_session(topic=question, on_progress=on_progress)
                session_initialized = True
                if not stream:
                    click.echo(click.style(f"Session ready. Found {len(explore_result.files)} relevant files.", fg="green"))

            # Run ask (session mode - reuses explore result)
            result = await agent.ask(question, on_progress=on_progress)

            # Output result
            if stream:
                _emit_event("done", {
                    "success": result.success,
                    "answer": result.answer,
                    "sources": result.sources,
                    "matched_tests": [t.to_dict() for t in result.matched_tests] if result.matched_tests else [],
                    "duration": result.duration,
                    "tool_calls_count": result.tool_calls_count,
                    "error": result.error,
                })
            else:
                click.echo("")
                if result.success:
                    click.echo(result.answer)
                else:
                    click.echo(click.style(f"Error: {result.error}", fg="red"))

                if result.sources:
                    click.echo("")
                    click.echo(click.style("Sources:", dim=True))
                    for source in result.sources:
                        click.echo(click.style(f"  - {source}", fg="cyan", dim=True))
                click.echo("")

        except KeyboardInterrupt:
            if not stream:
                click.echo("\n")
            break
        except Exception as e:
            if stream:
                _emit_event("error", {"message": str(e)})
            else:
                click.echo(click.style(f"Error: {e}", fg="red"))

    if stream:
        _emit_event("session_end", {})
    else:
        click.echo(click.style("Session ended.", dim=True))


async def _run_stdin_session(
    timeout: float,
    skip_tests: bool,
    config: Optional[dict],
) -> None:
    """Run a stdin-based ask session for backend integration.

    Reads NDJSON messages from stdin, outputs NDJSON events to stdout.
    Explores documentation once at the start.

    Protocol:
        Input: {"type": "question", "content": "..."} or {"type": "quit"}
        Output: Standard NDJSON events (explore_start, explore_end, done, etc.)

    Args:
        timeout: Timeout in seconds per question.
        skip_tests: Whether to skip test matching phase.
        config: Optional config dict.
    """
    await _run_interactive_session(
        timeout=timeout,
        skip_tests=skip_tests,
        config=config,
        stream=True,
    )


def _format_text_output(result) -> str:
    """Format AskResult as human-readable text.

    Args:
        result: AskResult object.

    Returns:
        Formatted text string.
    """
    lines = []

    if result.success:
        lines.append(click.style("Answer", fg="green", bold=True))
        lines.append("")
        lines.append(result.answer)
    else:
        lines.append(click.style("Failed", fg="red", bold=True))
        if result.error:
            lines.append(f"Error: {result.error}")

    lines.append("")

    # Sources
    if result.sources:
        lines.append(click.style("Sources:", bold=True))
        for source in result.sources:
            lines.append(f"  - {click.style(source, fg='cyan')}")
        lines.append("")

    # Matched tests
    if result.matched_tests:
        lines.append(click.style(f"Matched Tests ({len(result.matched_tests)}):", bold=True))
        for test in result.matched_tests:
            lines.append(f"  - {test.test_name} ({test.relevance_score:.0%})")
            if test.reason:
                lines.append(f"    {test.reason}")
        lines.append("")

    # Stats
    lines.append(click.style("Stats:", bold=True))
    lines.append(f"  Duration: {result.duration:.2f}s")
    lines.append(f"  Tool calls: {result.tool_calls_count}")

    return "\n".join(lines)


@click.command()
@click.argument("question", required=False, default=None)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format (text or json)",
)
@click.option(
    "--timeout",
    "-t",
    type=float,
    default=120.0,
    help="Timeout in seconds",
)
@click.option(
    "--skip-tests",
    is_flag=True,
    help="Skip test matching phase",
)
@click.option(
    "--stream",
    is_flag=True,
    help="Stream events as NDJSON (newline-delimited JSON)",
)
@click.option(
    "--stdin",
    "stdin_mode",
    is_flag=True,
    help="Read questions from stdin (NDJSON format, for backend integration)",
)
@click.option(
    "--context-file",
    type=click.Path(exists=False),
    help="Path to JSON file with conversation history (single question mode only)",
)
@click.pass_context
def ask(ctx, question: Optional[str], output: str, timeout: float, skip_tests: bool, stream: bool, stdin_mode: bool, context_file: str):
    """Ask questions about documentation.

    Supports three modes:

    \b
    1. SINGLE QUESTION MODE (default with argument):
       dokumen ask "What is the margin requirement?"

    \b
    2. INTERACTIVE MODE (no argument):
       dokumen ask
       > What is the margin requirement?
       > How about for futures?
       > exit

    \b
    3. STDIN MODE (for backend integration):
       dokumen ask --stdin --stream
       {"type": "question", "content": "What is X?"}
       {"type": "quit"}

    In interactive and stdin modes, documentation is explored ONCE at the
    start and reused for all follow-up questions. Conversation context is
    maintained throughout the session.

    Examples:

    \b
        # Single question
        dokumen ask "What is the margin requirement?"

        # Interactive session
        dokumen ask

        # Backend integration (stdin mode)
        dokumen ask --stdin --stream
    """
    logger.info(f"[ASK_CMD] Command invoked: question={question[:50] if question else None!r}, output={output}, stream={stream}, stdin={stdin_mode}")

    # Load config if available
    config = None
    try:
        config_path = ctx.obj.get("config_path") if ctx.obj else None
        config = load_config(config_path)
        logger.info("[ASK_CMD] Config loaded successfully")
    except Exception as e:
        logger.warning(f"[ASK_CMD] Failed to load config: {e}, continuing with defaults")

    # Determine mode
    if stdin_mode:
        # Stdin mode for backend integration
        logger.info("[ASK_CMD] Running in stdin mode")
        try:
            run_async(_run_stdin_session(
                timeout=timeout,
                skip_tests=skip_tests,
                config=config,
            ))
        except Exception as e:
            logger.error(f"[ASK_CMD] Stdin session failed: {e}", exc_info=True)
            _emit_event("error", {"message": str(e)})
            sys.exit(1)
        return

    if question is None:
        # Interactive mode
        logger.info("[ASK_CMD] Running in interactive mode")
        try:
            run_async(_run_interactive_session(
                timeout=timeout,
                skip_tests=skip_tests,
                config=config,
                stream=stream,
            ))
        except Exception as e:
            logger.error(f"[ASK_CMD] Interactive session failed: {e}", exc_info=True)
            if stream:
                _emit_event("error", {"message": str(e)})
            else:
                click.echo(click.style(f"Error: {e}", fg="red"), err=True)
            sys.exit(1)
        return

    # Single question mode (original behavior)
    try:
        result = run_async(
            _run_ask(
                question=question,
                timeout=timeout,
                skip_tests=skip_tests,
                config=config,
                stream=stream,
                context_file=context_file,
            )
        )
    except Exception as e:
        logger.error(f"[ASK_CMD] Ask failed with error: {e}", exc_info=True)
        if stream:
            # Emit error event for streaming mode
            _emit_event("done", {
                "success": False,
                "error": str(e),
                "sources": [],
            })
        elif output == "json":
            error_output = {
                "success": False,
                "error": str(e),
                "answer": "",
                "sources": [],
                "matched_tests": [],
                "explore_summary": None,
                "duration": 0,
                "tool_calls_count": 0,
            }
            click.echo(json.dumps(error_output, indent=2))
        else:
            click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)

    # Handle streaming output
    if stream:
        # In streaming mode, emit done event with final result
        _emit_event("done", {
            "success": result.success,
            "answer": result.answer,
            "sources": result.sources,
            "matched_tests": [t.to_dict() for t in result.matched_tests] if result.matched_tests else [],
            "duration": result.duration,
            "tool_calls_count": result.tool_calls_count,
            "error": result.error,
        })
        if not result.success:
            sys.exit(1)
        logger.info("[ASK_CMD] Command completed (streaming)")
        return

    # Handle failure (non-streaming)
    if not result.success:
        if output == "json":
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            click.echo(_format_text_output(result))
        sys.exit(1)

    # Output result (non-streaming)
    if output == "json":
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(_format_text_output(result))

    logger.info("[ASK_CMD] Command completed")
