"""
Explore Agent module for the Agent SOP Testing Framework.

Provides an exploration phase that runs before the main executor to discover
relevant documentation files and provide context.

Uses the Claude Agent SDK (via ``SDKQueryRunner``) for LLM interaction,
with read-only SDK tools (Read, Glob, Grep).

This module keeps the CLI-specific SDK integration, system prompts, local
result types, and debug logging.
"""

from typing import Any, Callable, Dict, List, Optional
import asyncio
import os
import time
import traceback

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
)

from .explore_types import (
    ExploreResult,
    ExploreToolRecord,
    FileDiscovery,
    VALID_EXPLORE_TYPES,
    extract_paths_from_response,
)

from .debug import is_debug, debug
from .logging_config import get_logger
from .sdk.query_runner import QueryRunner, SDKQueryRunner

# Module-level logger
logger = get_logger(__name__)

# SDK tool names for explore (read-only)
EXPLORE_SDK_TOOLS = ["Read", "Glob", "Grep"]


EXPLORE_SYSTEM_PROMPT = """You are a documentation explorer. Your job is to find files relevant to the user's request.

## Process
1. FIRST: Check if DOKUMEN_SUMMARIES_INDEX.md exists in the project root
   - If it exists, read it to get an overview of all documentation files and their summaries
   - Use the summaries to identify which files are most relevant to the request
   - Read only those specific files for deeper understanding
2. If no index exists, use available tools to search for relevant files
3. Examine file contents to assess relevance
4. Return a natural language summary of what you found

## Output Format
Respond with a natural language description of relevant files:
- List the MOST IMPORTANT files first
- Include less important files after
- Omit irrelevant files entirely
- For each file, briefly explain why it's relevant

Keep your response concise (under 500 words).
"""

EXPLORE_CODE_SYSTEM_PROMPT = """You are a code explorer. Your job is to find source code files relevant to the user's request.

## Process
1. Use the available SDK tools (Read, Glob, Grep) to search the local workspace
2. Look for implementation files, classes, functions, and modules related to the request
3. Examine source code to understand the implementation approach
4. Return a natural language summary of what you found

## Focus Areas
- Find the main implementation files for the requested feature or component
- Identify key classes, functions, and modules
- Note any configuration files or constants relevant to the request
- Look for test files that verify the implementation

## Output Format
Respond with a natural language description of relevant code files:
- List the MOST IMPORTANT implementation files first
- Include supporting files (tests, configs) after
- Omit irrelevant files entirely
- For each file, briefly explain what it implements and why it's relevant

Keep your response concise (under 500 words).
"""

EXPLORE_BOTH_SYSTEM_PROMPT = """You are a documentation and code explorer. Your job is to find both documentation files and source code files relevant to the user's request.

## Process
1. FIRST: Check if DOKUMEN_SUMMARIES_INDEX.md exists in the project root
   - If it exists, read it to get an overview of all documentation files and their summaries
   - Use the summaries to identify which documentation files are most relevant
2. Use the available SDK tools (Read, Glob, Grep) to explore documentation
3. Use the available SDK tools (Read, Glob, Grep) to search local source code
4. Cross-reference documentation with implementation to find discrepancies or related context
5. Return a natural language summary of what you found

## Focus Areas
- Find documentation that describes the feature or topic
- Find the source code that implements it
- Note any gaps between documentation and implementation
- Identify test files that verify the implementation

## Output Format
Respond with a natural language description of relevant files:
- Group documentation files and code files separately
- List the MOST IMPORTANT files first in each group
- For each file, briefly explain why it's relevant
- Note how the relevant documentation and implementation files relate

Keep your response concise (under 500 words).
"""

# Re-export explore types for backward compatibility.
# Existing code does ``from dokumen.explore_agent import FileDiscovery, ExploreResult``.
# The names are already imported at the top from ``dokumen.explore_types``.
__all__ = [
    "FileDiscovery",
    "ExploreResult",
    "ExploreToolRecord",
    "VALID_EXPLORE_TYPES",
    "ExploreAgent",
]


class ExploreAgent:
    """Agent that explores the workspace before main task execution.

    Uses the Claude Agent SDK (via SDKQueryRunner) to discover relevant files
    and provide context to the main executor agent.
    """

    def __init__(
        self,
        query_runner: Optional[QueryRunner] = None,
        base_dir: str = ".",
        max_files: int = 20,
        max_turns: int = 50,
        timeout: float = 60.0,
        explore_type: str = "docs",
        model: Optional[str] = None,
    ):
        """Initialize explore agent.

        Args:
            query_runner: SDK query runner (defaults to SDKQueryRunner).
                         Use MockQueryRunner for tests.
            base_dir: Base directory for file operations.
            max_files: Maximum number of files to return.
            max_turns: Maximum conversation turns before stopping.
            timeout: Timeout in seconds.
            explore_type: Type of exploration - "docs" (default), "code", or "both".
            model: Optional model override for the SDK query.
        """
        # Validate explore_type
        if explore_type not in VALID_EXPLORE_TYPES:
            raise ValueError(
                f"explore_type must be one of {VALID_EXPLORE_TYPES}, got '{explore_type}'"
            )

        self._runner = query_runner or SDKQueryRunner()
        self.base_dir = base_dir
        self.max_files = max_files
        self.max_turns = max_turns
        self.timeout = timeout
        self.explore_type = explore_type
        self.model = model
        # Tracks the latest assistant text for partial result extraction on timeout
        self._last_assistant_text: str = ""

        logger.debug(
            "explore.init",
            base_dir=base_dir,
            max_files=max_files,
            max_turns=max_turns,
            timeout=timeout,
            explore_type=explore_type,
            model=model,
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt based on explore_type.

        Returns:
            System prompt string appropriate for the exploration type.
        """
        if self.explore_type == "code":
            return EXPLORE_CODE_SYSTEM_PROMPT
        elif self.explore_type == "both":
            return EXPLORE_BOTH_SYSTEM_PROMPT
        else:
            return EXPLORE_SYSTEM_PROMPT

    async def explore(
        self, goal: str, on_progress: Optional[Callable[[str, Dict], None]] = None
    ) -> ExploreResult:
        """Run exploration to discover relevant files.

        Args:
            goal: What to explore/discover (derived from user task)
            on_progress: Optional callback for progress events
                        Signature: (event_type: str, data: dict) -> None
                        Events: 'start', 'file_found', 'complete'

        Returns:
            ExploreResult with discovered files and metadata
        """
        start_time = time.time()
        self._last_assistant_text = ""  # Reset for this run

        # Fail fast with a clear error if no query runner is configured
        if self._runner is None:
            logger.error(
                "explore.no_runner",
                error="No query runner configured for explore phase",
            )
            return ExploreResult(
                files=[],
                duration=0.0,
                tool_calls_count=0,
                success=False,
                error="No LLM provider configured. Set ANTHROPIC_API_KEY environment variable or configure a provider in dokumen.yaml.",
                summary="Exploration failed: no LLM provider configured",
                model=None,
            )

        logger.info("explore.start", goal=goal[:100])
        if is_debug():
            debug(f"[EXPLORE] Starting exploration for: {goal[:100]}...")

        # Emit start event
        if on_progress:
            on_progress("start", {"goal": goal})

        try:
            # Build SDK options with read-only tools
            system_prompt = self._get_system_prompt()

            # Determine permission mode based on whether running as root
            if os.getuid() == 0:
                perm_mode = "acceptEdits"
            else:
                perm_mode = "bypassPermissions"

            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                allowed_tools=list(EXPLORE_SDK_TOOLS),
                permission_mode=perm_mode,
                max_turns=self.max_turns,
                model=self.model,
            )

            user_prompt = f"Find files relevant to this request:\n\n{goal}"

            # Run SDK query with timeout
            result = await asyncio.wait_for(
                self._run_sdk_query(user_prompt, options),
                timeout=self.timeout,
            )

            # Check for SDK-level errors
            if result.get("is_error"):
                duration = time.time() - start_time
                error_msg = result.get("final_text", "SDK query returned an error")
                logger.warning(
                    "explore.sdk_error",
                    error=error_msg[:200],
                    num_turns=result.get("num_turns", 0),
                )

                # Still try to extract any partial files from the response
                partial_files, partial_summary = self._parse_explore_response(
                    result.get("final_text", "")
                )
                partial_files = partial_files[: self.max_files]

                return ExploreResult(
                    files=partial_files,
                    duration=duration,
                    tool_calls_count=result.get("num_turns", 0),
                    success=False,
                    error=f"SDK query error: {error_msg[:200]}",
                    summary=partial_summary or "Exploration encountered an SDK error",
                    tool_history=[],
                    model=self.model,
                )

            # Parse the final response text to extract files
            final_text = result.get("final_text", "")
            files, summary = self._parse_explore_response(final_text)

            # Limit files to max_files
            files = files[: self.max_files]

            duration = time.time() - start_time
            num_turns = result.get("num_turns", 0)

            # Emit complete event
            if on_progress:
                on_progress("complete", {"files_found": len(files), "duration": duration})

            logger.info(
                "explore.complete",
                files_found=len(files),
                duration_ms=int(duration * 1000),
                num_turns=num_turns,
            )
            if is_debug():
                debug(
                    f"[EXPLORE] Completed: {len(files)} files found in {duration:.2f}s ({num_turns} turns)"
                )
                for f in files:
                    debug(f"[EXPLORE]   - {f.path}: {f.summary[:50]}...")

            logger.info(
                "explore.result",
                summary_length=len(summary) if summary else 0,
                files_count=len(files),
            )

            return ExploreResult(
                files=files,
                duration=duration,
                tool_calls_count=num_turns,
                success=True,
                summary=summary,
                tool_history=[],
                model=self.model,
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time

            # Extract any partial results discovered before the timeout
            partial_files: List[FileDiscovery] = []
            partial_summary = ""
            if self._last_assistant_text:
                partial_files, partial_summary = self._parse_explore_response(
                    self._last_assistant_text
                )
                partial_files = partial_files[: self.max_files]

            logger.warning(
                "explore.timeout",
                timeout=self.timeout,
                partial_files_recovered=len(partial_files),
            )

            error_msg = f"Explore timed out after {duration:.1f}s"

            return ExploreResult(
                files=partial_files,
                duration=duration,
                tool_calls_count=0,
                success=False,
                error=error_msg,
                tool_history=[],
                summary=partial_summary or f"Exploration timed out after {duration:.1f}s",
                model=self.model,
            )
        except Exception as e:
            duration = time.time() - start_time

            # Extract any partial results discovered before the error
            partial_files: List[FileDiscovery] = []
            partial_summary = ""
            if self._last_assistant_text:
                partial_files, partial_summary = self._parse_explore_response(
                    self._last_assistant_text
                )
                partial_files = partial_files[: self.max_files]

            logger.error(
                "explore.error",
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc(),
                partial_files_recovered=len(partial_files),
            )

            return ExploreResult(
                files=partial_files,
                duration=duration,
                tool_calls_count=0,
                success=False,
                error=str(e),
                tool_history=[],
                summary=partial_summary or f"Exploration failed: {str(e)[:100]}",
                model=self.model,
            )

    async def _run_sdk_query(
        self,
        prompt: str,
        options: ClaudeAgentOptions,
    ) -> Dict[str, Any]:
        """Run an SDK query and collect the result.

        Args:
            prompt: User prompt to send.
            options: SDK options for the query.

        Returns:
            Dict with 'final_text', 'num_turns', 'duration_ms', 'is_error'.
        """
        final_text = ""
        num_turns = 0
        duration_ms = 0
        is_error = False

        logger.info(
            "explore.sdk_query.start",
            prompt_length=len(prompt),
            max_turns=options.max_turns,
            model=options.model,
        )

        async for msg in self._runner.run(prompt, options):
            if isinstance(msg, AssistantMessage):
                # Capture the last assistant message text as final response
                # msg.content is list[ContentBlock]; extract text from TextBlocks
                if hasattr(msg, "content") and msg.content:
                    # Extract text from ContentBlock list and store for timeout handler
                    self._last_assistant_text = "\n".join(
                        block.text for block in msg.content if hasattr(block, "text")
                    )
                    final_text = self._last_assistant_text
            elif isinstance(msg, ResultMessage):
                if hasattr(msg, "num_turns"):
                    num_turns = msg.num_turns
                if hasattr(msg, "duration_ms"):
                    duration_ms = msg.duration_ms
                if hasattr(msg, "is_error"):
                    is_error = msg.is_error

        logger.info(
            "explore.sdk_query.complete",
            final_text_length=len(final_text),
            num_turns=num_turns,
            duration_ms=duration_ms,
            is_error=is_error,
        )

        return {
            "final_text": final_text,
            "num_turns": num_turns,
            "duration_ms": duration_ms,
            "is_error": is_error,
        }

    def _parse_explore_response(self, content: str) -> tuple[List[FileDiscovery], str]:
        """Parse the explore response - natural language with file paths.

        Delegates to the local natural-language path extractor.

        Args:
            content: Response content (natural language)

        Returns:
            Tuple of (list of FileDiscovery objects, summary string)
        """
        if isinstance(content, list):
            normalized_content = "\n".join(
                part if isinstance(part, str) else str(part) for part in content
            )
        else:
            normalized_content = content

        logger.info(
            "explore.parse_response",
            content_length=len(normalized_content),
        )

        if not normalized_content:
            return [], ""

        files, summary = extract_paths_from_response(normalized_content)

        logger.info(
            "explore.parse_complete",
            files_found=len(files),
            summary_length=len(summary) if summary else 0,
        )

        if is_debug():
            debug(f"[EXPLORE] Parsed {len(files)} file paths from response")
            for f in files:
                debug(f"[EXPLORE]   extracted: {f.path}")

        return files, summary
