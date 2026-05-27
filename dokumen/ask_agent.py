"""
Ask agent for dokumen CLI.

Provides a documentation assistant that can answer questions about
documentation using explore-based test discovery.
"""
import logging
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import yaml

from .agent_object import Provider
from .explore_agent import ExploreAgent, ExploreResult
from .tools_object import ToolDefinition

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class MatchedTest:
    """A test scaffold that matches the user's question."""

    test_id: str
    test_name: str
    reason: str
    relevance_score: float
    success_criteria: str  # Combined judge system prompts
    files_covered: List[str]
    user_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "reason": self.reason,
            "relevance_score": self.relevance_score,
            "files_covered": self.files_covered,
        }


@dataclass
class AskResult:
    """Result from the AskAgent.ask() method."""

    success: bool
    answer: str
    sources: List[str]
    matched_tests: List[MatchedTest]
    explore_summary: Optional[str]
    duration: float
    tool_calls_count: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "answer": self.answer,
            "sources": self.sources,
            "matched_tests": [
                {
                    "test_id": m.test_id,
                    "test_name": m.test_name,
                    "reason": m.reason,
                    "relevance_score": m.relevance_score,
                    "files_covered": m.files_covered,
                }
                for m in self.matched_tests
            ],
            "explore_summary": self.explore_summary,
            "duration": self.duration,
            "tool_calls_count": self.tool_calls_count,
            "error": self.error,
        }


# =============================================================================
# System Prompts
# =============================================================================


UNIFIED_SYSTEM_PROMPT = """You are a documentation assistant for the Dokumen project.

## Context Provided

You have been given:
1. **Pre-discovered documentation files** - Relevant files found by the explore phase
2. **Matching test criteria** - Success criteria from tests that cover similar topics
3. **Tool access** - You can read files, search, run commands to gather more info

## Guidelines

1. **Ground ALL claims in documentation** - Only state what the docs explicitly say
2. **Cite sources** - Reference specific file paths and sections
3. **Meet test criteria** - If a matching test exists, ensure your answer would pass its judges
4. **Acknowledge gaps** - If docs don't cover something, say so clearly
5. **Be actionable** - Provide concrete steps when possible

## CRITICAL: Creating Tests

When asked to create a test, you MUST use the `create_test` tool. Do NOT generate YAML manually.

**How to create tests:**
1. Call the `create_test` tool with a clear goal describing what to validate
2. For browser/UI tests, include `type: "browser"` in the tool call
3. The tool will automatically discover relevant files and generate a scaffold
4. Show the generated scaffold to the user for approval
5. If approved, use the `write_file` tool to save the test file to the suggested path

**Test Types:**
- **Standard** (default): Validates documentation content using file tools
- **Browser**: Tests web UI behavior using browser automation tools (browser_navigate, browser_click, etc.)

Use `type: "browser"` when the user mentions: URLs, web pages, UI testing, login flows, clicking, navigating, screenshots, or browser automation.

**Single Judge Rule:** Tests have exactly ONE judge by default. Keep tests simple and focused.
Users can request additional judges in follow-up messages.

## Saving Files

Use the `write_file` tool to save any files (tests, documentation updates, etc.):
- Files are committed to the user's branch in GitLab
- Always show content to user before saving and ask for approval
- Use the suggested path from `create_test` for test files

## Re-Exploration

You have access to a `re_explore` tool that lets you search the codebase again with a different focus. Use it when:
- The user says you're looking at the wrong files
- You realize your initial exploration missed relevant documentation
- The user asks about a topic significantly different from your initial exploration
- You see phrases like "not what I meant", "wrong section", "different file"

Example scenarios:
- User: "No, I meant the API authentication, not the user login docs"
  Call re_explore with topic "API authentication endpoints"

- User: "You're looking at the old refund policy, I need the new one"
  Call re_explore with topic "new refund policy documentation"

The re_explore tool will search the codebase with the new topic and provide fresh context for your answers.

## Final Answer Requirement

**You must ALWAYS provide a natural language text response at the end of your turn.**
Even when you have used tools (read_file, search, write_file, etc.), you must always
conclude with a text summary that explains what you found, what you did, or what the
results mean. Never end your turn with only tool calls and no text.

## Output Format

[Your answer here]

**Sources:**
- `path/to/file.md`: Section referenced

**Confidence:** High | Medium | Low
"""

# Keep for backward compatibility in tests
ASK_SYSTEM_PROMPT = UNIFIED_SYSTEM_PROMPT


# =============================================================================
# AskAgent Class
# =============================================================================


class AskAgent:
    """Agent that answers questions about documentation.

    Supports session mode where explore runs once and is reused for follow-ups.
    """

    def __init__(
        self,
        provider: Provider,
        base_dir: str = ".",
        timeout: float = 120.0,
        tools: Optional[List[ToolDefinition]] = None,
        tests_dir: str = "tests",
    ):
        """Initialize the AskAgent.

        Args:
            provider: LLM provider for generating responses.
            base_dir: Base directory for file operations.
            timeout: Maximum time for the ask operation.
            tools: List of tools available to the agent.
            tests_dir: Directory containing test scaffolds.
        """
        self.provider = provider
        self.base_dir = base_dir
        self.timeout = timeout
        self.tools = tools or []
        self.tests_dir = tests_dir
        self._tool_calls_count = 0
        self._tool_history: List[Dict] = []
        # Session state - persists across multiple ask() calls
        self._explore_result: Optional[ExploreResult] = None
        self._matched_tests: Optional[List[MatchedTest]] = None
        self._conversation_history: List[Dict[str, str]] = []
        self._session_initialized: bool = False

    async def initialize_session(
        self,
        topic: Optional[str] = None,
        on_progress: Optional[Callable] = None,
    ) -> ExploreResult:
        """Initialize a session by running explore once.

        After initialization, subsequent ask() calls will reuse the explore
        result and maintain conversation history automatically.

        Args:
            topic: Optional topic to guide exploration. If not provided,
                   explores the general documentation structure.
            on_progress: Optional callback for progress updates.

        Returns:
            ExploreResult from the initial exploration.
        """
        logger.info(f"[ASK] Initializing session, topic={topic[:50] if topic else 'general'}...")

        # Run initial explore
        explore_topic = topic or "documentation structure and available files"
        if on_progress:
            on_progress("explore_start", {"question": explore_topic[:100]})

        self._explore_result = await self._run_explore(explore_topic, on_progress)

        if on_progress:
            on_progress("explore_end", {
                "files": len(self._explore_result.files),
                "tool_history": self._explore_result.tool_history,
                "summary": self._explore_result.summary,
            })

        # Run initial test exploration
        if on_progress:
            on_progress("explore_tests_start", {})
        self._matched_tests = await self._explore_tests(explore_topic)
        if on_progress:
            on_progress("explore_tests_end", {"matched": len(self._matched_tests)})

        # Mark session as initialized
        self._session_initialized = True
        self._conversation_history = []

        logger.info(
            f"[ASK] Session initialized: {len(self._explore_result.files)} files, "
            f"{len(self._matched_tests)} tests"
        )

        return self._explore_result

    def reset_session(self) -> None:
        """Reset the session state.

        Clears explore result, matched tests, and conversation history.
        The next ask() call will run a fresh explore.
        """
        logger.info("[ASK] Resetting session")
        self._explore_result = None
        self._matched_tests = None
        self._conversation_history = []
        self._session_initialized = False

    @property
    def is_session_initialized(self) -> bool:
        """Check if a session is currently initialized."""
        return self._session_initialized

    @property
    def conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self._conversation_history.copy()

    async def ask(
        self,
        question: str,
        explore_result: Optional[ExploreResult] = None,
        on_progress: Optional[Callable] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AskResult:
        """Ask a question about documentation or generate a skill.

        In session mode (when initialized via initialize_session()), explore
        runs once and is reused for all subsequent questions. Conversation
        history is automatically maintained.

        Args:
            question: The user's question or skill request.
            explore_result: Optional pre-computed explore result.
            on_progress: Optional callback for progress updates.
            conversation_history: Optional list of previous messages
                                  [{"role": "user", "content": "..."}, ...]

        Returns:
            AskResult with the answer or generated skill.
        """
        start_time = time.time()
        self._tool_calls_count = 0
        self._tool_history = []

        logger.info(f"[ASK] Starting ask for question: {question[:100]!r}")

        try:
            # Use session state if available, otherwise use provided/run new
            if self._session_initialized and self._explore_result is not None:
                # Session mode: reuse stored explore result
                explore_result = self._explore_result
                matched_tests = self._matched_tests or []
                logger.info(f"[ASK] Using session explore result ({len(explore_result.files)} files)")
            else:
                # Step 1: Run explore phase for docs if not provided
                if explore_result is None:
                    logger.info("[ASK] Step 1: Exploring documentation...")
                    if on_progress:
                        on_progress("explore_start", {"question": question[:100]})
                    explore_result = await self._run_explore(question, on_progress)
                    logger.info(f"[ASK] Explore found {len(explore_result.files)} files, summary={repr(explore_result.summary)[:100]}")
                    if on_progress:
                        on_progress("explore_end", {
                            "files": len(explore_result.files),
                            "tool_history": explore_result.tool_history,
                            "summary": explore_result.summary,
                        })

                # Step 2: Explore tests (using ExploreAgent instead of loading all)
                logger.info("[ASK] Step 2: Exploring tests...")
                if on_progress:
                    on_progress("explore_tests_start", {})
                matched_tests = await self._explore_tests(question)
                logger.info(f"[ASK] Found {len(matched_tests)} matched tests")
                if on_progress:
                    on_progress("explore_tests_end", {"matched": len(matched_tests)})
                # Also emit old event names for backward compatibility
                if on_progress:
                    on_progress("match_tests_start", {})
                    on_progress("match_tests_end", {"matched": len(matched_tests)})

            # Use session conversation history if in session mode
            effective_history = conversation_history
            if self._session_initialized:
                effective_history = self._conversation_history if self._conversation_history else None

            # Step 3: Generate response (unified - LLM decides answer vs skill)
            logger.info("[ASK] Step 3: Generating response...")
            if on_progress:
                on_progress("generate_response_start", {})
            result = await self._generate_response(
                question, explore_result, matched_tests, on_progress, effective_history
            )

            duration = time.time() - start_time
            result.duration = duration
            result.tool_calls_count = self._tool_calls_count
            result.explore_summary = explore_result.summary if explore_result else None
            result.matched_tests = matched_tests

            # Update session conversation history
            if self._session_initialized:
                self._conversation_history.append({"role": "user", "content": question})
                if result.answer:
                    self._conversation_history.append({"role": "assistant", "content": result.answer})
                else:
                    self._conversation_history.append({"role": "assistant", "content": "[Tool operations completed]"})

            logger.info(f"[ASK] Complete in {duration:.2f}s")
            return result

        except Exception as e:
            logger.error(
                "ask_agent.ask.error",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
            )
            logger.error(f"[ASK] Ask failed: {e}", exc_info=True)
            return AskResult(
                success=False,
                answer="",
                sources=[],
                matched_tests=[],
                explore_summary=None,
                duration=time.time() - start_time,
                tool_calls_count=self._tool_calls_count,
                error=str(e),
            )

    async def _run_explore(
        self, question: str, on_progress: Optional[Callable] = None
    ) -> ExploreResult:
        """Run the explore phase to find relevant documentation.

        Args:
            question: The question to explore documentation for.
            on_progress: Optional progress callback.

        Returns:
            ExploreResult with discovered files.
        """
        explore_agent = ExploreAgent(
            provider=self.provider,
            base_dir=self.base_dir,
            max_files=20,
            max_iterations=50,
            timeout=60.0,
            tools=self.tools,
        )

        return await explore_agent.explore(question, on_progress)

    async def _explore_tests(self, question: str) -> List[MatchedTest]:
        """Use ExploreAgent to find relevant tests.

        Instead of loading all test scaffolds and doing keyword matching,
        we use an ExploreAgent to discover relevant tests semantically.

        Args:
            question: The user's question.

        Returns:
            List of matched tests with success criteria.
        """
        # Skip if tests_dir is "__skip__" (for testing)
        if self.tests_dir == "__skip__":
            logger.debug("[ASK] Skipping test exploration (tests_dir='__skip__')")
            return []

        # Check if tests directory exists
        tests_path = Path(self.tests_dir)
        if not tests_path.exists():
            logger.debug(f"[ASK] Tests directory not found: {self.tests_dir}")
            return []

        logger.debug(f"[ASK] Exploring tests in: {self.tests_dir}")

        try:
            # Create explore agent for tests directory
            explore_agent = ExploreAgent(
                provider=self.provider,
                base_dir=self.tests_dir,
                max_files=10,
                max_iterations=50,
                timeout=30.0,
                tools=self.tools,
            )

            # Run exploration with a goal focused on finding relevant tests
            goal = f"Find test files (*.test.yaml) relevant to: {question}"
            explore_result = await explore_agent.explore(goal)

            # For each discovered test file, extract success criteria
            matched_tests = []
            for file_info in explore_result.files:
                file_path = file_info.path if hasattr(file_info, "path") else file_info.get("path", "")
                if file_path.endswith('.test.yaml'):
                    relevance = file_info.relevance if hasattr(file_info, "relevance") else file_info.get("relevance", 0.5)
                    test_data = await self._extract_test_criteria(file_path)
                    if test_data:
                        matched_tests.append(MatchedTest(
                            test_id=test_data['name'],
                            test_name=test_data['name'],
                            reason=test_data.get('reason', ''),
                            relevance_score=relevance,
                            success_criteria=test_data['success_criteria'],
                            files_covered=test_data['files_covered'],
                            user_prompt=test_data.get('user_prompt', ''),
                        ))
                        logger.debug(f"[ASK] Matched test: {test_data['name']} ({relevance:.0%})")

            # Return top 5 by relevance
            matched_tests.sort(key=lambda m: m.relevance_score, reverse=True)
            return matched_tests[:5]

        except Exception as e:
            logger.warning(f"[ASK] Test exploration failed: {e}")
            return []

    async def _extract_test_criteria(self, test_path: str) -> Optional[Dict]:
        """Extract success criteria from a test scaffold file.

        Args:
            test_path: Path to the test YAML file.

        Returns:
            Dictionary with test name, reason, success criteria, and files covered.
        """
        try:
            # Resolve path relative to tests_dir
            full_path = Path(self.tests_dir) / test_path
            if not full_path.exists():
                # Try as absolute path
                full_path = Path(test_path)
                if not full_path.exists():
                    logger.debug(f"[ASK] Test file not found: {test_path}")
                    return None

            logger.debug(f"[ASK] Extracting criteria from: {full_path}")

            # Read and parse YAML
            with open(full_path, 'r', encoding='utf-8') as f:
                scaffold = yaml.safe_load(f)

            if not scaffold:
                return None

            # Build success criteria from judge prompts
            criteria_parts = []
            for judge in scaffold.get('judges', []):
                if isinstance(judge, dict) and judge.get('system_prompt'):
                    criteria_parts.append(judge['system_prompt'])

            # Extract files covered
            files_covered = []
            for f in scaffold.get('files', []):
                if isinstance(f, dict) and f.get('path'):
                    files_covered.append(f['path'])
                elif isinstance(f, str):
                    files_covered.append(f)

            # Extract user prompt from executor
            user_prompt = ''
            executor = scaffold.get('executor', {})
            if isinstance(executor, dict):
                user_prompt = executor.get('user_prompt', '') or ''

            return {
                'name': scaffold.get('name', 'unknown'),
                'reason': scaffold.get('reason', ''),
                'success_criteria': '\n'.join(criteria_parts),
                'files_covered': files_covered,
                'user_prompt': user_prompt,
            }

        except Exception as e:
            logger.warning(f"[ASK] Failed to extract criteria from {test_path}: {e}")
            return None

    # Keep these for backward compatibility (tests may use them)
    def _load_scaffolds(self) -> List[Any]:
        """DEPRECATED: Load all test scaffolds. Use _explore_tests instead."""
        logger.warning("[ASK] _load_scaffolds is deprecated, use _explore_tests")
        return []

    def _match_tests(
        self,
        question: str,
        scaffolds: List[Any],
        explored_files: Set[str],
    ) -> List[MatchedTest]:
        """DEPRECATED: Match tests by keywords. Use _explore_tests instead."""
        logger.warning("[ASK] _match_tests is deprecated, use _explore_tests")
        return []

    def _calculate_test_relevance(self, **kwargs) -> float:
        """DEPRECATED: Calculate relevance score. Use _explore_tests instead."""
        logger.warning("[ASK] _calculate_test_relevance is deprecated")
        return 0.0

    def _extract_words(self, text: str) -> Set[str]:
        """Extract significant words from text.

        Args:
            text: Input text.

        Returns:
            Set of words longer than 3 characters.
        """
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return {w for w in words if len(w) > 3}

    async def _generate_response(
        self,
        question: str,
        explore_result: ExploreResult,
        matched_tests: List[MatchedTest],
        on_progress: Optional[Callable] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AskResult:
        """Generate a response to the question.

        Args:
            question: The user's question.
            explore_result: Results from explore phase.
            matched_tests: Matched test scaffolds.
            on_progress: Optional progress callback.
            conversation_history: Optional list of previous messages.

        Returns:
            AskResult with the answer.
        """
        logger.debug(f"[ASK] Generating response for: {question[:50]}...")

        # Build context
        context = self._build_context(question, explore_result, matched_tests)

        # Run agent loop
        response, sources = await self._run_agent_loop(
            system_prompt=UNIFIED_SYSTEM_PROMPT,
            user_prompt=context,
            on_progress=on_progress,
            conversation_history=conversation_history,
        )

        return AskResult(
            success=True,
            answer=response,
            sources=sources,
            matched_tests=matched_tests,
            explore_summary=explore_result.summary if explore_result else None,
            duration=0.0,  # Will be set by caller
            tool_calls_count=0,  # Will be set by caller
            error=None,
        )

    def _build_context(
        self,
        question: str,
        explore_result: ExploreResult,
        matched_tests: List[MatchedTest],
    ) -> str:
        """Build context string for the agent.

        Args:
            question: The user's question.
            explore_result: Results from explore phase.
            matched_tests: Matched test scaffolds.

        Returns:
            Formatted context string.
        """
        parts = []

        # User's question
        parts.append(f"## User's Question\n\n{question}")

        # Explore results
        if explore_result and explore_result.summary:
            parts.append(f"\n## Discovered Documentation\n\n{explore_result.summary}")

        # Matched tests
        if matched_tests:
            parts.append("\n## Related Tests Found\n")
            for test in matched_tests:
                parts.append(f"\n### {test.test_name} ({test.relevance_score:.0%} relevant)")
                parts.append(f"**Why this test exists:** {test.reason}")
                if test.success_criteria:
                    parts.append(f"\n**Success criteria for this topic:**\n{test.success_criteria}")
                if test.files_covered:
                    parts.append("\n**Files to reference:**")
                    for f in test.files_covered:
                        parts.append(f"- {f}")

        parts.append("\n---\n\nPlease answer the question above based on the documentation.")

        return "\n".join(parts)

    async def _run_agent_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        on_progress: Optional[Callable] = None,
        max_iterations: int = 100,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> tuple:
        """Run the agent loop with tool use.

        Args:
            system_prompt: System prompt for the agent.
            user_prompt: User prompt with context.
            on_progress: Optional progress callback.
            max_iterations: Maximum number of iterations.
            conversation_history: Optional list of previous messages.

        Returns:
            Tuple of (answer, sources).
        """
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                content = msg.get("content", "")
                if content:
                    messages.append({"role": msg["role"], "content": content})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": "[Previous response used tool operations]"})
            logger.debug(f"[ASK] Added {len(conversation_history)} messages from history")

        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})

        # Format tools for provider
        tools_formatted = self._format_tools_for_provider() if self.tools else None

        sources = []
        answer = ""

        for iteration in range(max_iterations):
            logger.info(f"[ASK] Agent loop iteration {iteration + 1}/{max_iterations}")

            # Call provider
            response = await self.provider.complete(messages, tools=tools_formatted)

            # Extract content
            content = response.get("content", "")
            # Handle both Anthropic format (tool_use) and OpenAI format (tool_calls)
            tool_calls = response.get("tool_use", response.get("tool_calls", []))

            logger.info(f"[ASK] Response: content_len={len(content)}, tool_calls={len(tool_calls) if tool_calls else 0}")
            if tool_calls:
                for tc in tool_calls:
                    tc_name = tc.get("name", "unknown")
                    tc_args = tc.get("input", tc.get("arguments", {}))
                    logger.info(f"[ASK]   Tool: {tc_name}, args: {tc_args}")

            if not tool_calls:
                # No more tool calls, we have the final answer
                logger.info("[ASK] No tool calls, returning final answer")
                answer = content
                break

            # Execute tool calls
            for tool_call in tool_calls:
                self._tool_calls_count += 1
                tool_name = tool_call.get("name", "")
                # Handle both Anthropic format (input) and OpenAI format (arguments)
                tool_args = tool_call.get("input", tool_call.get("arguments", {}))

                # Emit tool_start event
                if on_progress:
                    on_progress("tool_start", {
                        "tool": tool_name,
                        "params": tool_args,
                    })

                # Special handling for re_explore tool
                if tool_name == "re_explore":
                    topic = tool_args.get("topic", "")
                    logger.info(f"[ASK] Re-explore requested with topic: {topic[:100]!r}")

                    # Reset and re-initialize session with new topic
                    self.reset_session()
                    new_explore_result = await self.initialize_session(
                        topic=topic,
                        on_progress=on_progress,
                    )

                    # Build result message for the tool
                    result = (
                        f"Re-explored codebase for topic: {topic}\n"
                        f"Found {len(new_explore_result.files)} relevant files.\n"
                        f"Summary: {new_explore_result.summary}"
                    )

                    # Emit tool_end event
                    if on_progress:
                        on_progress("tool_end", {
                            "tool": tool_name,
                            "result": result[:500],
                        })

                    # Add tool result to messages
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [tool_call],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": result,
                    })

                    logger.info(f"[ASK] Re-explore complete: {len(new_explore_result.files)} files found")
                    continue  # Skip normal tool execution

                # Track file reads as sources
                if tool_name == "read_file":
                    file_path = tool_args.get("file_path", "")
                    if file_path and file_path not in sources:
                        sources.append(file_path)

                # Find and execute the tool
                result = await self._execute_tool(tool_name, tool_args)

                # Emit tool_end event
                if on_progress:
                    result_str = str(result)
                    # Pass full result for create_test (backend needs it for auto-save)
                    # Truncate other tools to avoid huge outputs in streaming
                    if tool_name == "create_test":
                        on_progress("tool_end", {
                            "tool": tool_name,
                            "result": result_str,
                        })
                    else:
                        on_progress("tool_end", {
                            "tool": tool_name,
                            "result": result_str[:500] if len(result_str) > 500 else result_str,
                        })

                # Add tool result to messages
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [tool_call],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": str(result),
                })

        # Follow-up call when answer is empty but tools were used
        if not answer and self._tool_calls_count > 0:
            logger.info(
                "[ASK] Empty answer after tool calls, requesting summary",
                extra={"tool_calls_count": self._tool_calls_count},
            )
            try:
                messages.append({
                    "role": "user",
                    "content": (
                        "Please provide a concise summary of what you found or did "
                        "based on the tool results above."
                    ),
                })
                follow_up_response = await self.provider.complete(
                    messages, tools=None, max_tokens=1024
                )
                answer = follow_up_response.get("content", "")
                logger.info(
                    "[ASK] Follow-up response generated",
                    extra={"content_length": len(answer)},
                )
                if answer and on_progress:
                    on_progress("chunk", {"content": answer})
            except Exception as e:
                logger.warning(
                    "[ASK] Follow-up call failed",
                    extra={"error": str(e)},
                )
        elif answer:
            logger.debug("[ASK] Answer present, skipping follow-up")

        # Extract additional sources from answer
        answer_sources = self._extract_sources(answer)
        sources.extend(s for s in answer_sources if s not in sources)

        return answer, sources

    async def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            Tool result as string.
        """
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = await tool.handler(arguments)
                    if hasattr(result, "output"):
                        if result.output is not None:
                            return str(result.output) if not isinstance(result.output, str) else result.output
                        # Return error if output is None but error exists
                        if hasattr(result, "error") and result.error:
                            return f"Error: {result.error}"
                        return "Tool completed but returned no output"
                    return str(result)
                except Exception as e:
                    return f"Error executing {tool_name}: {e}"
        return f"Unknown tool: {tool_name}"

    def _format_tools_for_provider(self) -> List[Dict]:
        """Format tools for the provider.

        Returns:
            List of tool definitions in provider format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

    def _extract_sources(self, text: str) -> List[str]:
        """Extract source paths from text.

        Args:
            text: Text containing source references.

        Returns:
            List of file paths.
        """
        # Match patterns like `path/to/file.md` or - path/to/file.md
        pattern = r"(?:`|^-\s+)([a-zA-Z0-9/_.-]+\.(?:md|yaml|yml|txt|json|py|pdf))"
        matches = re.findall(pattern, text, re.MULTILINE)
        return list(set(matches))
