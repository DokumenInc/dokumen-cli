"""
QueryRunner protocol and implementations.

Provides an injectable abstraction over claude_agent_sdk.query() for testing.
Production code uses SDKQueryRunner; tests use MockQueryRunner.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, AsyncIterator, List, Optional, Protocol, runtime_checkable

from claude_agent_sdk import ClaudeAgentOptions, Message, query

logger = logging.getLogger(__name__)


@runtime_checkable
class QueryRunner(Protocol):
    """Protocol for running Claude Agent SDK queries."""

    async def run(
        self, prompt: str | AsyncIterable[dict[str, Any]], options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        """Run a query and yield messages from the stream."""
        ...  # pragma: no cover


class SDKQueryRunner:
    """Production implementation: calls claude_agent_sdk.query()."""

    async def run(
        self, prompt: str | AsyncIterable[dict[str, Any]], options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        """Run a real SDK query, yielding each message from the stream.

        Note: Unsets CLAUDECODE env var before spawning the SDK subprocess
        to avoid "nested session" errors when running inside Claude Code.
        """
        import os

        # Prevent "nested session" error when running inside Claude Code
        claudecode_val = os.environ.pop("CLAUDECODE", None)

        logger.info(
            "Starting SDK query",
            extra={
                "prompt_length": len(prompt) if isinstance(prompt, str) else None,
                "permission_mode": options.permission_mode,
                "max_turns": options.max_turns,
            },
        )
        try:
            async for msg in query(prompt=prompt, options=options):
                yield msg
            logger.info("SDK query stream completed")
        finally:
            # Restore CLAUDECODE if it was set
            if claudecode_val is not None:
                os.environ["CLAUDECODE"] = claudecode_val


@dataclass
class MockCall:
    """Record of a call made to MockQueryRunner."""

    prompt: str
    options: ClaudeAgentOptions


class MockQueryRunner:
    """Test implementation: replays pre-configured message sequences.

    Usage:
        runner = MockQueryRunner([make_init(), make_assistant("hello"), make_result("hello")])
        async for msg in runner.run("prompt", options):
            process(msg)
        assert len(runner.calls) == 1
    """

    def __init__(self, messages: List[Any]):
        self._messages = messages
        self.calls: List[MockCall] = field(default_factory=list)
        self.calls = []

    async def run(
        self, prompt: str, options: Optional[ClaudeAgentOptions] = None
    ) -> AsyncIterator[Any]:
        """Yield pre-configured messages, recording the call."""
        self.calls.append(MockCall(prompt=prompt, options=options))
        for msg in self._messages:
            yield msg
