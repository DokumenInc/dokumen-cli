"""
ExecutorAgent — runs test executor via Claude Agent SDK.

Wraps DokumenAgent._collect() and extracts ExecutorResult
from the message stream. retries on rate limits.
"""

import asyncio
import logging

from .base import DokumenAgent
from .messages import build_conversation_log, extract_tool_calls, extract_usage
from .types import ExecutorResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 60


class ExecutorAgent(DokumenAgent):
    """Executor agent that runs SOP tasks via the SDK.

    Usage:
        agent = ExecutorAgent(
            id="test-executor",
            system_prompt="You are a business SOP executor.",
            user_prompt="Read the customer ticket and follow the refund SOP...",
            sdk_tools=["Read", "Glob"],
        )
        result = await agent.run()
    """

    async def run(self) -> ExecutorResult:
        """Execute the agent and return structured results.

        retries up to MAX_RETRIES times on rate limits with a 60s pause.
        """
        from claude_agent_sdk import ResultMessage as _ResultMessage

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(
                "Executor starting",
                extra={
                    "agent_id": self.id,
                    "timeout": self.timeout,
                    "attempt": attempt,
                },
            )

            try:
                qr = await asyncio.wait_for(
                    self._collect(self.user_prompt),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Executor timed out",
                    extra={
                        "agent_id": self.id,
                        "timeout": self.timeout,
                    },
                )
                return ExecutorResult(
                    success=False,
                    final_response="",
                    tool_calls=[],
                    input_tokens=0,
                    output_tokens=0,
                    conversation_log=[],
                    duration_ms=int(self.timeout * 1000),
                    error=f"Executor timed out after {self.timeout}s",
                )
            except Exception as exc:
                # bundled CLI sometimes exits with code 1 after completing work.
                # treat as a failed result rather than crashing the whole worker.
                err_msg = str(exc)
                logger.warning(
                    "Executor _collect raised exception",
                    extra={
                        "agent_id": self.id,
                        "error": err_msg,
                        "error_type": type(exc).__name__,
                        "attempt": attempt,
                    },
                )
                return ExecutorResult(
                    success=False,
                    final_response="",
                    tool_calls=[],
                    input_tokens=0,
                    output_tokens=0,
                    conversation_log=[],
                    duration_ms=0,
                    error=err_msg,
                )

            # check if we got a real result or got rate limited
            result_msg = qr.result if isinstance(qr.result, _ResultMessage) else None

            # retry on rate limit or any failure without a valid result
            if result_msg is None and attempt < MAX_RETRIES:
                print(
                    f"  ⚠ no result (likely rate limited), attempt {attempt}/{MAX_RETRIES} — retrying in {RETRY_DELAY_SECONDS}s...",
                    flush=True,
                )
                logger.warning(
                    "No valid result, retrying",
                    extra={
                        "agent_id": self.id,
                        "attempt": attempt,
                        "delay": RETRY_DELAY_SECONDS,
                        "qr_result_type": type(qr.result).__name__ if qr.result else "None",
                    },
                )
                await asyncio.sleep(RETRY_DELAY_SECONDS)
                continue

            is_error = result_msg.is_error if result_msg else True
            final_text = (result_msg.result or "") if result_msg else ""
            duration = result_msg.duration_ms if result_msg else 0

            # bundled CLI often crashes before populating result_msg.result.
            # fall back to extracting the last assistant text from collected messages.
            if not final_text.strip() and qr.messages:
                for msg in reversed(qr.messages):
                    if hasattr(msg, "content") and getattr(msg, "role", "") != "user":
                        content = getattr(msg, "content", None)
                        if isinstance(content, str) and content.strip():
                            final_text = content
                            break
                        elif isinstance(content, list):
                            texts = []
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    texts.append(block.get("text", ""))
                                elif (
                                    hasattr(block, "type") and getattr(block, "type", "") == "text"
                                ):
                                    texts.append(getattr(block, "text", ""))
                            combined = "\n".join(t for t in texts if t.strip())
                            if combined.strip():
                                final_text = combined
                                break

            has_output = bool(final_text and final_text.strip())
            effective_error = is_error and not has_output

            usage = extract_usage(result_msg) if result_msg else {}
            result = ExecutorResult(
                success=not effective_error,
                final_response=final_text,
                tool_calls=extract_tool_calls(qr.messages),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                conversation_log=build_conversation_log(qr.messages),
                duration_ms=duration,
                error=None if not effective_error else (final_text or "Executor failed"),
            )

            logger.info(
                "Executor completed",
                extra={
                    "agent_id": self.id,
                    "success": result.success,
                    "tool_call_count": len(result.tool_calls),
                    "duration_ms": result.duration_ms,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
            )

            return result

        # exhausted retries
        return ExecutorResult(
            success=False,
            final_response="",
            tool_calls=[],
            input_tokens=0,
            output_tokens=0,
            conversation_log=[],
            duration_ms=0,
            error=f"Rate limited after {MAX_RETRIES} retries",
        )
