"""
DokumenAgent base class.

Wraps claude_agent_sdk.query() with Dokumen-specific configuration:
SDK built-in tools, Dokumen MCP tools, external MCP servers (Playwright),
and validation hooks.
"""

import logging
from typing import Any, AsyncIterable, Callable, Dict, List, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    UserMessage,
)

from .query_runner import MockQueryRunner, QueryRunner, SDKQueryRunner
from .types import QueryResult

logger = logging.getLogger(__name__)


class DokumenAgent:
    """Base class for Dokumen agents running on the Claude Agent SDK.

    Handles:
    - Building ClaudeAgentOptions from sdk_tools + mcp_tools + mcp_servers + hooks
    - Running queries via QueryRunner (injectable for testing)
    - Collecting message streams into QueryResult
    """

    def __init__(
        self,
        id: str,
        system_prompt: str,
        user_prompt: str,
        sdk_tools: Optional[List[str]] = None,
        mcp_tools: Optional[List[Any]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        playwright_tool_names: Optional[List[str]] = None,
        max_turns: int = 100,
        timeout: float = 60.0,
        query_runner: Optional[QueryRunner] = None,
        on_tool_call: Optional[Callable[[str, dict, Any], None]] = None,
        tools_config: Optional[Any] = None,
        model: Optional[str] = None,
        agents: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a DokumenAgent.

        Args:
            id: Unique agent identifier.
            system_prompt: LLM system prompt.
            user_prompt: User prompt template.
            sdk_tools: SDK built-in tool names (e.g. ["Read", "Glob", "Bash"]).
            mcp_tools: Dokumen-specific ToolDefinition objects for MCP.
            mcp_servers: External MCP server configs (e.g. Playwright).
                         Dict of name → McpServerConfig.
            playwright_tool_names: Prefixed Playwright tool names to add to
                                   allowed_tools (e.g. ["mcp__playwright__browser_navigate"]).
            max_turns: Maximum conversation turns.
            timeout: Timeout in seconds for the entire agent run.
            query_runner: Injectable query runner (defaults to SDKQueryRunner).
            on_tool_call: Callback for tool call tracking.
            tools_config: ToolsConfig for hook validation rules.
            model: Optional model override.
            agents: Optional dict of agent name → AgentDefinition for SDK-native
                    subagent support. When provided, the executor LLM can use the
                    built-in Agent tool to spawn these as isolated subagents.
        """
        self.id = id
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.timeout = timeout
        self._runner = query_runner or SDKQueryRunner()

        logger.info(
            "Initializing DokumenAgent",
            extra={
                "agent_id": id,
                "sdk_tools": sdk_tools,
                "mcp_tool_count": len(mcp_tools) if mcp_tools else 0,
                "max_turns": max_turns,
                "timeout": timeout,
                "model": model,
            },
        )

        # Build allowed_tools list
        allowed: List[str] = list(sdk_tools or [])
        if mcp_tools:
            allowed += [f"mcp__dokumen__{t.name}" for t in mcp_tools]
        if playwright_tool_names:
            allowed += list(playwright_tool_names)

        # Build MCP server configs dict
        all_mcp_servers: Dict[str, Any] = {}
        if mcp_tools:
            from .tools import create_dokumen_mcp_server

            server_config = create_dokumen_mcp_server(mcp_tools, on_tool_call)
            if server_config:
                all_mcp_servers["dokumen"] = server_config
        if mcp_servers:
            all_mcp_servers.update(mcp_servers)

        # Build hooks only when tool policy or callbacks are active.
        # Explore needs broad read access, and hooks can cause SDK CLI issues.
        hooks = None
        if tools_config is not None or on_tool_call is not None:
            from .hooks import build_validation_hooks
            hooks = build_validation_hooks(tools_config, on_tool_call)

        # Determine permission mode: can't use bypassPermissions as root
        # (Claude CLI rejects --dangerously-skip-permissions under root/sudo)
        import os

        if os.getuid() == 0:
            perm_mode = "acceptEdits"
            logger.info(
                "Running as root, using acceptEdits instead of bypassPermissions",
                extra={"agent_id": id},
            )
        else:
            perm_mode = "bypassPermissions"

        self._options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=allowed,
            permission_mode=perm_mode,
            max_turns=max_turns,
            mcp_servers=all_mcp_servers if all_mcp_servers else {},
            hooks=hooks if hooks else None,
            model=model,
            max_buffer_size=10 * 1024 * 1024,  # 10MB for large MCP responses (screenshots)
            agents=agents,
        )

    async def _collect(self, prompt: str) -> QueryResult:
        """Run query, collect all messages, return structured result.

        Args:
            prompt: The user prompt to send.

        Returns:
            QueryResult with session_id, messages, and result.
        """
        session_id = None
        all_messages = []
        result_msg = None

        prompt_for_runner: str | AsyncIterable[dict[str, Any]] = prompt
        use_streaming_prompt = (
            (self._options.mcp_servers or self._options.hooks)
            and isinstance(self._runner, SDKQueryRunner)
        )
        if use_streaming_prompt:
            async def _stream_prompt() -> AsyncIterable[dict[str, Any]]:
                yield {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": prompt,
                    },
                }
            prompt_for_runner = _stream_prompt()

        logger.info(
            "Starting agent query",
            extra={
                "agent_id": self.id,
                "prompt_length": len(prompt),
                "streaming_prompt": use_streaming_prompt,
            },
        )

        try:
            async for msg in self._runner.run(prompt_for_runner, self._options):
                if isinstance(msg, SystemMessage) and msg.subtype == "init":
                    session_id = msg.data.get("session_id")
                    logger.debug(
                        "Session initialized",
                        extra={"agent_id": self.id, "session_id": session_id},
                    )
                elif isinstance(msg, (AssistantMessage, UserMessage)):
                    all_messages.append(msg)
                    # live progress
                    if isinstance(msg, AssistantMessage):
                        content = getattr(msg, 'content', None)
                        if content:
                            for block in content:
                                # handle both object and dict blocks
                                if isinstance(block, dict):
                                    bt = block.get('type', '')
                                    if bt == 'tool_use':
                                        tn = block.get('name', '?')
                                        ti = block.get('input', {})
                                        d = ti.get('file_path', '') or ti.get('command', '') or ti.get('pattern', '') if isinstance(ti, dict) else ''
                                        print(f"  ↳ {tn}: {str(d)[:120]}" if d else f"  ↳ {tn}", flush=True)
                                    elif bt == 'text':
                                        t = block.get('text', '').strip()
                                        if t:
                                            print(f"  → {t.split(chr(10))[0][:150]}", flush=True)
                                else:
                                    bt = getattr(block, 'type', '')
                                    if bt == 'tool_use':
                                        tn = getattr(block, 'name', '?')
                                        ti = getattr(block, 'input', {})
                                        d = ''
                                        if isinstance(ti, dict):
                                            d = ti.get('file_path', '') or ti.get('command', '') or ti.get('pattern', '')
                                        print(f"  ↳ {tn}: {str(d)[:120]}" if d else f"  ↳ {tn}", flush=True)
                                    elif bt == 'text':
                                        t = getattr(block, 'text', '').strip()
                                        if t:
                                            print(f"  → {t.split(chr(10))[0][:150]}", flush=True)
                        else:
                            # fallback: try to print something useful
                            msg_str = str(msg)[:200]
                            if 'tool' in msg_str.lower() or len(msg_str) > 50:
                                print(f"  · {msg_str[:150]}", flush=True)
                elif isinstance(msg, ResultMessage):
                    result_msg = msg
                    logger.info(
                        "Query completed",
                        extra={
                            "agent_id": self.id,
                            "session_id": session_id,
                            "is_error": msg.is_error,
                            "num_turns": msg.num_turns,
                            "duration_ms": msg.duration_ms,
                        },
                    )
                else:
                    # handle rate limits, subagent events, task events, etc.
                    msg_type = type(msg).__name__
                    if 'RateLimit' in msg_type:
                        data = getattr(msg, 'data', {}) or {}
                        retry_after = data.get('retry_after', '') or data.get('retryAfter', '') or data.get('delay', '')
                        limit_type = data.get('type', '') or data.get('limit_type', '') or ''
                        msg_text = data.get('message', '') or data.get('error', '') or ''
                        parts = ["  ⚠ rate limited"]
                        if limit_type:
                            parts[0] += f" ({limit_type})"
                        if retry_after:
                            parts[0] += f" — retrying in {retry_after}s"
                        else:
                            parts[0] += " — waiting for retry..."
                        if msg_text and len(str(msg_text)) < 100:
                            parts.append(f"    {msg_text}")
                        for p in parts:
                            print(p, flush=True)
                    else:
                        subtype = getattr(msg, 'subtype', '') or ''
                        data = getattr(msg, 'data', {}) or {}
                        if subtype == 'task_started':
                            task_name = data.get('name', '') or data.get('task', '') or ''
                            print(f"  ▶ task started{': ' + task_name if task_name else ''}", flush=True)
                        elif subtype == 'task_progress':
                            # only print progress if there's meaningful content
                            progress = data.get('message', '') or data.get('progress', '') or data.get('content', '')
                            if isinstance(progress, str) and progress.strip():
                                print(f"  · {progress.strip()[:150]}", flush=True)
                            # else: suppress empty progress spam
                        elif subtype == 'task_notification':
                            status = data.get('status', '')
                            task_id = data.get('task_id', '')[:12] if data.get('task_id') else ''
                            note = data.get('message', '') or data.get('notification', '') or status
                            if note:
                                print(f"  ℹ {note}", flush=True)
                            elif status:
                                print(f"  ℹ task {task_id}: {status}", flush=True)
                        elif subtype == 'task_completed':
                            task_name = data.get('name', '') or data.get('task', '') or ''
                            print(f"  ✓ task completed{': ' + task_name if task_name else ''}", flush=True)
                        elif subtype == 'api_retry':
                            delay = data.get('delay', '') or data.get('retry_after', '')
                            attempt = data.get('attempt', '') or data.get('retry_count', '')
                            reason = data.get('error', '') or data.get('reason', '') or 'rate limited'
                            parts = [f"  ⏳ retrying api call"]
                            if attempt:
                                parts[0] += f" (attempt {attempt})"
                            if delay:
                                parts[0] += f" — waiting {delay}s"
                            if reason and str(reason) != 'rate limited':
                                parts[0] += f" ({str(reason)[:80]})"
                            print(parts[0], flush=True)
                        elif subtype and subtype != 'init':
                            # catch-all for unknown event types — show data if available
                            detail = data.get('message', '') or data.get('content', '') or ''
                            if detail:
                                print(f"  ◆ {subtype}: {str(detail)[:120]}", flush=True)
                            else:
                                print(f"  ◆ {subtype}", flush=True)
        except Exception as e:
            # bundled CLI sometimes crashes after completing work (exit code 1).
            # salvage whatever messages and result we already collected.
            logger.error(
                "Fatal error in message reader",
                extra={"error": str(e), "had_result": result_msg is not None, "messages_collected": len(all_messages)},
            )

        return QueryResult(
            session_id=session_id,
            messages=all_messages,
            result=result_msg,
        )
