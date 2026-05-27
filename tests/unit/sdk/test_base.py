"""Tests for dokumen.sdk.base — DokumenAgent base class."""

import pytest
from unittest.mock import MagicMock

from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, UserMessage

from dokumen.sdk.base import DokumenAgent
from dokumen.sdk.query_runner import MockQueryRunner
from dokumen.sdk.testing import make_assistant, make_init, make_result, make_tool_result
from dokumen.sdk.types import QueryResult


class TestDokumenAgentInit:
    def test_init_sets_basic_fields(self):
        """Agent stores id, system_prompt, user_prompt, timeout."""
        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="You are a tester.",
            user_prompt="Test this.",
            sdk_tools=["Read", "Glob"],
            query_runner=runner,
            timeout=30.0,
        )

        assert agent.id == "test-agent"
        assert agent.system_prompt == "You are a tester."
        assert agent.user_prompt == "Test this."
        assert agent.timeout == 30.0

    def test_init_builds_options_with_sdk_tools(self):
        """ClaudeAgentOptions.allowed_tools contains SDK tools."""
        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=["Read", "Glob", "Bash"],
            query_runner=runner,
        )

        assert "Read" in agent._options.allowed_tools
        assert "Glob" in agent._options.allowed_tools
        assert "Bash" in agent._options.allowed_tools

    def test_init_bypass_permissions(self):
        """ClaudeAgentOptions.permission_mode is bypassPermissions."""
        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        assert agent._options.permission_mode == "bypassPermissions"

    def test_init_max_turns(self):
        """max_turns is set on options."""
        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
            max_turns=50,
        )

        assert agent._options.max_turns == 50

    def test_init_default_max_turns(self):
        """Default max_turns is 100."""
        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        assert agent._options.max_turns == 100

    def test_init_model_override(self):
        """model is set on options when provided."""
        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
            model="claude-haiku-4-5-20251001",
        )

        assert agent._options.model == "claude-haiku-4-5-20251001"

    def test_init_no_model(self):
        """model is None when not provided."""
        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        assert agent._options.model is None

    def test_init_empty_sdk_tools(self):
        """Empty sdk_tools results in empty allowed_tools."""
        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        assert agent._options.allowed_tools == []

    def test_init_with_external_mcp_servers(self):
        """External MCP server configs are merged into mcp_servers."""
        runner = MockQueryRunner([])
        playwright_config = {"command": "npx", "args": ["playwright-mcp"]}
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=["Read"],
            mcp_servers={"playwright": playwright_config},
            query_runner=runner,
        )

        assert "playwright" in agent._options.mcp_servers
        assert agent._options.mcp_servers["playwright"] == playwright_config

    def test_init_with_mcp_tools_and_external_servers(self):
        """Both Dokumen MCP tools and external MCP servers are combined."""
        mock_tool = MagicMock()
        mock_tool.name = "code_read_file"
        runner = MockQueryRunner([])
        playwright_config = {"command": "npx", "args": ["playwright-mcp"]}
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=["Read"],
            mcp_tools=[mock_tool],
            mcp_servers={"playwright": playwright_config},
            query_runner=runner,
        )

        # Both dokumen and external MCP servers should be present
        assert "playwright" in agent._options.mcp_servers
        assert "mcp__dokumen__code_read_file" in agent._options.allowed_tools

    def test_init_with_mcp_tools(self):
        """MCP tools are added to allowed_tools with mcp__dokumen__ prefix."""
        mock_tool = MagicMock()
        mock_tool.name = "code_read_file"
        runner = MockQueryRunner([])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=["Read"],
            mcp_tools=[mock_tool],
            query_runner=runner,
        )

        assert "Read" in agent._options.allowed_tools
        assert "mcp__dokumen__code_read_file" in agent._options.allowed_tools


class TestDokumenAgentCollect:
    async def test_collect_returns_query_result(self):
        """_collect() returns QueryResult with session_id, messages, result."""
        messages = [
            make_init(session_id="sess-abc"),
            make_assistant("hello world"),
            make_result("hello world"),
        ]
        runner = MockQueryRunner(messages)
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        qr = await agent._collect("test prompt")

        assert isinstance(qr, QueryResult)
        assert qr.session_id == "sess-abc"
        assert len(qr.messages) == 1  # Only AssistantMessages
        assert qr.result is not None
        assert qr.result.result == "hello world"

    async def test_collect_captures_session_id(self):
        """_collect() captures session_id from init SystemMessage."""
        messages = [
            make_init(session_id="my-session-42"),
            make_result("done"),
        ]
        runner = MockQueryRunner(messages)
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        qr = await agent._collect("prompt")

        assert qr.session_id == "my-session-42"

    async def test_collect_no_init_message(self):
        """_collect() returns None session_id when no init message."""
        messages = [
            make_assistant("response"),
            make_result("response"),
        ]
        runner = MockQueryRunner(messages)
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        qr = await agent._collect("prompt")

        assert qr.session_id is None

    async def test_collect_multiple_assistant_messages(self):
        """_collect() stores all AssistantMessages and UserMessages."""
        messages = [
            make_init(),
            make_assistant("first"),
            make_tool_result("tc_1", "tool output"),
            make_assistant("second"),
            make_result("second"),
        ]
        runner = MockQueryRunner(messages)
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        qr = await agent._collect("prompt")

        # 2 assistant + 1 user (tool result)
        assert len(qr.messages) == 3

    async def test_collect_no_result_message(self):
        """_collect() returns None result when no ResultMessage."""
        messages = [make_init(), make_assistant("partial")]
        runner = MockQueryRunner(messages)
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        qr = await agent._collect("prompt")

        assert qr.result is None

    async def test_collect_passes_prompt_to_runner(self):
        """_collect() sends the correct prompt to the runner."""
        runner = MockQueryRunner([make_init(), make_result("ok")])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        await agent._collect("my specific prompt")

        assert runner.calls[0].prompt == "my specific prompt"

    async def test_collect_passes_options_to_runner(self):
        """_collect() passes the constructed ClaudeAgentOptions to the runner."""
        runner = MockQueryRunner([make_init(), make_result("ok")])
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="my system prompt",
            user_prompt="usr",
            sdk_tools=["Read"],
            query_runner=runner,
        )

        await agent._collect("prompt")

        opts = runner.calls[0].options
        assert opts.system_prompt == "my system prompt"
        assert "Read" in opts.allowed_tools

    async def test_collect_captures_user_messages(self):
        """_collect() stores both AssistantMessages and UserMessages (tool results)."""
        messages = [
            make_init(),
            make_assistant("calling tool", tool_calls=[{"id": "tc_1", "name": "Read", "input": {}}]),
            make_tool_result("tc_1", "file contents"),
            make_assistant("done"),
            make_result("done"),
        ]
        runner = MockQueryRunner(messages)
        agent = DokumenAgent(
            id="test-agent",
            system_prompt="sys",
            user_prompt="usr",
            sdk_tools=[],
            query_runner=runner,
        )

        qr = await agent._collect("prompt")

        # 2 AssistantMessages + 1 UserMessage (tool result)
        assert len(qr.messages) == 3
        assert isinstance(qr.messages[0], AssistantMessage)
        assert isinstance(qr.messages[1], UserMessage)
        assert isinstance(qr.messages[2], AssistantMessage)
