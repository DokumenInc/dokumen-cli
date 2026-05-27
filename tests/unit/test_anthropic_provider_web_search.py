"""
Tests for Anthropic provider handling of server-side web search tool.

Tests that:
1. Server-side tools are formatted as web_search_20250305 (not function tools)
2. server_tool_use and web_search_tool_result response blocks don't crash
3. Mixed responses (text + server tool use + tool_use) are handled correctly
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dokumen.providers.anthropic import AnthropicProvider


class TestServerSideToolFormatting:
    """Tests for formatting server-side tools in the Anthropic provider."""

    def test_server_side_tool_formatted_as_web_search_type(self):
        """Server-side tools should be formatted as web_search_20250305, not function tools."""
        provider = AnthropicProvider(api_key="test-key")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "anthropic_web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}, "_server_side": True},
                },
                "_server_tool_config": {
                    "type": "web_search_20250305",
                    "max_uses": 25,
                },
            }
        ]

        formatted = provider._format_tools_for_api(tools)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "web_search_20250305"
        assert formatted[0]["name"] == "web_search"
        assert formatted[0].get("max_uses") == 25

    def test_regular_function_tool_unchanged(self):
        """Regular function tools should still be formatted the standard way."""
        provider = AnthropicProvider(api_key="test-key")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]

        formatted = provider._format_tools_for_api(tools)
        assert len(formatted) == 1
        assert formatted[0]["name"] == "read_file"
        assert "input_schema" in formatted[0]

    def test_mixed_tools_formatted_correctly(self):
        """Mix of server-side and regular tools should both be formatted correctly."""
        provider = AnthropicProvider(api_key="test-key")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "anthropic_web_search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}, "_server_side": True},
                },
                "_server_tool_config": {
                    "type": "web_search_20250305",
                    "max_uses": 10,
                },
            },
        ]

        formatted = provider._format_tools_for_api(tools)
        assert len(formatted) == 2
        # First should be a regular function tool
        assert formatted[0]["name"] == "read_file"
        assert "input_schema" in formatted[0]
        # Second should be a server-side tool
        assert formatted[1]["type"] == "web_search_20250305"
        assert formatted[1]["name"] == "web_search"

    def test_server_tool_with_allowed_domains(self):
        """Server-side tool config with allowed_domains should pass through."""
        provider = AnthropicProvider(api_key="test-key")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "anthropic_web_search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}, "_server_side": True},
                },
                "_server_tool_config": {
                    "type": "web_search_20250305",
                    "max_uses": 20,
                    "allowed_domains": ["example.com"],
                    "blocked_domains": ["spam.com"],
                },
            }
        ]

        formatted = provider._format_tools_for_api(tools)
        assert formatted[0].get("allowed_domains") == ["example.com"]
        assert formatted[0].get("blocked_domains") == ["spam.com"]


class TestServerToolResponseHandling:
    """Tests for handling server_tool_use and web_search_tool_result blocks."""

    def test_normalize_response_handles_server_tool_use(self):
        """server_tool_use blocks should be logged but not crash."""
        provider = AnthropicProvider(api_key="test-key")

        # Create mock response with server_tool_use block
        server_block = MagicMock()
        server_block.type = "server_tool_use"
        server_block.id = "srv_123"
        server_block.name = "web_search"
        server_block.input = {"query": "biomass news"}

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Here are the results."

        response = MagicMock()
        response.content = [server_block, text_block]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)

        result = provider._normalize_response(response)
        assert "Here are the results." in result["content"]

    def test_normalize_response_handles_web_search_tool_result(self):
        """web_search_tool_result blocks should be handled without crash."""
        provider = AnthropicProvider(api_key="test-key")

        # Create mock web_search_tool_result block
        search_result_item = MagicMock()
        search_result_item.type = "web_search_result"
        search_result_item.url = "https://biomassmag.com/article"
        search_result_item.title = "Biomass News"
        search_result_item.encrypted_content = "encrypted..."
        search_result_item.page_age = "2 days ago"

        ws_result_block = MagicMock()
        ws_result_block.type = "web_search_tool_result"
        ws_result_block.tool_use_id = "srv_123"
        ws_result_block.content = [search_result_item]

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Based on my search..."

        response = MagicMock()
        response.content = [ws_result_block, text_block]
        response.usage = MagicMock(input_tokens=200, output_tokens=100)

        result = provider._normalize_response(response)
        assert "Based on my search..." in result["content"]

    def test_mixed_response_text_server_tool_regular_tool(self):
        """Mixed response with text, server_tool_use, web_search_tool_result, and tool_use."""
        provider = AnthropicProvider(api_key="test-key")

        text1 = MagicMock()
        text1.type = "text"
        text1.text = "Let me search. "

        server_use = MagicMock()
        server_use.type = "server_tool_use"
        server_use.id = "srv_1"
        server_use.name = "web_search"
        server_use.input = {"query": "test"}

        ws_result = MagicMock()
        ws_result.type = "web_search_tool_result"
        ws_result.tool_use_id = "srv_1"
        ws_result.content = []

        text2 = MagicMock()
        text2.type = "text"
        text2.text = "Now let me read a file."

        tool_use = MagicMock()
        tool_use.type = "tool_use"
        tool_use.id = "tu_1"
        tool_use.name = "read_file"
        tool_use.input = {"path": "docs/api.md"}

        response = MagicMock()
        response.content = [text1, server_use, ws_result, text2, tool_use]
        response.usage = MagicMock(input_tokens=300, output_tokens=150)

        result = provider._normalize_response(response)
        # Text blocks should be concatenated
        assert "Let me search." in result["content"]
        assert "Now let me read a file." in result["content"]
        # Regular tool_use should be captured
        assert "tool_use" in result
        assert len(result["tool_use"]) == 1
        assert result["tool_use"][0]["name"] == "read_file"


# TestAgentFormatToolsPassesServerConfig removed:
# These tests tested AgentObject._format_tools_for_provider() which was part
# of the legacy execution path. The SDK path handles tool formatting internally.
