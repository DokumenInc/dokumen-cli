"""
Tests for Anthropic Web Search tool (server-side tool).

TDD: Tests written first — the sentinel tool creation, config model,
handler guard, and server tool config attachment.
"""
import pytest

from dokumen.tools_object import ToolDefinition, ToolResult


class TestAnthropicWebSearchConfig:
    """Tests for AnthropicWebSearchConfig Pydantic model."""

    def test_config_model_exists(self):
        from dokumen.config import AnthropicWebSearchConfig
        assert AnthropicWebSearchConfig is not None

    def test_defaults(self):
        from dokumen.config import AnthropicWebSearchConfig
        cfg = AnthropicWebSearchConfig()
        assert cfg.max_uses is None
        assert cfg.allowed_domains is None
        assert cfg.blocked_domains is None

    def test_max_uses_validation(self):
        from pydantic import ValidationError
        from dokumen.config import AnthropicWebSearchConfig

        # Valid
        cfg = AnthropicWebSearchConfig(max_uses=30)
        assert cfg.max_uses == 30

        # Too low
        with pytest.raises(ValidationError):
            AnthropicWebSearchConfig(max_uses=0)

        # Too high
        with pytest.raises(ValidationError):
            AnthropicWebSearchConfig(max_uses=51)

    def test_domains_accepted(self):
        from dokumen.config import AnthropicWebSearchConfig
        cfg = AnthropicWebSearchConfig(
            allowed_domains=["biomassmag.com", "eia.gov"],
            blocked_domains=["spam.com"],
        )
        assert cfg.allowed_domains == ["biomassmag.com", "eia.gov"]
        assert cfg.blocked_domains == ["spam.com"]

    def test_tool_config_map_has_anthropic_web_search(self):
        from dokumen.config import ToolConfigMap
        tcm = ToolConfigMap()
        assert hasattr(tcm, "anthropic_web_search")
        from dokumen.config import AnthropicWebSearchConfig
        assert isinstance(tcm.anthropic_web_search, AnthropicWebSearchConfig)


class TestCreateAnthropicWebSearchTool:
    """Tests for create_anthropic_web_search_tool sentinel factory."""

    def test_factory_exists(self):
        from dokumen.tools_object import create_anthropic_web_search_tool
        assert callable(create_anthropic_web_search_tool)

    def test_returns_tool_definition(self):
        from dokumen.tools_object import create_anthropic_web_search_tool
        tool = create_anthropic_web_search_tool()
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "anthropic_web_search"

    def test_description_mentions_anthropic(self):
        from dokumen.tools_object import create_anthropic_web_search_tool
        tool = create_anthropic_web_search_tool()
        assert "anthropic" in tool.description.lower() or "web search" in tool.description.lower()

    def test_handler_raises_runtime_error(self):
        """Sentinel handler must raise RuntimeError if called — it's server-side."""
        import asyncio
        from dokumen.tools_object import create_anthropic_web_search_tool
        tool = create_anthropic_web_search_tool()
        with pytest.raises(RuntimeError, match="server-side"):
            asyncio.run(tool.handler({"query": "test"}))

    def test_server_tool_config_attached(self):
        """Handler must carry _server_tool_config for the provider to read."""
        from dokumen.tools_object import create_anthropic_web_search_tool
        tool = create_anthropic_web_search_tool(max_uses=25)
        assert hasattr(tool.handler, "_server_tool_config")
        cfg = tool.handler._server_tool_config
        assert cfg["type"] == "web_search_20250305"
        assert cfg["max_uses"] == 25

    def test_server_tool_config_defaults(self):
        from dokumen.tools_object import create_anthropic_web_search_tool
        tool = create_anthropic_web_search_tool()
        cfg = tool.handler._server_tool_config
        assert cfg["type"] == "web_search_20250305"
        assert cfg["max_uses"] == 20  # default

    def test_allowed_domains_passthrough(self):
        from dokumen.tools_object import create_anthropic_web_search_tool
        tool = create_anthropic_web_search_tool(
            allowed_domains=["example.com", "test.org"]
        )
        cfg = tool.handler._server_tool_config
        assert cfg["allowed_domains"] == ["example.com", "test.org"]

    def test_blocked_domains_passthrough(self):
        from dokumen.tools_object import create_anthropic_web_search_tool
        tool = create_anthropic_web_search_tool(
            blocked_domains=["spam.com"]
        )
        cfg = tool.handler._server_tool_config
        assert cfg["blocked_domains"] == ["spam.com"]

    def test_no_domains_omitted_from_config(self):
        """When domains are None, they should not be in _server_tool_config."""
        from dokumen.tools_object import create_anthropic_web_search_tool
        tool = create_anthropic_web_search_tool()
        cfg = tool.handler._server_tool_config
        assert "allowed_domains" not in cfg
        assert "blocked_domains" not in cfg

    def test_in_standalone_tools(self):
        """anthropic_web_search must be registered in STANDALONE_TOOLS."""
        from dokumen.tools_object import STANDALONE_TOOLS
        assert "anthropic_web_search" in STANDALONE_TOOLS


class TestConstantsIncludeAnthropicWebSearch:
    """Tests for anthropic_web_search in schema constants."""

    def test_in_valid_executor_tools(self):
        from dokumen_schema.constants import VALID_EXECUTOR_TOOLS
        assert "anthropic_web_search" in VALID_EXECUTOR_TOOLS

    def test_in_cli_resolvable_tools(self):
        from dokumen_schema.constants import CLI_RESOLVABLE_TOOLS
        assert "anthropic_web_search" in CLI_RESOLVABLE_TOOLS


class TestLoaderResolvesAnthropicWebSearch:
    """Tests for resolve_tools handling of anthropic_web_search."""

    def test_resolves_anthropic_web_search(self):
        from dokumen.loader import resolve_tools
        tools = resolve_tools(["anthropic_web_search"], base_dir=".")
        assert len(tools) == 1
        assert tools[0].name == "anthropic_web_search"

    def test_resolves_with_config(self):
        from dokumen.loader import resolve_tools
        from dokumen.config import ToolsConfig, ToolConfigMap, AnthropicWebSearchConfig

        tools_config = ToolsConfig(
            config=ToolConfigMap(
                anthropic_web_search=AnthropicWebSearchConfig(
                    max_uses=30,
                    allowed_domains=["biomassmag.com"],
                )
            )
        )
        tools = resolve_tools(
            ["anthropic_web_search"],
            base_dir=".",
            tools_config=tools_config,
        )
        assert len(tools) == 1
        cfg = tools[0].handler._server_tool_config
        assert cfg["max_uses"] == 30
        assert cfg["allowed_domains"] == ["biomassmag.com"]

    def test_coexists_with_perplexity_web_search(self):
        """Both web_search (Perplexity) and anthropic_web_search must resolve together."""
        from dokumen.loader import resolve_tools

        tools = resolve_tools(
            ["web_search", "anthropic_web_search"],
            base_dir=".",
        )
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "anthropic_web_search" in names

    def test_backward_compat_perplexity_still_works(self):
        """Existing web_search (Perplexity) must still resolve correctly."""
        from dokumen.loader import resolve_tools
        tools = resolve_tools(["web_search"], base_dir=".")
        assert len(tools) == 1
        assert tools[0].name == "web_search"
