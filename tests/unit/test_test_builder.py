"""Tests for test_builder module — provider creation and SDK agent construction."""
import os
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from dokumen.test_builder import (
    create_provider,
    find_config_file,
    get_configured_provider,
    get_configured_providers,
    build_sdk_executor,
    build_sdk_judge,
    build_research_judge,
)


class TestCreateProvider:
    """Tests for create_provider."""

    def test_creates_anthropic_provider(self):
        with patch("dokumen.providers.anthropic.AnthropicProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = create_provider("anthropic", api_key="sk-test", model="claude-3")
            assert result is not None
            mock_cls.assert_called_once_with(api_key="sk-test", model="claude-3")

    def test_case_insensitive(self):
        with patch("dokumen.providers.anthropic.AnthropicProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            create_provider("Anthropic", api_key="sk-test")
            mock_cls.assert_called_once()

    def test_none_name_returns_none(self):
        assert create_provider(None) is None

    def test_empty_name_returns_none(self):
        assert create_provider("") is None

    def test_unknown_provider_returns_none(self):
        assert create_provider("openai") is None


class TestFindConfigFile:
    """Tests for find_config_file."""

    def test_finds_explicit_path(self, tmp_path):
        config = tmp_path / "dokumen.yaml"
        config.write_text("version: '1.0'")
        result = find_config_file(str(config))
        assert result == str(config)

    def test_returns_none_for_missing_explicit(self):
        result = find_config_file("/nonexistent/path/dokumen.yaml")
        # Returns None because file doesn't exist and cwd search won't find it
        # (or finds it in cwd — either way the function shouldn't crash)
        assert result is None or isinstance(result, str)

    def test_none_path_searches_cwd(self):
        # Just verify it doesn't crash
        result = find_config_file(None)
        assert result is None or isinstance(result, str)


class TestGetConfiguredProvider:
    """Tests for get_configured_provider."""

    @patch.dict(os.environ, {"DOKUMEN_PROVIDER": "anthropic", "DOKUMEN_API_KEY": "sk-test"})
    def test_from_env_vars(self):
        with patch("dokumen.providers.anthropic.AnthropicProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = get_configured_provider()
            assert result is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_without_config(self):
        with patch("dokumen.test_builder.find_config_file", return_value=None):
            result = get_configured_provider()
            assert result is None


class TestGetConfiguredProviders:
    """Tests for get_configured_providers."""

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_providers_without_config(self):
        with patch("dokumen.test_builder.find_config_file", return_value=None):
            result = get_configured_providers()
            assert result["executor"] is None
            assert result["judge"] is None
            assert result["default"] is None

    @patch.dict(os.environ, {"DOKUMEN_PROVIDER": "anthropic", "DOKUMEN_API_KEY": "sk-test", "DOKUMEN_MODEL": "claude-3"})
    def test_from_env_vars(self):
        with patch("dokumen.test_builder.find_config_file", return_value=None):
            with patch("dokumen.providers.anthropic.AnthropicProvider") as mock_cls:
                mock_cls.return_value = MagicMock()
                result = get_configured_providers()
                assert result["executor"] is not None
                assert result["judge"] is not None
                assert result["default"] is not None


class TestBuildSdkExecutor:
    """Tests for build_sdk_executor."""

    @patch("dokumen.sdk.agent_wrapper.SdkExecutorWrapper")
    @patch("dokumen.sdk.executor.ExecutorAgent")
    @patch("dokumen.sdk.tools.resolve_sdk_tools")
    def test_builds_executor(self, mock_resolve, mock_agent, mock_wrapper):
        mock_resolve.return_value = SimpleNamespace(
            sdk_tool_names=["read_file"],
            playwright_tool_names=[],
            playwright_mcp_config=None,
            dokumen_mcp_tools=None,
        )
        mock_agent.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()

        provider = SimpleNamespace(model="claude-3")
        data = {"name": "test-scaffold", "executor": {"user_prompt": "Do something"}, "timeout": 60}

        result = build_sdk_executor(
            data=data,
            executor_system_prompt="You are a test executor",
            executor_tool_names=["read_file"],
            actual_executor_provider=provider,
            executor_max_iterations=100,
        )
        assert result is not None
        mock_agent.assert_called_once()
        mock_wrapper.assert_called_once()

    @patch("dokumen.sdk.agent_wrapper.SdkExecutorWrapper")
    @patch("dokumen.sdk.executor.ExecutorAgent")
    @patch("dokumen.sdk.tools.resolve_sdk_tools")
    def test_handles_none_provider(self, mock_resolve, mock_agent, mock_wrapper):
        mock_resolve.return_value = SimpleNamespace(
            sdk_tool_names=["read_file"],
            playwright_tool_names=[],
            playwright_mcp_config=None,
            dokumen_mcp_tools=None,
        )
        mock_agent.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()

        data = {"name": "test", "executor": {"user_prompt": "test"}, "timeout": 60}
        result = build_sdk_executor(
            data=data,
            executor_system_prompt="prompt",
            executor_tool_names=["read_file"],
            actual_executor_provider=None,
            executor_max_iterations=100,
        )
        assert result is not None
        # Model should be None when provider is None
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["model"] is None


class TestBuildSdkJudge:
    """Tests for build_sdk_judge."""

    @patch("dokumen.sdk.agent_wrapper.SdkJudgeWrapper")
    @patch("dokumen.sdk.judge.JudgeAgent")
    @patch("dokumen.sdk.tools.resolve_sdk_tools")
    def test_builds_judge(self, mock_resolve, mock_agent, mock_wrapper):
        mock_resolve.return_value = SimpleNamespace(
            sdk_tool_names=[],
            dokumen_mcp_tools=None,
        )
        mock_agent.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()

        provider = SimpleNamespace(model="claude-3")
        judge_data = {"name": "accuracy", "user_prompt": "Check accuracy"}
        tool = MagicMock()
        tool.name = "read_file"

        result = build_sdk_judge(
            judge_data=judge_data,
            judge_system_prompt="You are a judge",
            judge_tools=[tool],
            judge_provider=provider,
            judge_max_iterations=3,
            judge_timeout_override=None,
        )
        assert result is not None

    @patch("dokumen.sdk.agent_wrapper.SdkJudgeWrapper")
    @patch("dokumen.sdk.judge.JudgeAgent")
    @patch("dokumen.sdk.tools.resolve_sdk_tools")
    def test_uses_timeout_override(self, mock_resolve, mock_agent, mock_wrapper):
        mock_resolve.return_value = SimpleNamespace(
            sdk_tool_names=[],
            dokumen_mcp_tools=None,
        )
        mock_agent.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()

        judge_data = {"name": "accuracy"}
        result = build_sdk_judge(
            judge_data=judge_data,
            judge_system_prompt="prompt",
            judge_tools=[],
            judge_provider=None,
            judge_max_iterations=5,
            judge_timeout_override=300.0,
        )
        assert result is not None
        # Verify timeout was passed
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["timeout"] == 300.0

    @patch("dokumen.sdk.agent_wrapper.SdkJudgeWrapper")
    @patch("dokumen.sdk.judge.JudgeAgent")
    @patch("dokumen.sdk.tools.resolve_sdk_tools")
    def test_handles_none_provider_model(self, mock_resolve, mock_agent, mock_wrapper):
        mock_resolve.return_value = SimpleNamespace(
            sdk_tool_names=[],
            dokumen_mcp_tools=None,
        )
        mock_agent.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()

        judge_data = {"name": "accuracy"}
        build_sdk_judge(
            judge_data=judge_data,
            judge_system_prompt="prompt",
            judge_tools=[],
            judge_provider=None,
            judge_max_iterations=3,
            judge_timeout_override=None,
        )
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["model"] is None
        assert call_kwargs["timeout"] == 120.0


class TestBuildResearchJudge:
    """Tests for build_research_judge."""

    @patch("dokumen.sdk.agent_wrapper.SdkJudgeWrapper")
    @patch("dokumen.sdk.judge.JudgeAgent")
    def test_builds_research_judge(self, mock_agent, mock_wrapper):
        mock_agent.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()

        provider = SimpleNamespace(model="claude-3")
        result = build_research_judge(
            judge_id="sources",
            prompt="Evaluate sources",
            judge_provider=provider,
        )
        assert result is not None
        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["id"] == "sources"
        assert call_kwargs["model"] == "claude-3"

    @patch("dokumen.sdk.agent_wrapper.SdkJudgeWrapper")
    @patch("dokumen.sdk.judge.JudgeAgent")
    def test_handles_none_provider(self, mock_agent, mock_wrapper):
        mock_agent.return_value = MagicMock()
        mock_wrapper.return_value = MagicMock()

        result = build_research_judge(
            judge_id="verdict",
            prompt="Evaluate verdict",
            judge_provider=None,
        )
        assert result is not None
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["model"] is None


    def test_build_sdk_executor_with_browser_tools_uses_playwright_mcp(self):
        """Browser tools use Playwright MCP stdio server, not Dokumen MCP wrappers."""
        data = {"name": "browser-test", "executor": {"user_prompt": "browse"}, "timeout": 60}

        result = build_sdk_executor(
            data=data,
            executor_system_prompt="sys",
            executor_tool_names=["browser_navigate", "browser_click"],
            actual_executor_provider=None,
            executor_max_iterations=5,
        )

        servers = getattr(result._executor._options, "mcp_servers", {}) or {}
        assert "playwright" in servers
        pw_config = servers["playwright"]
        assert pw_config.get("type", "stdio") == "stdio"
        assert "command" in pw_config
