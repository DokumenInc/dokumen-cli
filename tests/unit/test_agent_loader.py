"""Tests for agent_loader module — loads agent config from backend API."""

import json
import os
from unittest.mock import patch, MagicMock, PropertyMock

import pytest


def _make_mock_session(auth_response=None, agent_response=None, agent_error=None, auth_error=None):
    """Create a mock requests.Session with auth + agent responses."""
    mock_session = MagicMock()

    # Auth response
    mock_auth_resp = MagicMock()
    if auth_error:
        mock_auth_resp.raise_for_status.side_effect = auth_error
    else:
        mock_auth_resp.raise_for_status = MagicMock()
        mock_auth_resp.json.return_value = auth_response or {
            "valid": True,
            "user": {"username": "test-user"},
        }
    mock_session.post.return_value = mock_auth_resp

    # Agent response
    mock_agent_resp = MagicMock()
    if agent_error:
        mock_agent_resp.raise_for_status.side_effect = agent_error
    else:
        mock_agent_resp.raise_for_status = MagicMock()
        mock_agent_resp.json.return_value = agent_response or {}
    mock_session.get.return_value = mock_agent_resp

    return mock_session


class TestLoadAgentConfig:
    """Tests for load_agent_config function."""

    def test_returns_none_when_no_env_var(self):
        """Returns None when DOKUMEN_AGENT_ID is not set."""
        from dokumen.agent_loader import load_agent_config

        with patch.dict(os.environ, {}, clear=True):
            result = load_agent_config()

        assert result is None

    def test_returns_none_when_agent_id_empty(self):
        """Returns None when DOKUMEN_AGENT_ID is empty string."""
        from dokumen.agent_loader import load_agent_config

        with patch.dict(os.environ, {"DOKUMEN_AGENT_ID": ""}):
            result = load_agent_config()

        assert result is None

    def test_returns_none_when_no_mcp_endpoint(self):
        """Returns None when DOKUMEN_MCP_ENDPOINT is not set."""
        from dokumen.agent_loader import load_agent_config

        env = {"DOKUMEN_AGENT_ID": "abc-123"}
        with patch.dict(os.environ, env, clear=True):
            result = load_agent_config()

        assert result is None

    def test_fetches_agent_from_api(self):
        """Fetches agent config via session auth and returns tools."""
        from dokumen.agent_loader import load_agent_config

        agent_response = {
            "id": "33a9ce22-2ccb-4514-a9f0-9ac7fe4a16e6",
            "name": "doc-validator",
            "tools": ["read_file", "list_files", "search_files", "glob_files"],
            "model": "claude-sonnet-4-5-20250929",
            "max_steps": 15,
            "system_prompt": "You are a doc validator.",
            "skill_count": 0,
        }

        mock_session = _make_mock_session(agent_response=agent_response)

        env = {
            "DOKUMEN_AGENT_ID": "33a9ce22-2ccb-4514-a9f0-9ac7fe4a16e6",
            "DOKUMEN_MCP_ENDPOINT": "https://staging-api.dokumen.app/api/mcp/stream/mcp",
            "GITLAB_TOKEN": "glpat-test-token",
        }

        with patch.dict(os.environ, env, clear=True), \
             patch("dokumen.agent_loader.requests.Session", return_value=mock_session):
            result = load_agent_config()

        assert result is not None
        assert result["agent_id"] == "33a9ce22-2ccb-4514-a9f0-9ac7fe4a16e6"
        assert result["name"] == "doc-validator"
        assert result["tools"] == ["read_file", "list_files", "search_files", "glob_files"]

        # Verify auth was called first
        mock_session.post.assert_called_once()
        auth_url = mock_session.post.call_args[0][0]
        assert auth_url == "https://staging-api.dokumen.app/api/auth/validate"

        # Verify correct agent API URL
        mock_session.get.assert_called_once()
        call_url = mock_session.get.call_args[0][0]
        assert call_url == "https://staging-api.dokumen.app/api/agents/33a9ce22-2ccb-4514-a9f0-9ac7fe4a16e6"

    def test_derives_api_url_from_mcp_endpoint(self):
        """Correctly derives API base URL from MCP endpoint."""
        from dokumen.agent_loader import _derive_api_base_url

        # Standard staging
        assert _derive_api_base_url("https://staging-api.dokumen.app/api/mcp/stream/mcp") == \
            "https://staging-api.dokumen.app/api"

        # Production
        assert _derive_api_base_url("https://api.dokumen.app/api/mcp/stream/mcp") == \
            "https://api.dokumen.app/api"

        # Demo
        assert _derive_api_base_url("https://demo-api.dokumen.io/api/mcp/stream/mcp") == \
            "https://demo-api.dokumen.io/api"

    def test_returns_none_on_api_error(self):
        """Returns None when agent API returns an error (non-blocking)."""
        from dokumen.agent_loader import load_agent_config

        mock_session = _make_mock_session(
            agent_error=Exception("404 Not Found"),
        )

        env = {
            "DOKUMEN_AGENT_ID": "bad-id",
            "DOKUMEN_MCP_ENDPOINT": "https://staging-api.dokumen.app/api/mcp/stream/mcp",
            "GITLAB_TOKEN": "glpat-test-token",
        }

        with patch.dict(os.environ, env, clear=True), \
             patch("dokumen.agent_loader.requests.Session", return_value=mock_session):
            result = load_agent_config()

        assert result is None

    def test_returns_none_on_auth_error(self):
        """Returns None when auth/validate fails (non-blocking)."""
        from dokumen.agent_loader import load_agent_config

        mock_session = _make_mock_session(
            auth_error=Exception("401 Unauthorized"),
        )

        env = {
            "DOKUMEN_AGENT_ID": "abc-123",
            "DOKUMEN_MCP_ENDPOINT": "https://staging-api.dokumen.app/api/mcp/stream/mcp",
            "GITLAB_TOKEN": "glpat-test-token",
        }

        with patch.dict(os.environ, env, clear=True), \
             patch("dokumen.agent_loader.requests.Session", return_value=mock_session):
            result = load_agent_config()

        assert result is None

    def test_returns_none_on_network_error(self):
        """Returns None when network request fails (non-blocking)."""
        from dokumen.agent_loader import load_agent_config
        import requests

        mock_session = MagicMock()
        mock_session.post.side_effect = requests.ConnectionError("timeout")

        env = {
            "DOKUMEN_AGENT_ID": "abc-123",
            "DOKUMEN_MCP_ENDPOINT": "https://staging-api.dokumen.app/api/mcp/stream/mcp",
            "GITLAB_TOKEN": "glpat-test-token",
        }

        with patch.dict(os.environ, env, clear=True), \
             patch("dokumen.agent_loader.requests.Session", return_value=mock_session):
            result = load_agent_config()

        assert result is None

    def test_handles_empty_tools_list(self):
        """Handles agent with no tools configured."""
        from dokumen.agent_loader import load_agent_config

        agent_response = {
            "id": "abc-123",
            "name": "empty-agent",
            "tools": [],
            "model": None,
            "max_steps": 25,
        }

        mock_session = _make_mock_session(agent_response=agent_response)

        env = {
            "DOKUMEN_AGENT_ID": "abc-123",
            "DOKUMEN_MCP_ENDPOINT": "https://staging-api.dokumen.app/api/mcp/stream/mcp",
            "GITLAB_TOKEN": "glpat-test-token",
        }

        with patch.dict(os.environ, env, clear=True), \
             patch("dokumen.agent_loader.requests.Session", return_value=mock_session):
            result = load_agent_config()

        assert result is not None
        assert result["tools"] == []

    def test_returns_none_when_no_gitlab_token(self):
        """Returns None when GITLAB_TOKEN is not set."""
        from dokumen.agent_loader import load_agent_config

        env = {
            "DOKUMEN_AGENT_ID": "abc-123",
            "DOKUMEN_MCP_ENDPOINT": "https://staging-api.dokumen.app/api/mcp/stream/mcp",
        }

        with patch.dict(os.environ, env, clear=True):
            result = load_agent_config()

        assert result is None

    def test_passes_pat_in_auth_body(self):
        """Passes GITLAB_TOKEN as pat in auth/validate request body."""
        from dokumen.agent_loader import load_agent_config

        mock_session = _make_mock_session(
            agent_response={"id": "abc", "name": "test", "tools": []},
        )

        env = {
            "DOKUMEN_AGENT_ID": "abc",
            "DOKUMEN_MCP_ENDPOINT": "https://staging-api.dokumen.app/api/mcp/stream/mcp",
            "GITLAB_TOKEN": "glpat-my-secret-token",
        }

        with patch.dict(os.environ, env, clear=True), \
             patch("dokumen.agent_loader.requests.Session", return_value=mock_session):
            load_agent_config()

        # Verify PAT was sent in the auth body
        auth_call = mock_session.post.call_args
        assert auth_call[1]["json"] == {"pat": "glpat-my-secret-token"}


class TestMapToolNames:
    """Tests for _map_tool_names — DB→CLI name translation."""

    def test_maps_known_names(self):
        """Known DB tool names are mapped to CLI equivalents."""
        from dokumen.agent_loader import _map_tool_names

        result = _map_tool_names(["list_files", "search_files", "glob_files"])
        assert result == ["list_directory", "search_file_content", "glob"]

    def test_passes_through_unknown_names(self):
        """Unknown tool names are passed through unchanged."""
        from dokumen.agent_loader import _map_tool_names

        result = _map_tool_names(["read_file", "run_shell_command", "custom_tool"])
        assert result == ["read_file", "run_shell_command", "custom_tool"]

    def test_empty_list(self):
        """Empty list returns empty list."""
        from dokumen.agent_loader import _map_tool_names

        assert _map_tool_names([]) == []


class TestGetAgentTools:
    """Tests for get_agent_tools — convenience function for loader integration."""

    def setup_method(self):
        """Reset cache before each test."""
        from dokumen.agent_loader import _reset_cache
        _reset_cache()

    def test_returns_empty_list_when_no_agent(self):
        """Returns empty list when no agent config available."""
        from dokumen.agent_loader import get_agent_tools

        with patch("dokumen.agent_loader.load_agent_config", return_value=None):
            tools = get_agent_tools()

        assert tools == []

    def test_returns_tools_from_agent_config(self):
        """Returns tool list from agent config, mapped to CLI names."""
        from dokumen.agent_loader import get_agent_tools

        config = {
            "agent_id": "abc",
            "name": "doc-validator",
            "tools": ["read_file", "list_files"],
        }

        with patch("dokumen.agent_loader.load_agent_config", return_value=config):
            tools = get_agent_tools()

        # list_files maps to list_directory
        assert tools == ["read_file", "list_directory"]

    def test_caches_result(self):
        """Caches result after first call to avoid repeated API calls."""
        from dokumen.agent_loader import get_agent_tools, _reset_cache

        _reset_cache()

        config = {
            "agent_id": "abc",
            "name": "doc-validator",
            "tools": ["read_file"],
        }

        with patch("dokumen.agent_loader.load_agent_config", return_value=config) as mock_load:
            tools1 = get_agent_tools()
            tools2 = get_agent_tools()

        assert tools1 == ["read_file"]  # read_file maps to itself
        assert tools2 == ["read_file"]
        mock_load.assert_called_once()  # Only one API call

        _reset_cache()


class TestGetAgentSkills:
    """Tests for get_agent_skills — fetches skills assigned to agent from API."""

    def setup_method(self):
        """Reset cache before each test."""
        from dokumen.agent_loader import _reset_cache
        _reset_cache()

    def test_returns_empty_list_when_no_agent(self):
        """Returns empty list when no agent config available."""
        from dokumen.agent_loader import get_agent_skills

        with patch("dokumen.agent_loader.load_agent_config", return_value=None):
            skills = get_agent_skills()

        assert skills == []

    def test_returns_skills_from_api(self):
        """Fetches and returns skills from the agent skills endpoint."""
        from dokumen.agent_loader import get_agent_skills, _reset_cache

        _reset_cache()

        config = {
            "agent_id": "abc-123",
            "name": "doc-validator",
            "tools": ["read_file"],
        }

        mock_session = MagicMock()
        mock_skills_resp = MagicMock()
        mock_skills_resp.raise_for_status = MagicMock()
        mock_skills_resp.json.return_value = {
            "skills": [
                {"name": "review-docs", "content": "# Review\nCheck docs...", "description": "Review docs"},
                {"name": "summarize", "content": "# Summarize\nSummarize...", "description": "Summarize docs"},
            ],
            "total": 2,
        }
        mock_session.get.return_value = mock_skills_resp

        env = {
            "DOKUMEN_AGENT_ID": "abc-123",
            "DOKUMEN_MCP_ENDPOINT": "https://api.dokumen.app/api/mcp/stream/mcp",
            "GITLAB_TOKEN": "glpat-test",
        }

        with patch.dict(os.environ, env, clear=True), \
             patch("dokumen.agent_loader.load_agent_config", return_value=config), \
             patch("dokumen.agent_loader._get_authenticated_session", return_value=mock_session):
            skills = get_agent_skills()

        assert len(skills) == 2
        assert skills[0]["name"] == "review-docs"
        assert skills[0]["content"] == "# Review\nCheck docs..."
        assert skills[1]["name"] == "summarize"

        _reset_cache()

    def test_returns_empty_list_when_api_fails(self):
        """Returns empty list when skills API call fails."""
        from dokumen.agent_loader import get_agent_skills, _reset_cache

        _reset_cache()

        config = {
            "agent_id": "abc-123",
            "name": "doc-validator",
            "tools": [],
        }

        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Connection error")

        env = {
            "DOKUMEN_AGENT_ID": "abc-123",
            "DOKUMEN_MCP_ENDPOINT": "https://api.dokumen.app/api/mcp/stream/mcp",
            "GITLAB_TOKEN": "glpat-test",
        }

        with patch.dict(os.environ, env, clear=True), \
             patch("dokumen.agent_loader.load_agent_config", return_value=config), \
             patch("dokumen.agent_loader._get_authenticated_session", return_value=mock_session):
            skills = get_agent_skills()

        assert skills == []

        _reset_cache()

    def test_returns_empty_list_when_no_session(self):
        """Returns empty list when auth session cannot be created."""
        from dokumen.agent_loader import get_agent_skills, _reset_cache

        _reset_cache()

        config = {
            "agent_id": "abc-123",
            "name": "doc-validator",
            "tools": [],
        }

        with patch("dokumen.agent_loader.load_agent_config", return_value=config), \
             patch("dokumen.agent_loader._get_authenticated_session", return_value=None):
            skills = get_agent_skills()

        assert skills == []

        _reset_cache()

    def test_caches_result(self):
        """Caches skills after first call to avoid repeated API calls."""
        from dokumen.agent_loader import get_agent_skills, _reset_cache

        _reset_cache()

        config = {
            "agent_id": "abc-123",
            "name": "doc-validator",
            "tools": [],
        }

        mock_session = MagicMock()
        mock_skills_resp = MagicMock()
        mock_skills_resp.raise_for_status = MagicMock()
        mock_skills_resp.json.return_value = {
            "skills": [{"name": "s1", "content": "c1", "description": "d1"}],
            "total": 1,
        }
        mock_session.get.return_value = mock_skills_resp

        env = {
            "DOKUMEN_AGENT_ID": "abc-123",
            "DOKUMEN_MCP_ENDPOINT": "https://api.dokumen.app/api/mcp/stream/mcp",
            "GITLAB_TOKEN": "glpat-test",
        }

        with patch.dict(os.environ, env, clear=True), \
             patch("dokumen.agent_loader.load_agent_config", return_value=config), \
             patch("dokumen.agent_loader._get_authenticated_session", return_value=mock_session):
            skills1 = get_agent_skills()
            skills2 = get_agent_skills()

        assert len(skills1) == 1
        assert len(skills2) == 1
        # Session.get should only be called once (cached)
        mock_session.get.assert_called_once()

        _reset_cache()
