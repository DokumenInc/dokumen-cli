"""Tests for project resolver."""

import pytest
from unittest.mock import patch, MagicMock

from dokumen.workspace.resolver import resolve_project


class TestResolveProject:
    """Test project resolution via API."""

    def test_resolve_success(self):
        """Resolves project name to git proxy URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "project_id": 10,
            "username": "alice",
            "git_proxy_url": "https://api.dokumen.app/api/git-proxy/10",
            "gitlab_url": "https://gitlab.dokumen.app",
        }

        with patch("dokumen.workspace.resolver.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response

            result = resolve_project(
                project="my-project",
                pat="glpat-abc123",
                api_url="https://api.dokumen.app",
            )

        assert result["project_id"] == 10
        assert result["git_proxy_url"] == "https://api.dokumen.app/api/git-proxy/10"

    def test_resolve_invalid_pat(self):
        """Invalid PAT raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"message": "Invalid PAT"}

        with patch("dokumen.workspace.resolver.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response

            with pytest.raises(RuntimeError, match="Authentication failed"):
                resolve_project(
                    project="my-project",
                    pat="glpat-invalid",
                    api_url="https://api.dokumen.app",
                )

    def test_resolve_network_error(self):
        """Network error raises RuntimeError."""
        with patch("dokumen.workspace.resolver.httpx") as mock_httpx:
            mock_httpx.post.side_effect = Exception("Connection refused")

            with pytest.raises(RuntimeError, match="resolve project"):
                resolve_project(
                    project="my-project",
                    pat="glpat-abc123",
                    api_url="https://api.dokumen.app",
                )
