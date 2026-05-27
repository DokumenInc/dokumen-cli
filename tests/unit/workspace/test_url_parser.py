"""Tests for dokumen:// URL parser."""

import pytest

from dokumen.workspace.url_parser import DokumenURL, parse_url


class TestParseURL:
    """Test dokumen:// URL parsing."""

    def test_parse_simple_url(self):
        """dokumen://my-project parses to project name only."""
        result = parse_url("dokumen://my-project")
        assert isinstance(result, DokumenURL)
        assert result.project == "my-project"
        assert result.api_host is None

    def test_parse_url_with_host(self):
        """dokumen://api.dokumen.app/my-project parses host + project."""
        result = parse_url("dokumen://api.dokumen.app/my-project")
        assert result.project == "my-project"
        assert result.api_host == "api.dokumen.app"

    def test_parse_url_with_host_and_port(self):
        """dokumen://localhost:8000/my-project parses host:port + project."""
        result = parse_url("dokumen://localhost:8000/my-project")
        assert result.project == "my-project"
        assert result.api_host == "localhost:8000"

    def test_parse_invalid_url_no_scheme(self):
        """URL without dokumen:// scheme raises ValueError."""
        with pytest.raises(ValueError, match="must start with dokumen://"):
            parse_url("https://example.com/project")

    def test_parse_invalid_url_empty_project(self):
        """dokumen:// with no project raises ValueError."""
        with pytest.raises(ValueError, match="project name"):
            parse_url("dokumen://")

    def test_proxy_url_simple(self):
        """Simple URL generates proxy URL using default host."""
        result = parse_url("dokumen://my-project")
        proxy_url = result.proxy_base_url("https://api.dokumen.app")
        assert proxy_url == "https://api.dokumen.app/api/git-proxy"

    def test_proxy_url_with_explicit_host(self):
        """URL with host uses that host for proxy URL."""
        result = parse_url("dokumen://staging-api.dokumen.app/my-project")
        proxy_url = result.proxy_base_url("https://api.dokumen.app")
        assert proxy_url == "https://staging-api.dokumen.app/api/git-proxy"
