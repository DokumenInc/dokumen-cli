"""Tests for git remote helper protocol."""

import io
import os
import stat
import subprocess
import sys

import pytest
from unittest.mock import patch, MagicMock

from dokumen.workspace.remote_helper import (
    handle_capabilities,
    handle_connect,
    main,
    _create_askpass_script,
    _ensure_filter_configured,
    FILTER_CONFIG,
)


class TestCapabilities:
    """Test capabilities response."""

    def test_capabilities_response(self):
        """Returns 'connect' capability."""
        output = io.BytesIO()
        handle_capabilities(output)
        result = output.getvalue()
        assert b"connect\n" in result
        assert result.endswith(b"\n")


class TestConnect:
    """Test connect command."""

    def test_connect_resolves_and_builds_url(self):
        """Connect resolves project and returns proxy URL."""
        mock_resolve = {
            "project_id": 10,
            "username": "alice",
            "git_proxy_url": "https://api.dokumen.app/api/git-proxy/10",
            "gitlab_url": "https://gitlab.dokumen.app",
        }

        with (
            patch("dokumen.workspace.remote_helper.get_pat", return_value="glpat-abc123"),
            patch("dokumen.workspace.remote_helper.resolve_project", return_value=mock_resolve),
        ):
            url, pat = handle_connect("my-project", "https://api.dokumen.app")

        assert url == "https://api.dokumen.app/api/git-proxy/10"
        assert pat == "glpat-abc123"

    def test_connect_no_pat_error(self):
        """Connect without PAT raises RuntimeError."""
        with patch(
            "dokumen.workspace.remote_helper.get_pat",
            side_effect=RuntimeError("DOKUMEN_PAT not set"),
        ):
            with pytest.raises(RuntimeError, match="DOKUMEN_PAT"):
                handle_connect("my-project", "https://api.dokumen.app")


class TestCreateAskpassScript:
    """Test askpass script creation."""

    def test_creates_executable_script(self):
        """Creates a temporary script that echoes the PAT."""
        path = _create_askpass_script("glpat-test123")
        try:
            assert os.path.exists(path)
            # Check it's executable
            mode = os.stat(path).st_mode
            assert mode & stat.S_IXUSR
            # Check content
            with open(path) as f:
                content = f.read()
            assert "glpat-test123" in content
            assert content.startswith("#!/bin/sh")
        finally:
            os.unlink(path)


def _mock_subprocess_configured():
    """Return a mock subprocess.run that reports all filter config as current."""

    def side_effect(cmd, **kwargs):
        result = MagicMock()
        if cmd[:3] == ["git", "config", "--global"]:
            key = cmd[3]
            for cfg_key, cfg_value in FILTER_CONFIG:
                if key == cfg_key:
                    result.returncode = 0
                    result.stdout = cfg_value + "\n"
                    return result
        result.returncode = 1
        result.stdout = ""
        return result

    return side_effect


def _mock_subprocess_not_configured():
    """Return a mock subprocess.run that reports filter config as missing."""

    def side_effect(cmd, **kwargs):
        result = MagicMock()
        if "check" in kwargs and kwargs["check"]:
            return result  # config write succeeds
        result.returncode = 1
        result.stdout = ""
        return result

    return side_effect


class TestMain:
    """Test main entry point."""

    def test_too_few_args(self):
        """Exits with error when fewer than 3 args."""
        with patch.object(sys, "argv", ["git-remote-dokumen"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_invalid_url(self):
        """Exits with error for invalid URL."""
        with patch.object(sys, "argv", ["git-remote-dokumen", "origin", "invalid://bad"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_resolves_and_execs_into_git_remote_https(self):
        """Resolves project and execs into git-remote-https with authed proxy URL."""
        mock_resolve = {
            "project_id": 10,
            "username": "alice",
            "git_proxy_url": "https://api.dokumen.app/api/git-proxy/10",
            "gitlab_url": "https://gitlab.dokumen.app",
        }

        with (
            patch.object(sys, "argv", ["git-remote-dokumen", "origin", "dokumen://my-project"]),
            patch("dokumen.workspace.remote_helper.get_pat", return_value="glpat-abc123"),
            patch("dokumen.workspace.remote_helper.resolve_project", return_value=mock_resolve),
            patch("dokumen.workspace.remote_helper.subprocess.run", side_effect=_mock_subprocess_configured()),
            patch("os.execvp") as mock_execvp,
        ):
            main()

        mock_execvp.assert_called_once_with(
            "git-remote-https",
            ["git-remote-https", "origin", "https://oauth2:glpat-abc123@api.dokumen.app/api/git-proxy/10"],
        )

    def test_sets_terminal_prompt_env(self):
        """Sets GIT_TERMINAL_PROMPT env var before exec."""
        mock_resolve = {
            "project_id": 10,
            "username": "alice",
            "git_proxy_url": "https://api.dokumen.app/api/git-proxy/10",
            "gitlab_url": "https://gitlab.dokumen.app",
        }

        with (
            patch.object(sys, "argv", ["git-remote-dokumen", "origin", "dokumen://my-project"]),
            patch("dokumen.workspace.remote_helper.get_pat", return_value="glpat-abc123"),
            patch("dokumen.workspace.remote_helper.resolve_project", return_value=mock_resolve),
            patch("dokumen.workspace.remote_helper.subprocess.run", side_effect=_mock_subprocess_configured()),
            patch("os.execvp"),
        ):
            main()

        assert os.environ.get("GIT_TERMINAL_PROMPT") == "0"

    def test_resolution_error_exits(self):
        """Exits with error when project resolution fails."""
        with (
            patch.object(sys, "argv", ["git-remote-dokumen", "origin", "dokumen://my-project"]),
            patch("dokumen.workspace.remote_helper.get_pat", return_value="glpat-abc123"),
            patch(
                "dokumen.workspace.remote_helper.resolve_project",
                side_effect=RuntimeError("Authentication failed"),
            ),
        ):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_pat_missing_exits(self):
        """Exits with error when PAT is not set."""
        with (
            patch.object(sys, "argv", ["git-remote-dokumen", "origin", "dokumen://my-project"]),
            patch(
                "dokumen.workspace.remote_helper.get_pat",
                side_effect=RuntimeError("DOKUMEN_PAT not set"),
            ),
        ):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_api_url_override_from_env(self):
        """Uses DOKUMEN_API_URL env var when set."""
        mock_resolve = {
            "project_id": 10,
            "username": "alice",
            "git_proxy_url": "https://custom-api.example.com/api/git-proxy/10",
            "gitlab_url": "https://gitlab.example.com",
        }

        with (
            patch.object(sys, "argv", ["git-remote-dokumen", "origin", "dokumen://my-project"]),
            patch.dict(os.environ, {"DOKUMEN_API_URL": "https://custom-api.example.com"}),
            patch("dokumen.workspace.remote_helper.get_pat", return_value="glpat-abc123"),
            patch("dokumen.workspace.remote_helper.resolve_project", return_value=mock_resolve) as mock_res,
            patch("dokumen.workspace.remote_helper.subprocess.run", side_effect=_mock_subprocess_configured()),
            patch("os.execvp"),
        ):
            main()

        mock_res.assert_called_once_with(
            project="my-project", pat="glpat-abc123", api_url="https://custom-api.example.com"
        )

    def test_api_host_from_url(self):
        """Uses api_host from dokumen:// URL when present."""
        mock_resolve = {
            "project_id": 10,
            "username": "alice",
            "git_proxy_url": "https://custom.host/api/git-proxy/10",
            "gitlab_url": "https://gitlab.custom.host",
        }

        with (
            patch.object(
                sys, "argv",
                ["git-remote-dokumen", "origin", "dokumen://custom.host/my-project"],
            ),
            patch("dokumen.workspace.remote_helper.get_pat", return_value="glpat-abc123"),
            patch("dokumen.workspace.remote_helper.resolve_project", return_value=mock_resolve) as mock_res,
            patch("dokumen.workspace.remote_helper.subprocess.run", side_effect=_mock_subprocess_configured()),
            patch("os.execvp"),
        ):
            main()

        mock_res.assert_called_once_with(
            project="my-project", pat="glpat-abc123", api_url="https://custom.host"
        )


class TestEnsureFilterConfigured:
    """Test _ensure_filter_configured()."""

    def test_filter_not_configured_sets_all(self):
        """When filter is not configured, writes clean, smudge, required."""
        with patch(
            "dokumen.workspace.remote_helper.subprocess.run",
            side_effect=_mock_subprocess_not_configured(),
        ) as mock_run:
            _ensure_filter_configured()

        # First call is the check (returns non-zero), then 3 config writes
        write_calls = [
            c for c in mock_run.call_args_list
            if "--replace-all" in c[0][0]
        ]
        assert len(write_calls) == 3

        # Verify the keys written
        written_keys = [c[0][0][4] for c in write_calls]
        assert written_keys == [
            "filter.dokumen.clean",
            "filter.dokumen.smudge",
            "filter.dokumen.required",
        ]

    def test_filter_already_configured_skips(self):
        """When filter is already correctly configured, no writes happen."""
        with patch(
            "dokumen.workspace.remote_helper.subprocess.run",
            side_effect=_mock_subprocess_configured(),
        ) as mock_run:
            _ensure_filter_configured()

        # Only check calls, no write calls
        write_calls = [
            c for c in mock_run.call_args_list
            if "--replace-all" in c[0][0]
        ]
        assert len(write_calls) == 0

    def test_filter_stale_smudge_updates_all(self):
        """When clean is correct but smudge is stale, rewrites all config."""
        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            result = MagicMock()

            if "check" in kwargs and kwargs["check"]:
                return result  # config write succeeds

            call_count += 1
            key = cmd[3] if len(cmd) > 3 else None

            if key == "filter.dokumen.clean":
                result.returncode = 0
                result.stdout = "dokumen-filter --clean\n"
            elif key == "filter.dokumen.smudge":
                # Stale value
                result.returncode = 0
                result.stdout = "old-filter --smudge\n"
            else:
                result.returncode = 1
                result.stdout = ""
            return result

        with patch(
            "dokumen.workspace.remote_helper.subprocess.run",
            side_effect=side_effect,
        ) as mock_run:
            _ensure_filter_configured()

        write_calls = [
            c for c in mock_run.call_args_list
            if "--replace-all" in c[0][0]
        ]
        assert len(write_calls) == 3

    def test_required_written_last(self):
        """Verify call order: clean, smudge, required=true."""
        with patch(
            "dokumen.workspace.remote_helper.subprocess.run",
            side_effect=_mock_subprocess_not_configured(),
        ) as mock_run:
            _ensure_filter_configured()

        write_calls = [
            c for c in mock_run.call_args_list
            if "--replace-all" in c[0][0]
        ]
        written_keys = [c[0][0][4] for c in write_calls]
        assert written_keys[-1] == "filter.dokumen.required"

    def test_config_write_failure_exits(self):
        """subprocess.run raising CalledProcessError triggers sys.exit(1)."""

        def side_effect(cmd, **kwargs):
            result = MagicMock()
            if "check" in kwargs and kwargs["check"]:
                raise subprocess.CalledProcessError(1, cmd)
            result.returncode = 1
            result.stdout = ""
            return result

        with patch(
            "dokumen.workspace.remote_helper.subprocess.run",
            side_effect=side_effect,
        ):
            with pytest.raises(SystemExit) as exc:
                _ensure_filter_configured()
            assert exc.value.code == 1
