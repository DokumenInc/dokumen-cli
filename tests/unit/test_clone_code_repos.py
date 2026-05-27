"""
Unit tests for clone_code_repos function in loader.py.

TDD: Written FIRST before implementation.
Tests validate that code repos are correctly cloned from GitLab
and code_repos_config is properly built and wired into load_all_scaffolds.
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo_config(**kwargs):
    """Build a CodeRepoConfig with sensible defaults."""
    from dokumen.config import CodeRepoConfig

    defaults = {
        "name": "product",
        "gitlab_project_id": 1,
        "gitlab_url": "https://gitlab.example.com",
        "branch": "main",
    }
    defaults.update(kwargs)
    return CodeRepoConfig(**defaults)


def _make_api_response(http_url="https://gitlab.example.com/ns/product.git"):
    """Build a fake urlopen context manager that returns project info."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {"http_url_to_repo": http_url}
    ).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# TestCloneCodeReposEmpty
# ---------------------------------------------------------------------------


class TestCloneCodeReposEmpty:
    """Edge cases for empty / degenerate inputs."""

    def test_empty_repos_list_returns_empty(self):
        """Returns empty list immediately when no repos configured."""
        from dokumen.loader import clone_code_repos

        result = clone_code_repos([], token="tok")
        assert result == []

    def test_no_gitlab_url_and_no_fallback_skips_repo(self, tmp_path):
        """Skips repo (returns []) when no gitlab_url and no fallback."""
        from dokumen.config import CodeRepoConfig
        from dokumen.loader import clone_code_repos

        repo = CodeRepoConfig(name="product", gitlab_project_id=42)
        # gitlab_url is None, no fallback

        with patch("urllib.request.urlopen") as mock_urlopen:
            result = clone_code_repos([repo], token="tok", cache_dir=str(tmp_path))

        assert result == []
        mock_urlopen.assert_not_called()

    def test_uses_fallback_url_when_repo_has_none(self, tmp_path):
        """Uses gitlab_url_fallback when repo.gitlab_url is None."""
        from dokumen.config import CodeRepoConfig
        from dokumen.loader import clone_code_repos

        repo = CodeRepoConfig(name="product", gitlab_project_id=42)
        # No gitlab_url on repo

        captured = []

        def capture_urlopen(req, timeout=None):
            captured.append(req)
            return _make_api_response("https://fallback.example.com/ns/p.git")

        with patch("urllib.request.urlopen", side_effect=capture_urlopen), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            # Pre-create dest so it looks like clone succeeded
            dest = tmp_path / "repos" / "product"
            dest.mkdir(parents=True)
            clone_code_repos(
                [repo],
                token="tok",
                cache_dir=str(tmp_path),
                gitlab_url_fallback="https://fallback.example.com",
            )

        assert len(captured) == 1
        assert "fallback.example.com" in captured[0].full_url


# ---------------------------------------------------------------------------
# TestCloneCodeReposApiCall
# ---------------------------------------------------------------------------


class TestCloneCodeReposApiCall:
    """Tests for GitLab API interaction."""

    def test_calls_gitlab_api_for_project_info(self, tmp_path):
        """Calls {gitlab_url}/api/v4/projects/{id} with the repo's project ID."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config(gitlab_project_id=42, gitlab_url="https://gitlab.example.com")

        captured = []

        def capture_urlopen(req, timeout=None):
            captured.append(req)
            return _make_api_response()

        with patch("urllib.request.urlopen", side_effect=capture_urlopen), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            dest = tmp_path / "repos" / "product"
            dest.mkdir(parents=True)
            clone_code_repos([repo], token="glpat-tok", cache_dir=str(tmp_path))

        assert len(captured) == 1
        assert "/api/v4/projects/42" in captured[0].full_url

    def test_sets_private_token_header(self, tmp_path):
        """Sends PRIVATE-TOKEN header with the provided token."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config()
        captured = []

        def capture_urlopen(req, timeout=None):
            captured.append(req)
            return _make_api_response()

        with patch("urllib.request.urlopen", side_effect=capture_urlopen), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            dest = tmp_path / "repos" / "product"
            dest.mkdir(parents=True)
            clone_code_repos([repo], token="glpat-mytoken", cache_dir=str(tmp_path))

        req = captured[0]
        # urllib capitalises header names: 'Private-token'
        assert req.get_header("Private-token") == "glpat-mytoken"

    def test_skips_repo_if_api_fails(self, tmp_path):
        """Skips repo and returns [] when the GitLab API call raises."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config()

        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")), \
             patch("subprocess.run") as mock_run:
            result = clone_code_repos([repo], token="tok", cache_dir=str(tmp_path))

        assert result == []
        mock_run.assert_not_called()

    def test_skips_repo_if_no_http_url_in_response(self, tmp_path):
        """Skips repo when API response has no http_url_to_repo."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config()
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"id": 1}).encode()  # no http_url_to_repo
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("subprocess.run") as mock_run:
            result = clone_code_repos([repo], token="tok", cache_dir=str(tmp_path))

        assert result == []
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# TestCloneCodeReposGitOps
# ---------------------------------------------------------------------------


class TestCloneCodeReposGitOps:
    """Tests for git clone / fetch operations."""

    def test_calls_git_clone_for_new_repo(self, tmp_path):
        """Calls 'git clone --depth=1 -b {branch} {url} {dest}' for new repos."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config(branch="staging")
        # dest does NOT exist → fresh clone

        clone_cmds = []

        def capture_run(cmd, **kwargs):
            if "clone" in cmd:
                clone_cmds.append(cmd)
            return MagicMock(returncode=0)

        with patch("urllib.request.urlopen", return_value=_make_api_response()), \
             patch("subprocess.run", side_effect=capture_run):
            # After clone, create dest to simulate success
            dest = tmp_path / "repos" / "product"

            original_exists = Path.exists
            call_count = {"n": 0}

            def mock_exists(self_):
                if self_ == dest or str(self_) == str(dest):
                    call_count["n"] += 1
                    # First check (pre-clone): False; subsequent: True
                    return call_count["n"] > 1
                return original_exists(self_)

            with patch.object(Path, "exists", mock_exists):
                result = clone_code_repos([repo], token="tok", cache_dir=str(tmp_path))

        assert len(clone_cmds) == 1, f"Expected 1 clone call, got: {clone_cmds}"
        cmd = clone_cmds[0]
        assert cmd[0] == "git"
        assert "clone" in cmd
        assert "--depth=1" in cmd
        assert "staging" in cmd

    def test_includes_oauth2_token_in_clone_url(self, tmp_path):
        """Inserts 'oauth2:{token}@' into the git clone URL."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config()
        clone_urls = []

        def capture_run(cmd, **kwargs):
            if "clone" in cmd:
                # cmd: ["git", "clone", "--depth=1", "-b", branch, url, dest]
                clone_urls.append(cmd[5])  # authenticated_url arg
            return MagicMock(returncode=0)

        with patch("urllib.request.urlopen", return_value=_make_api_response(
            "https://gitlab.example.com/ns/product.git"
        )), patch("subprocess.run", side_effect=capture_run):
            dest = tmp_path / "repos" / "product"
            original_exists = Path.exists
            call_count = {"n": 0}

            def mock_exists(self_):
                if self_ == dest or str(self_) == str(dest):
                    call_count["n"] += 1
                    return call_count["n"] > 1
                return original_exists(self_)

            with patch.object(Path, "exists", mock_exists):
                clone_code_repos([repo], token="glpat-abc123", cache_dir=str(tmp_path))

        assert len(clone_urls) == 1
        assert "oauth2:glpat-abc123@" in clone_urls[0]
        assert "gitlab.example.com" in clone_urls[0]

    def test_no_token_in_url_when_token_is_none(self, tmp_path):
        """Does NOT insert oauth2 credentials when token is None."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config()
        clone_urls = []

        def capture_run(cmd, **kwargs):
            if "clone" in cmd:
                clone_urls.append(cmd[4])
            return MagicMock(returncode=0)

        with patch("urllib.request.urlopen", return_value=_make_api_response()), \
             patch("subprocess.run", side_effect=capture_run):
            dest = tmp_path / "repos" / "product"
            original_exists = Path.exists
            call_count = {"n": 0}

            def mock_exists(self_):
                if self_ == dest or str(self_) == str(dest):
                    call_count["n"] += 1
                    return call_count["n"] > 1
                return original_exists(self_)

            with patch.object(Path, "exists", mock_exists):
                clone_code_repos([repo], token=None, cache_dir=str(tmp_path))

        if clone_urls:  # only check if clone was attempted
            assert "oauth2" not in clone_urls[0]

    def test_calls_git_fetch_for_existing_repo(self, tmp_path):
        """Calls 'git fetch' (not clone) when dest already exists."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config(branch="staging")

        # Pre-create the dest directory
        dest = tmp_path / "repos" / "product"
        dest.mkdir(parents=True)

        run_cmds = []

        def capture_run(cmd, **kwargs):
            run_cmds.append(cmd)
            return MagicMock(returncode=0)

        with patch("urllib.request.urlopen", return_value=_make_api_response()), \
             patch("subprocess.run", side_effect=capture_run):
            clone_code_repos([repo], token="tok", cache_dir=str(tmp_path))

        flat_cmds = [" ".join(cmd) for cmd in run_cmds]
        assert any("fetch" in c for c in flat_cmds), f"No fetch call found: {flat_cmds}"
        assert not any("clone" in c for c in flat_cmds), f"Unexpected clone: {flat_cmds}"

    def test_skips_failed_clone(self, tmp_path):
        """Returns [] when git clone exits with non-zero returncode."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config()

        with patch("urllib.request.urlopen", return_value=_make_api_response()), \
             patch("subprocess.run", return_value=MagicMock(returncode=128, stderr=b"auth fail")), \
             patch.object(Path, "exists", return_value=False):
            result = clone_code_repos([repo], token="tok", cache_dir=str(tmp_path))

        assert result == []

    def test_returns_correct_config_dict_structure(self, tmp_path):
        """Returned dict has name, base_dir, include_patterns, exclude_patterns."""
        from dokumen.loader import clone_code_repos

        repo = _make_repo_config(
            name="backend",
            paths_include=["src/**/*.py"],
            paths_exclude=["tests/**"],
        )

        # Pre-create dest so it looks like clone succeeded
        dest = tmp_path / "repos" / "backend"
        dest.mkdir(parents=True)

        with patch("urllib.request.urlopen", return_value=_make_api_response()), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = clone_code_repos([repo], token="tok", cache_dir=str(tmp_path))

        assert len(result) == 1
        cfg = result[0]
        assert cfg["name"] == "backend"
        assert "backend" in cfg["base_dir"]
        assert cfg["include_patterns"] == ["src/**/*.py"]
        assert cfg["exclude_patterns"] == ["tests/**"]

    def test_multiple_repos_all_cloned(self, tmp_path):
        """Handles multiple repos and returns config for each."""
        from dokumen.config import CodeRepoConfig
        from dokumen.loader import clone_code_repos

        repos = [
            CodeRepoConfig(name="backend", gitlab_project_id=1, gitlab_url="https://gl.example.com"),
            CodeRepoConfig(name="frontend", gitlab_project_id=2, gitlab_url="https://gl.example.com"),
        ]

        # Pre-create both dest dirs
        for r in repos:
            (tmp_path / "repos" / r.name).mkdir(parents=True)

        def api_response(req, timeout=None):
            project_id = int(req.full_url.split("/projects/")[1].split("/")[0])
            http_url = f"https://gl.example.com/ns/repo{project_id}.git"
            return _make_api_response(http_url)

        with patch("urllib.request.urlopen", side_effect=api_response), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = clone_code_repos(repos, token="tok", cache_dir=str(tmp_path))

        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"backend", "frontend"}


# ---------------------------------------------------------------------------
# TestLoadAllScaffoldsCodeRepos — integration wiring tests
# ---------------------------------------------------------------------------


class TestLoadAllScaffoldsCodeRepos:
    """Tests that load_all_scaffolds wires clone_code_repos → load_scaffold."""

    def _write_scaffold(self, tmp_path, name="my-test"):
        scaffold = {
            "name": name,
            "files": [],
            "executor": {
                "system_prompt": "@prompts/documentation-validation.txt",
                "user_prompt": "Test",
                "tools": ["read_file"],
            },
            "judges": [{"name": "check", "system_prompt": "Check."}],
        }
        (tmp_path / "tests").mkdir(exist_ok=True)
        (tmp_path / "tests" / f"{name}.test.yaml").write_text(yaml.dump(scaffold))

    def _write_config(self, tmp_path, code_repos=None):
        config = {
            "version": "1.0",
            "provider": {"name": "mock", "model": "test"},
        }
        if code_repos:
            config["code_repos"] = code_repos
        (tmp_path / "dokumen.yaml").write_text(yaml.dump(config))

    def test_passes_code_repos_config_to_load_scaffold(self, tmp_path):
        """load_all_scaffolds calls clone_code_repos and passes result to each load_scaffold."""
        self._write_scaffold(tmp_path)
        self._write_config(tmp_path, code_repos=[
            {"name": "product", "gitlab_project_id": 1, "gitlab_url": "https://gl.example.com"}
        ])

        mock_cfg = [{"name": "product", "base_dir": "/tmp/repos/product",
                     "include_patterns": [], "exclude_patterns": []}]
        load_scaffold_kwargs = []

        def mock_load_scaffold(path, *args, **kwargs):
            load_scaffold_kwargs.append(kwargs)
            return MagicMock()

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("dokumen.loader.load_scaffold", side_effect=mock_load_scaffold), \
                 patch("dokumen.loader.clone_code_repos", return_value=mock_cfg) as mock_clone:
                from dokumen.loader import load_all_scaffolds
                load_all_scaffolds(tests_dir="tests", config_path=str(tmp_path / "dokumen.yaml"))

            mock_clone.assert_called_once()
            assert len(load_scaffold_kwargs) == 1
            assert load_scaffold_kwargs[0].get("code_repos_config") == mock_cfg
        finally:
            os.chdir(original_dir)

    def test_does_not_call_clone_when_no_code_repos(self, tmp_path):
        """load_all_scaffolds does NOT call clone_code_repos when code_repos is empty."""
        self._write_scaffold(tmp_path)
        self._write_config(tmp_path)  # no code_repos

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("dokumen.loader.clone_code_repos") as mock_clone:
                from dokumen.loader import load_all_scaffolds
                load_all_scaffolds(tests_dir="tests", config_path=str(tmp_path / "dokumen.yaml"))

            mock_clone.assert_not_called()
        finally:
            os.chdir(original_dir)

    def test_code_repos_config_none_when_no_repos(self, tmp_path):
        """load_all_scaffolds passes code_repos_config=None to load_scaffold when no repos."""
        self._write_scaffold(tmp_path)
        self._write_config(tmp_path)

        load_scaffold_kwargs = []

        def mock_load_scaffold(path, *args, **kwargs):
            load_scaffold_kwargs.append(kwargs)
            return MagicMock()

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("dokumen.loader.load_scaffold", side_effect=mock_load_scaffold):
                from dokumen.loader import load_all_scaffolds
                load_all_scaffolds(tests_dir="tests", config_path=str(tmp_path / "dokumen.yaml"))

            assert len(load_scaffold_kwargs) == 1
            assert load_scaffold_kwargs[0].get("code_repos_config") is None
        finally:
            os.chdir(original_dir)
