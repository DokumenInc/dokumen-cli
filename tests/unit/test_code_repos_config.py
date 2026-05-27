"""
Unit tests for code_repos configuration in dokumen.yaml.

TDD: These tests are written FIRST, before implementation.
They validate the CodeRepoConfig model and its integration into DokumenConfig.
"""
from pathlib import Path

import pytest
from pydantic import ValidationError


class TestCodeRepoConfig:
    """Tests for CodeRepoConfig model."""

    def test_code_repo_config_exists(self):
        """CodeRepoConfig class should exist in config module."""
        from dokumen.config import CodeRepoConfig

        assert CodeRepoConfig is not None

    def test_code_repo_config_required_fields(self):
        """CodeRepoConfig requires name and gitlab_project_id."""
        from dokumen.config import CodeRepoConfig

        config = CodeRepoConfig(name="my-api", gitlab_project_id=42)
        assert config.name == "my-api"
        assert config.gitlab_project_id == 42

    def test_code_repo_config_missing_name_raises(self):
        """CodeRepoConfig without name should raise ValidationError."""
        from dokumen.config import CodeRepoConfig

        with pytest.raises(ValidationError, match="name"):
            CodeRepoConfig(gitlab_project_id=42)

    def test_code_repo_config_missing_project_id_raises(self):
        """CodeRepoConfig without gitlab_project_id should raise ValidationError."""
        from dokumen.config import CodeRepoConfig

        with pytest.raises(ValidationError, match="gitlab_project_id"):
            CodeRepoConfig(name="my-api")

    def test_code_repo_config_defaults(self):
        """CodeRepoConfig should have sensible defaults for optional fields."""
        from dokumen.config import CodeRepoConfig

        config = CodeRepoConfig(name="my-api", gitlab_project_id=42)
        assert config.branch == "main"
        assert config.paths_include == []
        assert config.paths_exclude == []
        assert config.gitlab_url is None

    def test_code_repo_config_custom_values(self):
        """CodeRepoConfig should accept all custom values."""
        from dokumen.config import CodeRepoConfig

        config = CodeRepoConfig(
            name="backend-service",
            gitlab_project_id=99,
            gitlab_url="https://gitlab.example.com",
            branch="develop",
            paths_include=["src/**/*.py"],
            paths_exclude=["src/tests/**"],
        )
        assert config.name == "backend-service"
        assert config.gitlab_project_id == 99
        assert config.gitlab_url == "https://gitlab.example.com"
        assert config.branch == "develop"
        assert config.paths_include == ["src/**/*.py"]
        assert config.paths_exclude == ["src/tests/**"]

    def test_code_repo_config_empty_name_raises(self):
        """CodeRepoConfig with empty name should raise ValidationError."""
        from dokumen.config import CodeRepoConfig

        with pytest.raises(ValidationError):
            CodeRepoConfig(name="", gitlab_project_id=42)

    def test_code_repo_config_project_id_must_be_positive(self):
        """CodeRepoConfig gitlab_project_id must be > 0."""
        from dokumen.config import CodeRepoConfig

        with pytest.raises(ValidationError):
            CodeRepoConfig(name="my-api", gitlab_project_id=0)

        with pytest.raises(ValidationError):
            CodeRepoConfig(name="my-api", gitlab_project_id=-1)

    def test_code_repo_config_empty_branch_raises(self):
        """CodeRepoConfig with empty branch should raise ValidationError."""
        from dokumen.config import CodeRepoConfig

        with pytest.raises(ValidationError):
            CodeRepoConfig(name="my-api", gitlab_project_id=42, branch="")


class TestDokumenConfigCodeRepos:
    """Tests for code_repos in DokumenConfig."""

    def test_dokumen_config_without_code_repos(self):
        """DokumenConfig should work without code_repos (backward compat)."""
        from dokumen.config import DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic"),
        )
        assert config.code_repos == []

    def test_dokumen_config_with_empty_code_repos(self):
        """DokumenConfig with empty code_repos list should be valid."""
        from dokumen.config import DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic"),
            code_repos=[],
        )
        assert config.code_repos == []

    def test_dokumen_config_with_one_code_repo(self):
        """DokumenConfig with a single code repo should parse correctly."""
        from dokumen.config import CodeRepoConfig, DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic"),
            code_repos=[
                CodeRepoConfig(name="backend", gitlab_project_id=10),
            ],
        )
        assert len(config.code_repos) == 1
        assert config.code_repos[0].name == "backend"
        assert config.code_repos[0].gitlab_project_id == 10

    def test_dokumen_config_with_multiple_code_repos(self):
        """DokumenConfig with multiple code repos should parse correctly."""
        from dokumen.config import CodeRepoConfig, DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic"),
            code_repos=[
                CodeRepoConfig(name="backend", gitlab_project_id=10),
                CodeRepoConfig(name="frontend", gitlab_project_id=20),
                CodeRepoConfig(name="shared-lib", gitlab_project_id=30),
            ],
        )
        assert len(config.code_repos) == 3
        assert config.code_repos[0].name == "backend"
        assert config.code_repos[1].name == "frontend"
        assert config.code_repos[2].name == "shared-lib"

    def test_dokumen_config_duplicate_repo_names_raises(self):
        """DokumenConfig should reject duplicate code_repos names."""
        from dokumen.config import CodeRepoConfig, DokumenConfig, ProviderConfig

        with pytest.raises(ValidationError, match="Duplicate"):
            DokumenConfig(
                version="1.0",
                provider=ProviderConfig(name="anthropic"),
                code_repos=[
                    CodeRepoConfig(name="backend", gitlab_project_id=10),
                    CodeRepoConfig(name="backend", gitlab_project_id=20),
                ],
            )

    def test_load_config_parses_code_repos_from_yaml(self, tmp_path):
        """load_config should parse code_repos from a YAML file."""
        from dokumen.config import load_config

        config_content = """
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
code_repos:
  - name: backend
    gitlab_project_id: 10
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert len(config.code_repos) == 1
        assert config.code_repos[0].name == "backend"
        assert config.code_repos[0].gitlab_project_id == 10
        assert config.code_repos[0].branch == "main"

    def test_load_config_with_full_code_repos_yaml(self, tmp_path):
        """load_config should parse a full code_repos YAML example."""
        from dokumen.config import load_config

        config_content = """
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
code_repos:
  - name: backend-api
    gitlab_project_id: 42
    gitlab_url: https://gitlab.example.com
    branch: develop
    paths_include:
      - "src/**/*.py"
      - "lib/**/*.py"
    paths_exclude:
      - "src/tests/**"
  - name: frontend-app
    gitlab_project_id: 55
    branch: main
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert len(config.code_repos) == 2

        backend = config.code_repos[0]
        assert backend.name == "backend-api"
        assert backend.gitlab_project_id == 42
        assert backend.gitlab_url == "https://gitlab.example.com"
        assert backend.branch == "develop"
        assert backend.paths_include == ["src/**/*.py", "lib/**/*.py"]
        assert backend.paths_exclude == ["src/tests/**"]

        frontend = config.code_repos[1]
        assert frontend.name == "frontend-app"
        assert frontend.gitlab_project_id == 55
        assert frontend.branch == "main"
        assert frontend.paths_include == []
        assert frontend.paths_exclude == []

    def test_load_config_without_code_repos_backward_compat(self, tmp_path):
        """load_config should work with YAML that has no code_repos (backward compat)."""
        from dokumen.config import load_config

        config_content = """
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert config.code_repos == []

    def test_load_config_rejects_duplicate_names_from_yaml(self, tmp_path):
        """load_config should raise ConfigError for duplicate code_repos names in YAML."""
        from dokumen.config import load_config, ConfigError

        config_content = """
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
code_repos:
  - name: same-name
    gitlab_project_id: 10
  - name: same-name
    gitlab_project_id: 20
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        with pytest.raises(ConfigError, match="[Dd]uplicate"):
            load_config(str(config_path))

    def test_dokumen_config_duplicate_names_case_sensitive(self):
        """Duplicate detection is case-sensitive (different case = different name)."""
        from dokumen.config import CodeRepoConfig, DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic"),
            code_repos=[
                CodeRepoConfig(name="my-repo", gitlab_project_id=10),
                CodeRepoConfig(name="My-Repo", gitlab_project_id=20),
            ],
        )
        assert len(config.code_repos) == 2

    def test_load_config_code_repos_defaults_applied_from_yaml(self, tmp_path):
        """load_config should apply defaults to minimally-specified code_repos entries."""
        from dokumen.config import load_config

        config_content = """
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
code_repos:
  - name: minimal-repo
    gitlab_project_id: 7
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        repo = config.code_repos[0]
        assert repo.branch == "main"
        assert repo.paths_include == []
        assert repo.paths_exclude == []
        assert repo.gitlab_url is None
