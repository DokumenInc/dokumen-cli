"""
Unit tests for config parsing (dokumen.yaml).

TDD: These tests are written first, before implementation.
They should fail initially until config.py is implemented.
"""
from pathlib import Path

import pytest


class TestDokumenConfigModel:
    """Tests for the DokumenConfig Pydantic model."""

    def test_config_model_exists(self):
        """Config module and DokumenConfig class should exist."""
        from dokumen.config import DokumenConfig
        assert DokumenConfig is not None

    def test_config_model_provider_required(self):
        """Provider section is required - missing it should raise ValidationError."""
        from pydantic import ValidationError
        from dokumen.config import DokumenConfig

        with pytest.raises(ValidationError) as exc_info:
            DokumenConfig(version="1.0")

        # Check that 'provider' is mentioned in the error
        assert "provider" in str(exc_info.value).lower()

    def test_config_model_provider_name_literal(self):
        """Provider name must be 'anthropic' or 'mock'."""
        from pydantic import ValidationError
        from dokumen.config import DokumenConfig, ProviderConfig

        with pytest.raises(ValidationError):
            DokumenConfig(
                version="1.0",
                provider=ProviderConfig(name="invalid_provider")
            )

    def test_config_model_valid_anthropic(self):
        """Valid config with anthropic provider should parse."""
        from dokumen.config import DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic", model="claude-haiku-4-5-20251001")
        )
        assert config.provider.name == "anthropic"
        assert config.provider.model == "claude-haiku-4-5-20251001"

    def test_config_model_valid_mock(self):
        """Valid config with mock provider should parse."""
        from dokumen.config import DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="mock")
        )
        assert config.provider.name == "mock"

    def test_config_model_defaults(self):
        """Optional fields should have sensible defaults."""
        from dokumen.config import DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic")
        )
        # Check defaults
        assert config.execution.timeout == 60
        assert "docs/**/*" in config.coverage.include
        assert config.coverage.exclude == []
        assert config.coverage.min_threshold is None

    def test_default_patterns_include_txt_files(self):
        """Default include patterns should match .txt files in docs/."""
        import fnmatch
        from dokumen.config import CoverageConfig

        config = CoverageConfig()
        txt_file = "docs/notes/readme.txt"

        # Check if any default pattern matches .txt files
        def matches_pattern(path: str, pattern: str) -> bool:
            """Simple glob match for testing."""
            if "**" in pattern:
                parts = pattern.split("**")
                prefix = parts[0].rstrip("/")
                suffix = parts[1].lstrip("/") if len(parts) > 1 else ""
                if prefix and not path.startswith(prefix + "/"):
                    return False
                if suffix:
                    # For docs/**/* pattern, suffix is "*" which should match any filename
                    remaining = path[len(prefix):].lstrip("/") if prefix else path
                    filename = remaining.split("/")[-1]
                    return fnmatch.fnmatch(filename, suffix)
                return True
            return fnmatch.fnmatch(path, pattern)

        matches = any(matches_pattern(txt_file, pattern) for pattern in config.include)
        assert matches is True, f"Default patterns {config.include} should match {txt_file}"

    def test_config_model_execution_timeout_range(self):
        """Execution timeout must be >= 1 (no upper cap)."""
        from pydantic import ValidationError
        from dokumen.config import DokumenConfig, ProviderConfig, ExecutionConfig

        # Too low
        with pytest.raises(ValidationError):
            DokumenConfig(
                version="1.0",
                provider=ProviderConfig(name="anthropic"),
                execution=ExecutionConfig(timeout=0)
            )

        # Large value accepted (no upper cap)
        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic"),
            execution=ExecutionConfig(timeout=2400)
        )
        assert config.execution.timeout == 2400

    def test_config_model_coverage_threshold_range(self):
        """Coverage min_threshold must be between 0 and 100 if set."""
        from pydantic import ValidationError
        from dokumen.config import DokumenConfig, ProviderConfig, CoverageConfig

        # Negative
        with pytest.raises(ValidationError):
            DokumenConfig(
                version="1.0",
                provider=ProviderConfig(name="anthropic"),
                coverage=CoverageConfig(min_threshold=-1)
            )

        # Over 100
        with pytest.raises(ValidationError):
            DokumenConfig(
                version="1.0",
                provider=ProviderConfig(name="anthropic"),
                coverage=CoverageConfig(min_threshold=101)
            )


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_config_valid_file(self, valid_config_path: Path):
        """Loading a valid config file should return DokumenConfig."""
        from dokumen.config import load_config, DokumenConfig

        config = load_config(str(valid_config_path))
        assert isinstance(config, DokumenConfig)
        assert config.provider.name == "anthropic"
        assert config.execution.timeout == 60
        assert config.coverage.min_threshold == 80

    def test_load_config_missing_file(self, tmp_path: Path):
        """Loading a non-existent file should raise ConfigError."""
        from dokumen.config import load_config, ConfigError

        missing_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigError) as exc_info:
            load_config(str(missing_path))

        assert "not found" in str(exc_info.value).lower()

    def test_load_config_invalid_yaml(self, invalid_config_path: Path):
        """Loading malformed YAML should raise ConfigError."""
        from dokumen.config import load_config, ConfigError

        with pytest.raises(ConfigError) as exc_info:
            load_config(str(invalid_config_path))

        assert "yaml" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()

    def test_load_config_missing_provider(self, missing_provider_config_path: Path):
        """Loading config without provider section should raise ConfigError."""
        from dokumen.config import load_config, ConfigError

        with pytest.raises(ConfigError) as exc_info:
            load_config(str(missing_provider_config_path))

        assert "provider" in str(exc_info.value).lower()

    def test_load_config_empty_file(self, tmp_path: Path):
        """Loading empty config file should raise ConfigError."""
        from dokumen.config import load_config, ConfigError

        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("")

        with pytest.raises(ConfigError) as exc_info:
            load_config(str(empty_path))

        assert "empty" in str(exc_info.value).lower()

    def test_load_config_minimal(self, minimal_config_path: Path):
        """Loading minimal config should use defaults for optional fields."""
        from dokumen.config import load_config

        config = load_config(str(minimal_config_path))
        assert config.provider.name == "anthropic"
        # Defaults should be applied
        assert config.execution.timeout == 60
        assert config.provider.model == "claude-haiku-4-5-20251001"

    def test_load_config_default_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Loading without path should look for dokumen.yaml in current directory."""
        from dokumen.config import load_config, ConfigError

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # No dokumen.yaml exists - should raise ConfigError
        with pytest.raises(ConfigError):
            load_config()


class TestConfigError:
    """Tests for the ConfigError exception."""

    def test_config_error_exists(self):
        """ConfigError exception class should exist."""
        from dokumen.config import ConfigError
        assert issubclass(ConfigError, Exception)

    def test_config_error_message(self):
        """ConfigError should preserve error message."""
        from dokumen.config import ConfigError

        error = ConfigError("Test error message")
        assert "Test error message" in str(error)


class TestExploreConfig:
    """Tests for ExploreConfig model."""

    def test_explore_config_exists(self):
        """ExploreConfig class should exist."""
        from dokumen.config import ExploreConfig
        assert ExploreConfig is not None

    def test_explore_config_defaults(self):
        """ExploreConfig should have sensible defaults."""
        from dokumen.config import ExploreConfig

        config = ExploreConfig()
        assert config.enabled is True
        assert config.model is not None
        assert config.max_files == 20
        assert config.timeout == 60

    def test_explore_config_custom_values(self):
        """ExploreConfig should accept custom values."""
        from dokumen.config import ExploreConfig

        config = ExploreConfig(
            enabled=False,
            model="claude-haiku-4-5-20251001",
            max_files=10,
            timeout=30
        )
        assert config.enabled is False
        assert config.model == "claude-haiku-4-5-20251001"
        assert config.max_files == 10
        assert config.timeout == 30

    def test_explore_config_max_files_range(self):
        """ExploreConfig max_files must be positive."""
        from pydantic import ValidationError
        from dokumen.config import ExploreConfig

        with pytest.raises(ValidationError):
            ExploreConfig(max_files=0)

    def test_explore_config_timeout_range(self):
        """ExploreConfig timeout must be positive."""
        from pydantic import ValidationError
        from dokumen.config import ExploreConfig

        with pytest.raises(ValidationError):
            ExploreConfig(timeout=0)

    def test_dokumen_config_includes_explore(self):
        """DokumenConfig should include explore section."""
        from dokumen.config import DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic")
        )
        # Explore should have defaults
        assert config.explore is not None
        assert config.explore.enabled is True

    def test_dokumen_config_custom_explore(self):
        """DokumenConfig should accept custom explore settings."""
        from dokumen.config import DokumenConfig, ProviderConfig, ExploreConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic"),
            explore=ExploreConfig(
                enabled=False,
                model="claude-haiku-4-5-20251001",
                max_files=15
            )
        )
        assert config.explore.enabled is False
        assert config.explore.model == "claude-haiku-4-5-20251001"
        assert config.explore.max_files == 15


class TestToolsConfig:
    """Tests for ToolsConfig and per-tool config models."""

    def test_tools_config_defaults_none_by_default(self):
        """ToolsConfig defaults and allowed should be None by default."""
        from dokumen.config import ToolsConfig

        config = ToolsConfig()
        assert config.defaults is None
        assert config.allowed is None

    def test_tools_config_defaults_valid_names(self):
        """ToolsConfig accepts valid tool names in defaults."""
        from dokumen.config import ToolsConfig

        config = ToolsConfig(defaults=["read_file", "glob", "run_shell_command"])
        assert config.defaults == ["read_file", "glob", "run_shell_command"]

    def test_tools_config_defaults_rejects_unknown_tools(self):
        """ToolsConfig rejects unknown tool names in defaults."""
        from pydantic import ValidationError
        from dokumen.config import ToolsConfig

        with pytest.raises(ValidationError, match="not_a_real_tool"):
            ToolsConfig(defaults=["read_file", "not_a_real_tool"])

    def test_tools_config_allowed_rejects_unknown_tools(self):
        """ToolsConfig rejects unknown tool names in allowed."""
        from pydantic import ValidationError
        from dokumen.config import ToolsConfig

        with pytest.raises(ValidationError, match="fake_tool"):
            ToolsConfig(allowed=["read_file", "fake_tool"])

    def test_tools_config_defaults_must_be_subset_of_allowed(self):
        """If both defaults and allowed are set, defaults must be subset of allowed."""
        from pydantic import ValidationError
        from dokumen.config import ToolsConfig

        with pytest.raises(ValidationError, match="not in allowed"):
            ToolsConfig(
                defaults=["read_file", "web_fetch"],
                allowed=["read_file", "glob"]
            )

    def test_tools_config_defaults_subset_of_allowed_valid(self):
        """Defaults that are a subset of allowed should pass validation."""
        from dokumen.config import ToolsConfig

        config = ToolsConfig(
            defaults=["read_file"],
            allowed=["read_file", "glob", "run_shell_command"]
        )
        assert config.defaults == ["read_file"]
        assert config.allowed == ["read_file", "glob", "run_shell_command"]

    def test_tools_config_per_tool_defaults(self):
        """Per-tool config should have sensible defaults."""
        from dokumen.config import ToolConfigMap

        config = ToolConfigMap()
        assert config.run_shell_command.timeout == 30.0
        assert config.web_fetch.timeout == 30.0
        assert config.web_search.model is None
        assert config.web_search.max_searches is None

    def test_tools_config_per_tool_overrides(self):
        """Per-tool config should accept custom values."""
        from dokumen.config import ShellToolConfig, HttpToolConfig, WebSearchToolConfig, ToolConfigMap

        config = ToolConfigMap(
            run_shell_command=ShellToolConfig(timeout=60.0),
            web_fetch=HttpToolConfig(timeout=15.0),
            web_search=WebSearchToolConfig(model="sonar-pro", max_searches=10),
        )
        assert config.run_shell_command.timeout == 60.0
        assert config.web_fetch.timeout == 15.0
        assert config.web_search.model == "sonar-pro"
        assert config.web_search.max_searches == 10

    def test_shell_tool_config_timeout_range(self):
        """ShellToolConfig timeout must be between 1.0 and 300.0."""
        from pydantic import ValidationError
        from dokumen.config import ShellToolConfig

        with pytest.raises(ValidationError):
            ShellToolConfig(timeout=0.5)

        with pytest.raises(ValidationError):
            ShellToolConfig(timeout=301.0)

    def test_http_tool_config_timeout_range(self):
        """HttpToolConfig timeout must be between 1.0 and 120.0."""
        from pydantic import ValidationError
        from dokumen.config import HttpToolConfig

        with pytest.raises(ValidationError):
            HttpToolConfig(timeout=0.5)

        with pytest.raises(ValidationError):
            HttpToolConfig(timeout=121.0)

    def test_web_search_config_max_searches_range(self):
        """WebSearchToolConfig max_searches must be between 1 and 200."""
        from pydantic import ValidationError
        from dokumen.config import WebSearchToolConfig

        with pytest.raises(ValidationError):
            WebSearchToolConfig(max_searches=0)

        with pytest.raises(ValidationError):
            WebSearchToolConfig(max_searches=201)

    def test_dokumen_config_tools_section_optional(self):
        """DokumenConfig should work without tools section (backward compat)."""
        from dokumen.config import DokumenConfig, ProviderConfig

        config = DokumenConfig(
            version="1.0",
            provider=ProviderConfig(name="anthropic")
        )
        assert config.tools is not None
        assert config.tools.defaults is None
        assert config.tools.allowed is None

    def test_load_config_with_tools_section(self, tmp_path):
        """load_config should parse tools section from YAML."""
        from dokumen.config import load_config

        config_content = """
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
tools:
  defaults:
    - read_file
    - glob
  allowed:
    - read_file
    - glob
    - run_shell_command
    - web_fetch
  config:
    run_shell_command:
      timeout: 60.0
    web_fetch:
      timeout: 15.0
    web_search:
      model: sonar-pro
      max_searches: 10
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert config.tools.defaults == ["read_file", "glob"]
        assert "run_shell_command" in config.tools.allowed
        assert config.tools.config.run_shell_command.timeout == 60.0
        assert config.tools.config.web_fetch.timeout == 15.0
        assert config.tools.config.web_search.model == "sonar-pro"
        assert config.tools.config.web_search.max_searches == 10


class TestPreprocessYaml:
    """Tests for _preprocess_yaml stripping colon-prefixed comment lines."""

    def test_preprocess_strips_colon_first_line(self):
        """A line starting with ':' on the first line should be stripped."""
        from dokumen.config import _preprocess_yaml

        content = ": DOKUMEN\nversion: '1.0'\nprovider:\n  name: anthropic\n"
        result = _preprocess_yaml(content)
        assert ": DOKUMEN" not in result
        assert "version: '1.0'" in result

    def test_preprocess_strips_multiple_colon_lines(self):
        """Multiple lines starting with ':' should all be stripped."""
        from dokumen.config import _preprocess_yaml

        content = ": DOKUMEN\n: This is a label\nversion: '1.0'\n: Another label\nprovider:\n  name: anthropic\n"
        result = _preprocess_yaml(content)
        assert ": DOKUMEN" not in result
        assert ": This is a label" not in result
        assert ": Another label" not in result
        assert "version: '1.0'" in result
        assert "provider:" in result

    def test_preprocess_strips_indented_colon_lines(self):
        """Indented lines where first non-whitespace is ':' should be stripped."""
        from dokumen.config import _preprocess_yaml

        content = "  : indented comment\nversion: '1.0'\n"
        result = _preprocess_yaml(content)
        assert ": indented comment" not in result
        assert "version: '1.0'" in result

    def test_preprocess_preserves_colon_in_values(self):
        """Colons in YAML values (e.g., 'model: claude-sonnet') should NOT be affected."""
        from dokumen.config import _preprocess_yaml

        content = "version: '1.0'\nprovider:\n  name: anthropic\n  model: claude-sonnet-4-5-20250929\n"
        result = _preprocess_yaml(content)
        assert result == content

    def test_preprocess_preserves_empty_lines(self):
        """Empty lines should be preserved."""
        from dokumen.config import _preprocess_yaml

        content = "version: '1.0'\n\nprovider:\n  name: anthropic\n"
        result = _preprocess_yaml(content)
        assert result == content

    def test_preprocess_no_colon_lines_unchanged(self):
        """Content with no colon-prefixed lines should pass through unchanged."""
        from dokumen.config import _preprocess_yaml

        content = "version: '1.0'\nprovider:\n  name: anthropic\n"
        result = _preprocess_yaml(content)
        assert result == content

    def test_load_config_with_colon_comment(self, tmp_path):
        """load_config should handle files with ':' comment lines."""
        from dokumen.config import load_config

        config_content = """: DOKUMEN
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert config.provider.name == "anthropic"
        assert config.provider.model == "claude-haiku-4-5-20251001"

    def test_load_config_with_multiple_colon_comments(self, tmp_path):
        """load_config should strip multiple ':' comment lines from config."""
        from dokumen.config import load_config

        config_content = """: DOKUMEN
: Generated by dokumen-employee
version: "1.0"
provider:
  name: mock
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert config.provider.name == "mock"


class TestExecutionConfigMaxToolResultChars:
    """Tests for ExecutionConfig.max_tool_result_chars field."""

    def test_default_value(self):
        """ExecutionConfig should default max_tool_result_chars to 50000."""
        from dokumen.config import ExecutionConfig

        config = ExecutionConfig()
        assert config.max_tool_result_chars == 50000

    def test_custom_value(self):
        """ExecutionConfig should accept custom max_tool_result_chars."""
        from dokumen.config import ExecutionConfig

        config = ExecutionConfig(max_tool_result_chars=30000)
        assert config.max_tool_result_chars == 30000

    def test_zero_disables(self):
        """max_tool_result_chars=0 should be valid (disables truncation)."""
        from dokumen.config import ExecutionConfig

        config = ExecutionConfig(max_tool_result_chars=0)
        assert config.max_tool_result_chars == 0

    def test_rejects_too_small(self):
        """max_tool_result_chars below 1000 (and not 0) should raise ValidationError."""
        from pydantic import ValidationError
        from dokumen.config import ExecutionConfig

        with pytest.raises(ValidationError):
            ExecutionConfig(max_tool_result_chars=500)

    def test_rejects_too_large(self):
        """max_tool_result_chars above 500000 should raise ValidationError."""
        from pydantic import ValidationError
        from dokumen.config import ExecutionConfig

        with pytest.raises(ValidationError):
            ExecutionConfig(max_tool_result_chars=600000)

    def test_parsed_from_dokumen_yaml(self, tmp_path):
        """DokumenConfig should parse execution.max_tool_result_chars from YAML."""
        from dokumen.config import load_config

        config_content = """
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
execution:
  timeout: 120
  max_tool_result_chars: 25000
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert config.execution.max_tool_result_chars == 25000


# =============================================================================
# TestJudgeRetriesConfig - judge_retries in ExecutionConfig
# =============================================================================


class TestJudgeRetriesConfig:
    """Test judge_retries field in ExecutionConfig."""

    def test_judge_retries_config_default(self):
        """Global ExecutionConfig.judge_retries default of 2."""
        from dokumen.config import ExecutionConfig

        config = ExecutionConfig()
        assert config.judge_retries == 2

    def test_judge_retries_config_custom(self):
        """judge_retries accepts custom value."""
        from dokumen.config import ExecutionConfig

        config = ExecutionConfig(judge_retries=5)
        assert config.judge_retries == 5

    def test_judge_retries_config_zero(self):
        """judge_retries=0 disables retries."""
        from dokumen.config import ExecutionConfig

        config = ExecutionConfig(judge_retries=0)
        assert config.judge_retries == 0

    def test_judge_retries_config_rejects_negative(self):
        """judge_retries below 0 raises ValidationError."""
        from pydantic import ValidationError
        from dokumen.config import ExecutionConfig

        with pytest.raises(ValidationError):
            ExecutionConfig(judge_retries=-1)

    def test_judge_retries_config_rejects_too_high(self):
        """judge_retries above 5 raises ValidationError."""
        from pydantic import ValidationError
        from dokumen.config import ExecutionConfig

        with pytest.raises(ValidationError):
            ExecutionConfig(judge_retries=6)

    def test_judge_retries_config_from_yaml(self, tmp_path):
        """judge_retries parsed from yaml."""
        from dokumen.config import load_config

        config_content = """
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
execution:
  timeout: 120
  judge_retries: 3
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert config.execution.judge_retries == 3


class TestToolsConfigBlocked:
    """Tests for ToolsConfig.blocked field."""

    def test_blocked_defaults_to_none(self):
        """ToolsConfig.blocked defaults to None when not specified."""
        from dokumen.config import ToolsConfig

        config = ToolsConfig()
        assert config.blocked is None

    def test_blocked_parsed_from_yaml(self, tmp_path):
        """tools.blocked is parsed from YAML config."""
        from dokumen.config import load_config

        config_content = """
version: "1.0"
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001
tools:
  blocked:
    - web_fetch
    - web_search
"""
        config_path = tmp_path / "dokumen.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert config.tools.blocked == ["web_fetch", "web_search"]

    def test_blocked_with_valid_tool_names(self):
        """ToolsConfig accepts valid tool names in blocked."""
        from dokumen.config import ToolsConfig

        config = ToolsConfig(blocked=["web_fetch", "run_shell_command"])
        assert config.blocked == ["web_fetch", "run_shell_command"]

    def test_blocked_with_invalid_tool_names_raises(self):
        """ToolsConfig rejects unknown tool names in blocked."""
        from pydantic import ValidationError
        from dokumen.config import ToolsConfig

        with pytest.raises(ValidationError, match="not_a_tool"):
            ToolsConfig(blocked=["read_file", "not_a_tool"])

    def test_blocked_empty_list(self):
        """ToolsConfig accepts empty blocked list."""
        from dokumen.config import ToolsConfig

        config = ToolsConfig(blocked=[])
        assert config.blocked == []
