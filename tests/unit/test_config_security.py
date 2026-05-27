"""Tests for coverage config security validation — blocking sensitive file patterns."""

import pytest
from pydantic import ValidationError

from dokumen.config import CoverageConfig


class TestCoveragePatternValidation:
    """Ensure coverage include patterns reject sensitive file targets."""

    def test_coverage_allows_docs_pattern(self):
        config = CoverageConfig(include=["docs/**/*.md"])
        assert config.include == ["docs/**/*.md"]

    def test_coverage_allows_readme(self):
        config = CoverageConfig(include=["README.md"])
        assert config.include == ["README.md"]

    def test_coverage_allows_tests_pattern(self):
        config = CoverageConfig(include=["tests/**/*"])
        assert config.include == ["tests/**/*"]

    def test_coverage_rejects_env_pattern(self):
        with pytest.raises(ValidationError, match="sensitive files"):
            CoverageConfig(include=["**/.env"])

    def test_coverage_rejects_traversal(self):
        with pytest.raises(ValidationError, match="path traversal"):
            CoverageConfig(include=["../../../etc/passwd"])

    def test_coverage_rejects_pem_pattern(self):
        with pytest.raises(ValidationError, match="sensitive files"):
            CoverageConfig(include=["**/*.pem"])

    def test_coverage_rejects_key_pattern(self):
        with pytest.raises(ValidationError, match="sensitive files"):
            CoverageConfig(include=["**/*.key"])

    def test_coverage_rejects_secrets_pattern(self):
        with pytest.raises(ValidationError, match="sensitive files"):
            CoverageConfig(include=["secrets/**/*"])

    def test_coverage_rejects_credential_pattern(self):
        with pytest.raises(ValidationError, match="sensitive files"):
            CoverageConfig(include=["**/credential*"])

    def test_coverage_default_patterns_valid(self):
        config = CoverageConfig()
        assert config.include == ["docs/**/*", "README.md"]


class TestCoverageExcludePatternValidation:
    """Ensure coverage exclude patterns also reject sensitive file targets."""

    def test_exclude_allows_normal_pattern(self):
        config = CoverageConfig(exclude=["docs/drafts/**"])
        assert config.exclude == ["docs/drafts/**"]

    def test_exclude_rejects_env_pattern(self):
        with pytest.raises(ValidationError, match="sensitive files"):
            CoverageConfig(exclude=["**/.env"])

    def test_exclude_rejects_traversal(self):
        with pytest.raises(ValidationError, match="path traversal"):
            CoverageConfig(exclude=["../../etc/passwd"])

    def test_exclude_rejects_pem_pattern(self):
        with pytest.raises(ValidationError, match="sensitive files"):
            CoverageConfig(exclude=["**/*.pem"])
