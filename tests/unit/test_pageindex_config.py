"""Tests for CLI PageIndex configuration.

Covers:
- Default disabled state
- Enabling via config dict
- Custom model override
- max_files default and validation boundaries
"""

import pytest
from pydantic import ValidationError

from dokumen.config import DokumenConfig, ProviderConfig, PageIndexConfig


def _make_config_dict(**overrides):
    """Create a minimal valid config dict with optional overrides."""
    base = {
        "version": "1.0",
        "provider": ProviderConfig(name="anthropic", model="claude-haiku-4-5-20251001"),
    }
    base.update(overrides)
    return base


class TestPageIndexConfigDefaults:
    """Tests for PageIndex default configuration."""

    def test_default_disabled(self):
        """PageIndex should be disabled by default when not in config."""
        config = DokumenConfig(**_make_config_dict())
        assert config.pageindex.enabled is False

    def test_default_max_files(self):
        """Default max_files should be 50."""
        config = DokumenConfig(**_make_config_dict())
        assert config.pageindex.max_files == 50

    def test_default_model(self):
        """Default model should be the fast model."""
        config = DokumenConfig(**_make_config_dict())
        assert config.pageindex.model == "claude-haiku-4-5-20251001"

    def test_pageindex_field_exists_on_config(self):
        """DokumenConfig should have a pageindex field."""
        config = DokumenConfig(**_make_config_dict())
        assert hasattr(config, "pageindex")
        assert isinstance(config.pageindex, PageIndexConfig)


class TestPageIndexConfigEnabled:
    """Tests for enabling PageIndex."""

    def test_enable_via_config(self):
        """PageIndex can be enabled in config."""
        config = DokumenConfig(
            **_make_config_dict(pageindex=PageIndexConfig(enabled=True))
        )
        assert config.pageindex.enabled is True

    def test_enable_via_dict(self):
        """PageIndex can be enabled via raw dict (as from YAML)."""
        config = DokumenConfig(
            **_make_config_dict(pageindex={"enabled": True})
        )
        assert config.pageindex.enabled is True

    def test_disable_explicitly(self):
        """PageIndex can be explicitly disabled."""
        config = DokumenConfig(
            **_make_config_dict(pageindex={"enabled": False})
        )
        assert config.pageindex.enabled is False


class TestPageIndexConfigModel:
    """Tests for PageIndex model customization."""

    def test_custom_model(self):
        """PageIndex model can be customized."""
        config = DokumenConfig(
            **_make_config_dict(
                pageindex={"enabled": True, "model": "claude-sonnet-4-6"}
            )
        )
        assert config.pageindex.model == "claude-sonnet-4-6"

    def test_model_accepts_any_string(self):
        """Model field should accept any string (no enum restriction)."""
        config = DokumenConfig(
            **_make_config_dict(
                pageindex={"model": "some-future-model-id"}
            )
        )
        assert config.pageindex.model == "some-future-model-id"


class TestPageIndexConfigMaxFiles:
    """Tests for max_files validation boundaries."""

    def test_max_files_custom_value(self):
        """max_files can be set to a custom valid value."""
        config = DokumenConfig(
            **_make_config_dict(pageindex={"max_files": 100})
        )
        assert config.pageindex.max_files == 100

    def test_max_files_minimum_valid(self):
        """max_files of 1 should be valid (lower boundary)."""
        config = DokumenConfig(
            **_make_config_dict(pageindex={"max_files": 1})
        )
        assert config.pageindex.max_files == 1

    def test_max_files_maximum_valid(self):
        """max_files of 500 should be valid (upper boundary)."""
        config = DokumenConfig(
            **_make_config_dict(pageindex={"max_files": 500})
        )
        assert config.pageindex.max_files == 500

    def test_max_files_validation_too_low(self):
        """max_files below 1 should fail validation."""
        with pytest.raises(ValidationError):
            DokumenConfig(
                **_make_config_dict(pageindex={"max_files": 0})
            )

    def test_max_files_validation_negative(self):
        """Negative max_files should fail validation."""
        with pytest.raises(ValidationError):
            DokumenConfig(
                **_make_config_dict(pageindex={"max_files": -1})
            )

    def test_max_files_validation_too_high(self):
        """max_files above 500 should fail validation."""
        with pytest.raises(ValidationError):
            DokumenConfig(
                **_make_config_dict(pageindex={"max_files": 501})
            )
