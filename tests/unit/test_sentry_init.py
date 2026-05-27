"""Tests for Sentry SDK initialization in the CLI."""
import os
from unittest.mock import patch, MagicMock

import pytest


class TestGetCliVersion:
    """Tests for _get_cli_version() helper function."""

    def test_returns_dokumen_version(self):
        """Returns dokumen.__version__ from the VERSION file."""
        from dokumen.sentry_config import _get_cli_version

        with patch("dokumen.__version__", "4.0.1"):
            result = _get_cli_version()
        assert result == "4.0.1"

    def test_returns_unknown_on_import_error(self):
        """Returns 'unknown' when dokumen module cannot provide __version__."""
        from dokumen.sentry_config import _get_cli_version

        with patch.dict("sys.modules", {"dokumen": None}):
            result = _get_cli_version()
        assert result == "unknown"


class TestSentryInit:
    """Tests for init_sentry() function."""

    @patch.dict(os.environ, {"SENTRY_DSN": "https://examplePublicKey@o0.ingest.sentry.io/0"})
    @patch("dokumen.sentry_config.sentry_sdk")
    def test_sentry_init_with_dsn(self, mock_sentry_sdk):
        """When SENTRY_DSN is set, sentry_sdk.init should be called."""
        from dokumen.sentry_config import init_sentry

        init_sentry()

        mock_sentry_sdk.init.assert_called_once()
        call_kwargs = mock_sentry_sdk.init.call_args[1]
        assert call_kwargs["dsn"] == "https://examplePublicKey@o0.ingest.sentry.io/0"
        assert call_kwargs["release"].startswith("cli@")
        assert call_kwargs["environment"] == "production"

    @patch.dict(os.environ, {
        "SENTRY_DSN": "https://examplePublicKey@o0.ingest.sentry.io/0",
        "SENTRY_RELEASE": "cli@3.1.0",
        "SENTRY_ENVIRONMENT": "staging",
    })
    @patch("dokumen.sentry_config.sentry_sdk")
    def test_sentry_init_with_release_env_vars(self, mock_sentry_sdk):
        """When SENTRY_RELEASE and SENTRY_ENVIRONMENT are set, they should be passed to init."""
        from dokumen.sentry_config import init_sentry

        init_sentry()

        call_kwargs = mock_sentry_sdk.init.call_args[1]
        assert call_kwargs["release"] == "cli@3.1.0"
        assert call_kwargs["environment"] == "staging"

    @patch.dict(os.environ, {}, clear=True)
    @patch("dokumen.sentry_config.sentry_sdk")
    def test_sentry_init_without_dsn(self, mock_sentry_sdk):
        """When SENTRY_DSN is not set, sentry_sdk.init should NOT be called."""
        # Remove SENTRY_DSN if present
        env = os.environ.copy()
        env.pop("SENTRY_DSN", None)
        with patch.dict(os.environ, env, clear=True):
            from dokumen.sentry_config import init_sentry

            init_sentry()

            mock_sentry_sdk.init.assert_not_called()

    @patch.dict(os.environ, {"SENTRY_DSN": "https://examplePublicKey@o0.ingest.sentry.io/0"})
    @patch("dokumen.sentry_config.sentry_sdk")
    def test_logging_integration_included(self, mock_sentry_sdk):
        """LoggingIntegration should be included in the integrations list."""
        from dokumen.sentry_config import init_sentry

        init_sentry()

        call_kwargs = mock_sentry_sdk.init.call_args[1]
        integrations = call_kwargs["integrations"]
        # Check that at least one integration is a LoggingIntegration
        integration_types = [type(i).__name__ for i in integrations]
        assert "LoggingIntegration" in integration_types

    @patch.dict(os.environ, {"SENTRY_DSN": "https://examplePublicKey@o0.ingest.sentry.io/0"})
    @patch("dokumen.sentry_config.sentry_sdk")
    def test_sentry_init_sets_traces_sample_rate(self, mock_sentry_sdk):
        """Sentry init should set a traces_sample_rate."""
        from dokumen.sentry_config import init_sentry

        init_sentry()

        call_kwargs = mock_sentry_sdk.init.call_args[1]
        assert "traces_sample_rate" in call_kwargs
        assert 0 <= call_kwargs["traces_sample_rate"] <= 1.0

    @patch.dict(os.environ, {"SENTRY_DSN": "https://example@sentry.io/0"})
    @patch("dokumen.sentry_config.sentry_sdk")
    @patch("dokumen.sentry_config.logger")
    def test_sentry_init_handles_init_error(self, mock_logger, mock_sentry_sdk):
        """If sentry_sdk.init raises, it should be caught, logged, and not crash."""
        from dokumen.sentry_config import init_sentry

        mock_sentry_sdk.init.side_effect = Exception("Connection failed")

        # Should not raise
        init_sentry()

        # Should log the error
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "Failed to initialize Sentry" in call_args[0][0]
        assert "Connection failed" in str(call_args[1].get("extra", {}).get("error", ""))
