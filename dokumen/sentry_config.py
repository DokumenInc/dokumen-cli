"""Sentry SDK initialization for the Dokumen CLI."""

import os
import logging

logger = logging.getLogger(__name__)


def _get_cli_version() -> str:
    """Get the CLI version from dokumen.__version__ (read from VERSION file)."""
    try:
        from dokumen import __version__

        return __version__
    except Exception:
        return "unknown"


def init_sentry() -> None:
    """Initialize Sentry SDK if SENTRY_DSN environment variable is set.

    When SENTRY_DSN is not set, Sentry is silently skipped (common in local dev).
    LoggingIntegration captures WARNING+ logs as breadcrumbs and ERROR+ as events.
    """
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        logger.debug("SENTRY_DSN not set, skipping Sentry initialization")
        return

    release = os.environ.get("SENTRY_RELEASE") or f"cli@{_get_cli_version()}"
    environment = os.environ.get("SENTRY_ENVIRONMENT", "production")

    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_logging = LoggingIntegration(
            level=logging.WARNING,  # Breadcrumbs from WARNING+
            event_level=logging.ERROR,  # Events from ERROR+
        )

        sentry_sdk.init(
            dsn=dsn,
            integrations=[sentry_logging],
            traces_sample_rate=0.1,
            release=release,
            environment=environment,
        )
        logger.info(
            "Sentry initialized", extra={"dsn_prefix": dsn[:20] + "...", "release": release}
        )
    except Exception as e:
        logger.warning(
            "Failed to initialize Sentry, continuing without it", extra={"error": str(e)}
        )
