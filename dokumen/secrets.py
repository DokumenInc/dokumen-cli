"""1Password secret management for Dokumen CLI.

This module provides synchronous wrappers for loading secrets from 1Password.
The CLI runs in synchronous context (Click), so async calls are wrapped with asyncio.run().

Usage:
    from dokumen.secrets import get_anthropic_key

    api_key = get_anthropic_key()  # Returns key from 1Password or env var
"""

import asyncio
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


def _use_1password() -> bool:
    """Check if 1Password integration is enabled.

    Returns:
        True if USE_1PASSWORD env var is set to "true" (case-insensitive)
    """
    return os.getenv("USE_1PASSWORD", "false").lower() == "true"


def _get_environment() -> str:
    """Get current environment from DOKUMEN_ENV.

    Returns:
        Environment name (Production, Staging, Development)
        Defaults to "Development" for CLI usage.
    """
    return os.getenv("DOKUMEN_ENV", "Development")


class CLISecretsManager:
    """Synchronous wrapper for 1Password secret retrieval in CLI context."""

    def __init__(self) -> None:
        """Initialize CLI secrets manager."""
        self._client = None
        self._cache: dict[str, str] = {}
        logger.debug("CLISecretsManager initialized")

    async def _get_client(self):
        """Get or create the 1Password client (async).

        Returns:
            Authenticated 1Password client

        Raises:
            ValueError: If OP_SERVICE_ACCOUNT_TOKEN not set or auth fails
        """
        if self._client is None:
            try:
                from onepassword import client
            except ImportError as exc:
                raise ValueError(
                    "1Password integration requires the optional integrations extra. "
                    'Install with: pip install "dokumen[integrations]"'
                ) from exc

            token = os.getenv("OP_SERVICE_ACCOUNT_TOKEN")
            if not token:
                raise ValueError(
                    "OP_SERVICE_ACCOUNT_TOKEN environment variable not set. "
                    "Set USE_1PASSWORD=false to use env vars instead."
                )

            logger.debug("Authenticating 1Password client")
            self._client = await client.Client.authenticate(
                auth=token,
                integration_name="Dokumen-CLI",
                integration_version="2.0",
            )
            logger.debug("1Password client authenticated successfully")

        return self._client

    async def _get_secret_async(self, vault: str, item: str, field: str) -> str:
        """Retrieve a secret from 1Password (async).

        Args:
            vault: Vault name (e.g., "Dokumen-Development")
            item: Item title (e.g., "Anthropic API")
            field: Field name (e.g., "api_key")

        Returns:
            Secret value as string

        Raises:
            ValueError: If secret not found or invalid
        """
        secret_ref = f"op://{vault}/{item}/{field}"

        # Check cache first
        if secret_ref in self._cache:
            logger.debug(f"Retrieved secret from cache: {vault}/{item}")
            return self._cache[secret_ref]

        try:
            client = await self._get_client()
            value = client.secrets.resolve(secret_ref)
            logger.debug(f"Retrieved secret from 1Password: {vault}/{item}")

            # Cache the value
            self._cache[secret_ref] = value

            return value
        except Exception as e:
            logger.error(f"Failed to retrieve secret {vault}/{item}: {e}")
            raise ValueError(f"Failed to retrieve {secret_ref}: {e}")

    def get_secret(self, vault: str, item: str, field: str) -> str:
        """Retrieve a secret from 1Password (synchronous wrapper).

        Args:
            vault: Vault name
            item: Item title
            field: Field name

        Returns:
            Secret value as string
        """
        return asyncio.run(self._get_secret_async(vault, item, field))


# Singleton instance
_cli_secrets_manager: Optional[CLISecretsManager] = None


def _get_cli_secrets_manager() -> CLISecretsManager:
    """Get or create the CLI secrets manager singleton."""
    global _cli_secrets_manager
    if _cli_secrets_manager is None:
        _cli_secrets_manager = CLISecretsManager()
    return _cli_secrets_manager


def get_anthropic_key() -> str:
    """Load Anthropic API key from 1Password or environment variable.

    Priority:
    1. If USE_1PASSWORD=true, load from 1Password vault
    2. Otherwise, use ANTHROPIC_API_KEY environment variable

    Returns:
        Anthropic API key string

    Raises:
        ValueError: If no API key found in either source
    """
    if _use_1password():
        logger.info("Loading Anthropic API key from 1Password")
        env = _get_environment()
        manager = _get_cli_secrets_manager()
        try:
            key = manager.get_secret(f"Dokumen-{env}", "Anthropic API", "api_key")
            logger.info(
                "Secret resolved",
                extra={"source": "1password", "vault": f"Dokumen-{env}", "item": "Anthropic API"},
            )
            return key
        except ValueError as e:
            logger.warning(f"Failed to load from 1Password: {e}, falling back to env var")

    # Fallback to environment variable
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error(
            "Secret resolution failed",
            extra={
                "item": "Anthropic API",
                "sources_tried": "1password,env_var" if _use_1password() else "env_var",
            },
        )
        raise ValueError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable "
            "or enable 1Password with USE_1PASSWORD=true"
        )
    logger.info("Secret resolved", extra={"source": "env_var", "item": "ANTHROPIC_API_KEY"})
    return api_key


@lru_cache(maxsize=1)
def get_gitlab_token() -> Optional[str]:
    """Load GitLab token from 1Password or environment variable.

    Returns:
        GitLab PAT or service token, or None if not found
    """
    if _use_1password():
        logger.info("Loading GitLab token from 1Password")
        env = _get_environment()
        manager = _get_cli_secrets_manager()
        try:
            token = manager.get_secret(f"Dokumen-{env}", "GitLab Service Account", "token")
            logger.info(
                "Secret resolved",
                extra={
                    "source": "1password",
                    "vault": f"Dokumen-{env}",
                    "item": "GitLab Service Account",
                },
            )
            return token
        except ValueError:
            logger.debug("GitLab token not in 1Password, falling back to env var")

    token = os.getenv("GITLAB_SERVICE_TOKEN") or os.getenv("GITLAB_TOKEN")
    if token:
        source_var = "GITLAB_SERVICE_TOKEN" if os.getenv("GITLAB_SERVICE_TOKEN") else "GITLAB_TOKEN"
        logger.info("Secret resolved", extra={"source": "env_var", "item": source_var})
    else:
        logger.warning("GitLab token not found in any source")
    return token


def reset_cli_secrets_manager() -> None:
    """Reset the CLI secrets manager singleton (for testing)."""
    global _cli_secrets_manager
    _cli_secrets_manager = None
    get_gitlab_token.cache_clear()
