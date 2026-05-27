"""Tests for 1Password secrets integration in CLI."""
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dokumen.secrets import (
    CLISecretsManager,
    _get_cli_secrets_manager,
    _get_environment,
    _use_1password,
    get_anthropic_key,
    get_gitlab_token,
    reset_cli_secrets_manager,
)


class TestUse1Password:
    """Tests for _use_1password function."""

    def test_returns_false_by_default(self):
        """Returns False when USE_1PASSWORD not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert _use_1password() is False

    def test_returns_false_when_set_to_false(self):
        """Returns False when USE_1PASSWORD=false."""
        with patch.dict(os.environ, {"USE_1PASSWORD": "false"}):
            assert _use_1password() is False

    def test_returns_true_when_set_to_true(self):
        """Returns True when USE_1PASSWORD=true."""
        with patch.dict(os.environ, {"USE_1PASSWORD": "true"}):
            assert _use_1password() is True

    def test_returns_true_case_insensitive(self):
        """Returns True for any case variation of 'true'."""
        with patch.dict(os.environ, {"USE_1PASSWORD": "TRUE"}):
            assert _use_1password() is True
        with patch.dict(os.environ, {"USE_1PASSWORD": "True"}):
            assert _use_1password() is True


class TestGetEnvironment:
    """Tests for _get_environment function."""

    def test_defaults_to_development(self):
        """Defaults to Development for CLI usage."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_environment() == "Development"

    def test_returns_production_when_set(self):
        """Returns Production when DOKUMEN_ENV=Production."""
        with patch.dict(os.environ, {"DOKUMEN_ENV": "Production"}):
            assert _get_environment() == "Production"

    def test_returns_staging_when_set(self):
        """Returns Staging when DOKUMEN_ENV=Staging."""
        with patch.dict(os.environ, {"DOKUMEN_ENV": "Staging"}):
            assert _get_environment() == "Staging"


class TestCLISecretsManager:
    """Tests for CLISecretsManager class."""

    @pytest.fixture
    def manager(self):
        """Create fresh manager instance."""
        reset_cli_secrets_manager()
        return CLISecretsManager()

    @pytest.fixture
    def mock_op_client(self):
        """Create mock 1Password client."""
        client = Mock()
        client.secrets = Mock()
        client.secrets.resolve = Mock(return_value="test-secret-value")
        return client

    @pytest.mark.asyncio
    async def test_get_client_raises_without_token(self, manager):
        """_get_client raises ValueError when token not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OP_SERVICE_ACCOUNT_TOKEN"):
                await manager._get_client()

    @pytest.mark.asyncio
    async def test_get_secret_async_success(self, manager, mock_op_client):
        """_get_secret_async returns secret value."""

        async def mock_get_client():
            return mock_op_client

        manager._get_client = mock_get_client

        result = await manager._get_secret_async("Vault", "Item", "field")

        assert result == "test-secret-value"
        mock_op_client.secrets.resolve.assert_called_once_with("op://Vault/Item/field")

    @pytest.mark.asyncio
    async def test_get_secret_async_caching(self, manager, mock_op_client):
        """_get_secret_async caches results."""

        async def mock_get_client():
            return mock_op_client

        manager._get_client = mock_get_client

        # First call
        await manager._get_secret_async("Vault", "Item", "field")
        # Second call (should use cache)
        await manager._get_secret_async("Vault", "Item", "field")

        # Should only call resolve once
        assert mock_op_client.secrets.resolve.call_count == 1

    @pytest.mark.asyncio
    async def test_get_secret_async_error(self, manager, mock_op_client):
        """_get_secret_async raises ValueError on error."""
        mock_op_client.secrets.resolve.side_effect = Exception("Not found")

        async def mock_get_client():
            return mock_op_client

        manager._get_client = mock_get_client

        with pytest.raises(ValueError, match="Failed to retrieve"):
            await manager._get_secret_async("Vault", "Missing", "field")


class TestGetAnthropicKey:
    """Tests for get_anthropic_key function."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Reset state before/after each test."""
        reset_cli_secrets_manager()
        yield
        reset_cli_secrets_manager()

    def test_returns_env_var_when_1password_disabled(self):
        """Returns ANTHROPIC_API_KEY when USE_1PASSWORD=false."""
        with patch.dict(
            os.environ,
            {"USE_1PASSWORD": "false", "ANTHROPIC_API_KEY": "sk-ant-from-env"},
            clear=True,
        ):
            result = get_anthropic_key()
            assert result == "sk-ant-from-env"

    def test_raises_when_no_key_found(self):
        """Raises ValueError when no API key available."""
        with patch.dict(os.environ, {"USE_1PASSWORD": "false"}, clear=True):
            with pytest.raises(ValueError, match="No Anthropic API key found"):
                get_anthropic_key()

    def test_loads_from_1password_when_enabled(self):
        """Loads from 1Password when USE_1PASSWORD=true."""
        mock_manager = Mock()
        mock_manager.get_secret = Mock(return_value="sk-ant-from-1password")

        with patch.dict(
            os.environ,
            {"USE_1PASSWORD": "true", "DOKUMEN_ENV": "Development"},
            clear=True,
        ):
            with patch(
                "dokumen.secrets._get_cli_secrets_manager",
                return_value=mock_manager,
            ):
                result = get_anthropic_key()

        assert result == "sk-ant-from-1password"
        mock_manager.get_secret.assert_called_once_with(
            "Dokumen-Development", "Anthropic API", "api_key"
        )

    def test_falls_back_to_env_on_1password_error(self):
        """Falls back to env var when 1Password fails."""
        mock_manager = Mock()
        mock_manager.get_secret = Mock(side_effect=ValueError("Connection failed"))

        with patch.dict(
            os.environ,
            {
                "USE_1PASSWORD": "true",
                "DOKUMEN_ENV": "Development",
                "ANTHROPIC_API_KEY": "sk-ant-fallback",
            },
            clear=True,
        ):
            with patch(
                "dokumen.secrets._get_cli_secrets_manager",
                return_value=mock_manager,
            ):
                result = get_anthropic_key()

        assert result == "sk-ant-fallback"


class TestGetGitlabToken:
    """Tests for get_gitlab_token function."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Reset state before/after each test."""
        reset_cli_secrets_manager()
        get_gitlab_token.cache_clear()
        yield
        reset_cli_secrets_manager()
        get_gitlab_token.cache_clear()

    def test_returns_env_var_when_1password_disabled(self):
        """Returns GITLAB_SERVICE_TOKEN when USE_1PASSWORD=false."""
        with patch.dict(
            os.environ,
            {"USE_1PASSWORD": "false", "GITLAB_SERVICE_TOKEN": "glpat-from-env"},
            clear=True,
        ):
            result = get_gitlab_token()
            assert result == "glpat-from-env"

    def test_returns_gitlab_token_as_fallback(self):
        """Returns GITLAB_TOKEN as fallback."""
        with patch.dict(
            os.environ,
            {"USE_1PASSWORD": "false", "GITLAB_TOKEN": "glpat-gitlab-token"},
            clear=True,
        ):
            result = get_gitlab_token()
            assert result == "glpat-gitlab-token"

    def test_returns_none_when_no_token(self):
        """Returns None when no token available."""
        with patch.dict(os.environ, {"USE_1PASSWORD": "false"}, clear=True):
            result = get_gitlab_token()
            assert result is None

    def test_loads_from_1password_when_enabled(self):
        """Loads from 1Password when USE_1PASSWORD=true."""
        mock_manager = Mock()
        mock_manager.get_secret = Mock(return_value="glpat-from-1password")

        with patch.dict(
            os.environ,
            {"USE_1PASSWORD": "true", "DOKUMEN_ENV": "Production"},
            clear=True,
        ):
            with patch(
                "dokumen.secrets._get_cli_secrets_manager",
                return_value=mock_manager,
            ):
                result = get_gitlab_token()

        assert result == "glpat-from-1password"


class TestSingleton:
    """Tests for singleton pattern."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Reset singleton before/after each test."""
        reset_cli_secrets_manager()
        yield
        reset_cli_secrets_manager()

    def test_get_cli_secrets_manager_returns_same_instance(self):
        """_get_cli_secrets_manager returns same instance on repeated calls."""
        manager1 = _get_cli_secrets_manager()
        manager2 = _get_cli_secrets_manager()
        assert manager1 is manager2

    def test_reset_clears_singleton(self):
        """reset_cli_secrets_manager clears the singleton."""
        manager1 = _get_cli_secrets_manager()
        reset_cli_secrets_manager()
        manager2 = _get_cli_secrets_manager()
        assert manager1 is not manager2
