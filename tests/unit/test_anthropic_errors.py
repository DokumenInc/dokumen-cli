"""Tests for Anthropic provider error handling.

Tests rate limiting, timeouts, malformed responses, and API key validation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dokumen.providers.anthropic import AnthropicProvider, parse_image_data, parse_pdf_data
from dokumen.providers.retry import (
    is_rate_limit_error,
    is_retryable_error,
    retry_with_exponential_backoff,
)


# =============================================================================
# Rate Limit Detection Tests
# =============================================================================


class TestIsRateLimitError:
    """Tests for is_rate_limit_error() function."""

    def test_detects_429_status_code(self):
        """Detects 429 in exception message."""
        exc = Exception("Error: 429 Too Many Requests")
        assert is_rate_limit_error(exc) is True

    def test_detects_rate_limit_in_message(self):
        """Detects 'rate limit' in exception message."""
        exc = Exception("You have exceeded the rate limit")
        assert is_rate_limit_error(exc) is True

    def test_detects_rate_limit_underscore(self):
        """Detects 'rate_limit' in exception message."""
        exc = Exception("Error code: rate_limit_exceeded")
        assert is_rate_limit_error(exc) is True

    def test_detects_ratelimit_exception_type(self):
        """Detects RateLimitError exception type name."""
        class RateLimitError(Exception):
            pass

        exc = RateLimitError("Too fast")
        assert is_rate_limit_error(exc) is True

    def test_non_rate_limit_error_returns_false(self):
        """Returns False for non-rate-limit errors."""
        exc = Exception("Invalid API key")
        assert is_rate_limit_error(exc) is False


class TestIsRetryableError:
    """Tests for is_retryable_error() function."""

    def test_rate_limit_is_retryable(self):
        """Rate limit errors are retryable."""
        exc = Exception("Error: 429 Rate Limit")
        assert is_retryable_error(exc) is True

    def test_500_error_is_retryable(self):
        """500 Internal Server Error is retryable."""
        exc = Exception("HTTP Error 500: Internal Server Error")
        assert is_retryable_error(exc) is True

    def test_502_error_is_retryable(self):
        """502 Bad Gateway is retryable."""
        exc = Exception("HTTP 502 Bad Gateway")
        assert is_retryable_error(exc) is True

    def test_503_error_is_retryable(self):
        """503 Service Unavailable is retryable."""
        exc = Exception("503 Service Unavailable")
        assert is_retryable_error(exc) is True

    def test_504_error_is_retryable(self):
        """504 Gateway Timeout is retryable."""
        exc = Exception("504 Gateway Timeout")
        assert is_retryable_error(exc) is True

    def test_overloaded_is_retryable(self):
        """Overloaded errors are retryable."""
        exc = Exception("The server is overloaded")
        assert is_retryable_error(exc) is True

    def test_timeout_is_retryable(self):
        """Timeout errors are retryable."""
        exc = Exception("Connection timed out")
        assert is_retryable_error(exc) is True

    def test_connection_reset_is_retryable(self):
        """Connection reset errors are retryable."""
        exc = Exception("Connection reset by peer")
        assert is_retryable_error(exc) is True

    def test_invalid_api_key_not_retryable(self):
        """Invalid API key is not retryable."""
        exc = Exception("Invalid API key")
        assert is_retryable_error(exc) is False

    def test_bad_request_not_retryable(self):
        """400 Bad Request is not retryable."""
        exc = Exception("HTTP 400 Bad Request: malformed JSON")
        assert is_retryable_error(exc) is False


# =============================================================================
# Retry With Exponential Backoff Tests
# =============================================================================


class TestRetryWithExponentialBackoff:
    """Tests for retry_with_exponential_backoff() function."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        """Succeeds immediately when no error occurs."""
        async_func = AsyncMock(return_value="success")

        result = await retry_with_exponential_backoff(async_func)

        assert result == "success"
        assert async_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        """Retries on rate limit error."""
        async_func = AsyncMock(side_effect=[
            Exception("429 Too Many Requests"),
            Exception("429 Too Many Requests"),
            "success",
        ])

        with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_exponential_backoff(
                async_func,
                max_retries=3,
                base_delay=0.01,
            )

        assert result == "success"
        assert async_func.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Raises exception after exhausting retries."""
        async_func = AsyncMock(side_effect=Exception("429 Rate Limit"))

        with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="429 Rate Limit"):
                await retry_with_exponential_backoff(
                    async_func,
                    max_retries=2,
                    base_delay=0.01,
                )

        # Initial attempt + 2 retries = 3 calls
        assert async_func.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_immediately_on_non_retryable_error(self):
        """Does not retry on non-retryable errors."""
        async_func = AsyncMock(side_effect=Exception("Invalid API key"))

        with pytest.raises(Exception, match="Invalid API key"):
            await retry_with_exponential_backoff(async_func, max_retries=5)

        # Should only try once
        assert async_func.call_count == 1


# =============================================================================
# Anthropic Provider Tests
# =============================================================================


class TestAnthropicProviderInit:
    """Tests for AnthropicProvider initialization."""

    def test_init_with_api_key(self):
        """Uses provided API key."""
        provider = AnthropicProvider(api_key="sk-test-key")
        assert provider.api_key == "sk-test-key"

    def test_init_without_api_key_uses_env(self):
        """Falls back to environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-env-key"}):
            provider = AnthropicProvider()
            assert provider.api_key == "sk-env-key"

    def test_init_with_model(self):
        """Uses provided model."""
        provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-0-20250514")
        assert provider.model == "claude-opus-4-0-20250514"

    def test_init_default_model(self):
        """Uses default model when not specified."""
        provider = AnthropicProvider(api_key="sk-test")
        assert provider.model == "claude-haiku-4-5-20251001"


class TestAnthropicProviderComplete:
    """Tests for AnthropicProvider.complete() method."""

    @pytest.mark.asyncio
    async def test_complete_handles_timeout(self):
        """Handles timeout gracefully."""
        provider = AnthropicProvider(api_key="sk-test")

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=asyncio.TimeoutError("Request timed out")
            )
            mock_get_client.return_value = mock_client

            with pytest.raises(asyncio.TimeoutError):
                await provider.complete([{"role": "user", "content": "Hello"}])

    @pytest.mark.asyncio
    async def test_complete_handles_api_error(self):
        """Handles API errors gracefully."""
        provider = AnthropicProvider(api_key="sk-test")

        with patch.object(provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(
                side_effect=Exception("API Error: Invalid request")
            )
            mock_get_client.return_value = mock_client

            with pytest.raises(Exception, match="Invalid request"):
                await provider.complete([{"role": "user", "content": "Hello"}])


class TestAnthropicProviderNormalizeResponse:
    """Tests for _normalize_response() method."""

    def test_normalize_response_with_text(self):
        """Normalizes text response correctly."""
        provider = AnthropicProvider(api_key="sk-test")

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello, world!"
        mock_response.content = [mock_text_block]

        result = provider._normalize_response(mock_response)

        assert result["content"] == "Hello, world!"
        assert "tool_use" not in result

    def test_normalize_response_with_tool_use(self):
        """Normalizes tool use response correctly."""
        provider = AnthropicProvider(api_key="sk-test")

        mock_response = MagicMock()
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool-123"
        mock_tool_block.name = "read_file"
        mock_tool_block.input = {"path": "docs/api.md"}
        mock_response.content = [mock_tool_block]

        result = provider._normalize_response(mock_response)

        assert len(result["tool_use"]) == 1
        assert result["tool_use"][0]["name"] == "read_file"
        assert result["tool_use"][0]["id"] == "tool-123"

    def test_normalize_response_with_text_and_tool(self):
        """Normalizes mixed text and tool response."""
        provider = AnthropicProvider(api_key="sk-test")

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Let me read that file."
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool-456"
        mock_tool_block.name = "read_file"
        mock_tool_block.input = {"path": "README.md"}
        mock_response.content = [mock_text_block, mock_tool_block]

        result = provider._normalize_response(mock_response)

        assert result["content"] == "Let me read that file."
        assert len(result["tool_use"]) == 1


# =============================================================================
# Image and PDF Parsing Tests
# =============================================================================


class TestParseImageData:
    """Tests for parse_image_data() function."""

    def test_parse_image_data_valid(self):
        """Parses valid image data marker."""
        content = """__IMAGE_DATA__
media_type: image/png
prompt: What is in this image?
data: base64encodeddata
__END_IMAGE_DATA__"""

        media_type, prompt, data, remaining = parse_image_data(content)

        assert media_type == "image/png"
        assert prompt == "What is in this image?"
        assert data == "base64encodeddata"
        assert remaining == ""

    def test_parse_image_data_with_surrounding_text(self):
        """Parses image data with surrounding text."""
        content = """Some text before
__IMAGE_DATA__
media_type: image/jpeg
prompt: Analyze this
data: imagedata123
__END_IMAGE_DATA__
Some text after"""

        media_type, prompt, data, remaining = parse_image_data(content)

        assert media_type == "image/jpeg"
        assert "Some text before" in remaining
        assert "Some text after" in remaining

    def test_parse_image_data_no_marker(self):
        """Returns None when no image marker found."""
        content = "Just regular text without image data"

        media_type, prompt, data, remaining = parse_image_data(content)

        assert media_type is None
        assert prompt is None
        assert data is None
        assert remaining == content


class TestParsePdfData:
    """Tests for parse_pdf_data() function."""

    def test_parse_pdf_data_valid(self):
        """Parses valid PDF data marker."""
        content = """__PDF_DATA__
media_type: application/pdf
path: docs/manual.pdf
data: pdfbase64data
__END_PDF_DATA__"""

        media_type, path, data, remaining = parse_pdf_data(content)

        assert media_type == "application/pdf"
        assert path == "docs/manual.pdf"
        assert data == "pdfbase64data"
        assert remaining == ""

    def test_parse_pdf_data_no_marker(self):
        """Returns None when no PDF marker found."""
        content = "Just regular text without PDF data"

        media_type, path, data, remaining = parse_pdf_data(content)

        assert media_type is None
        assert path is None
        assert data is None
        assert remaining == content
