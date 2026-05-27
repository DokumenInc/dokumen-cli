"""Tests for retry module with exponential backoff."""

import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock

from dokumen.providers.retry import (
    is_rate_limit_error,
    is_retryable_error,
    retry_with_exponential_backoff,
    with_retry,
    RetryBudgetExhausted,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_EXPONENTIAL_BASE,
)


class TestIsRateLimitError:
    """Tests for is_rate_limit_error function."""

    def test_429_status_code(self):
        """Should detect 429 status code."""
        exc = Exception("HTTP 429 Too Many Requests")
        assert is_rate_limit_error(exc) is True

    def test_rate_limit_in_message(self):
        """Should detect 'rate_limit' in message."""
        exc = Exception("rate_limit exceeded")
        assert is_rate_limit_error(exc) is True

    def test_rate_limit_with_space(self):
        """Should detect 'rate limit' with space."""
        exc = Exception("Rate limit exceeded, try again later")
        assert is_rate_limit_error(exc) is True

    def test_ratelimit_exception_type(self):
        """Should detect RateLimit in exception type name."""
        class RateLimitError(Exception):
            pass

        exc = RateLimitError("too many requests")
        assert is_rate_limit_error(exc) is True

    def test_non_rate_limit_returns_false(self):
        """Should return False for non-rate-limit errors."""
        exc = Exception("Connection refused")
        assert is_rate_limit_error(exc) is False

    def test_empty_message(self):
        """Should handle empty exception message."""
        exc = Exception("")
        assert is_rate_limit_error(exc) is False


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_rate_limit_is_retryable(self):
        """Rate limit errors should be retryable."""
        exc = Exception("429 Too Many Requests")
        assert is_retryable_error(exc) is True

    def test_500_is_retryable(self):
        """HTTP 500 should be retryable."""
        exc = Exception("HTTP 500 Internal Server Error")
        assert is_retryable_error(exc) is True

    def test_502_is_retryable(self):
        """HTTP 502 should be retryable."""
        exc = Exception("502 Bad Gateway")
        assert is_retryable_error(exc) is True

    def test_503_is_retryable(self):
        """HTTP 503 should be retryable."""
        exc = Exception("503 Service Unavailable")
        assert is_retryable_error(exc) is True

    def test_504_is_retryable(self):
        """HTTP 504 should be retryable."""
        exc = Exception("504 Gateway Timeout")
        assert is_retryable_error(exc) is True

    def test_overloaded_is_retryable(self):
        """Overloaded errors should be retryable."""
        exc = Exception("Server is overloaded")
        assert is_retryable_error(exc) is True

    def test_timeout_is_retryable(self):
        """Timeout errors should be retryable."""
        exc = Exception("Request timed out")
        assert is_retryable_error(exc) is True

        exc2 = Exception("Connection timeout")
        assert is_retryable_error(exc2) is True

    def test_connection_reset_is_retryable(self):
        """Connection reset should be retryable."""
        exc = Exception("Connection reset by peer")
        assert is_retryable_error(exc) is True

    def test_connection_refused_is_retryable(self):
        """Connection refused should be retryable."""
        exc = Exception("Connection refused")
        assert is_retryable_error(exc) is True

    def test_connection_aborted_is_retryable(self):
        """Connection aborted should be retryable."""
        exc = Exception("Connection aborted")
        assert is_retryable_error(exc) is True

    def test_interrupted_is_retryable(self):
        """Interrupted errors should be retryable."""
        exc = Exception("Request interrupted")
        assert is_retryable_error(exc) is True

    def test_other_errors_not_retryable(self):
        """Other errors should not be retryable."""
        exc = Exception("Invalid API key")
        assert is_retryable_error(exc) is False

        exc2 = ValueError("Bad request format")
        assert is_retryable_error(exc2) is False

    def test_404_not_retryable(self):
        """HTTP 404 should not be retryable."""
        exc = Exception("404 Not Found")
        assert is_retryable_error(exc) is False


class TestRetryWithExponentialBackoff:
    """Tests for retry_with_exponential_backoff function."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        """Should return result on first successful attempt."""
        mock_func = AsyncMock(return_value="success")

        result = await retry_with_exponential_backoff(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        """Should succeed after retrying on retryable error."""
        mock_func = AsyncMock(side_effect=[
            Exception("429 Rate Limit"),
            "success"
        ])

        with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_exponential_backoff(mock_func, max_retries=2)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self):
        """Should raise non-retryable errors immediately without retry."""
        mock_func = AsyncMock(side_effect=ValueError("Invalid input"))

        with pytest.raises(ValueError, match="Invalid input"):
            await retry_with_exponential_backoff(mock_func, max_retries=3)

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausts_max_retries(self):
        """Should raise after exhausting max retries."""
        mock_func = AsyncMock(side_effect=Exception("429 Rate Limit"))

        with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="429 Rate Limit"):
                await retry_with_exponential_backoff(mock_func, max_retries=2)

        # Initial attempt + 2 retries = 3 total calls
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_delay_calculation(self):
        """Should calculate exponential backoff delays."""
        mock_func = AsyncMock(side_effect=[
            Exception("503 Service Unavailable"),
            Exception("503 Service Unavailable"),
            "success"
        ])
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        with patch("dokumen.providers.retry.asyncio.sleep", side_effect=mock_sleep):
            with patch("dokumen.providers.retry.random.random", return_value=0.5):
                # With jitter=True and random()=0.5, multiplier is 1.0 (0.5 + 0.5)
                await retry_with_exponential_backoff(
                    mock_func,
                    max_retries=3,
                    base_delay=1.0,
                    exponential_base=2,
                    jitter=True
                )

        # First retry: 1.0 * (2^0) * 1.0 = 1.0
        # Second retry: 1.0 * (2^1) * 1.0 = 2.0
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Should cap delay at max_delay."""
        mock_func = AsyncMock(side_effect=[
            Exception("429"),
            Exception("429"),
            Exception("429"),
            Exception("429"),
            "success"
        ])
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        with patch("dokumen.providers.retry.asyncio.sleep", side_effect=mock_sleep):
            with patch("dokumen.providers.retry.random.random", return_value=0.5):
                await retry_with_exponential_backoff(
                    mock_func,
                    max_retries=5,
                    base_delay=10.0,
                    max_delay=25.0,
                    exponential_base=2,
                    jitter=True
                )

        # Attempt 0: 10 * (2^0) = 10
        # Attempt 1: 10 * (2^1) = 20
        # Attempt 2: 10 * (2^2) = 40 -> capped at 25
        # Attempt 3: 10 * (2^3) = 80 -> capped at 25
        assert sleep_calls[0] == 10.0
        assert sleep_calls[1] == 20.0
        assert sleep_calls[2] == 25.0  # Capped
        assert sleep_calls[3] == 25.0  # Capped

    @pytest.mark.asyncio
    async def test_jitter_disabled(self):
        """Should use exact delay when jitter is disabled."""
        mock_func = AsyncMock(side_effect=[
            Exception("429"),
            "success"
        ])
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        with patch("dokumen.providers.retry.asyncio.sleep", side_effect=mock_sleep):
            await retry_with_exponential_backoff(
                mock_func,
                max_retries=2,
                base_delay=5.0,
                jitter=False
            )

        assert sleep_calls[0] == 5.0  # Exact delay, no jitter

    @pytest.mark.asyncio
    async def test_args_passed_through(self):
        """Should pass positional args to function."""
        mock_func = AsyncMock(return_value="done")

        await retry_with_exponential_backoff(mock_func, "arg1", "arg2")

        mock_func.assert_called_once_with("arg1", "arg2")

    @pytest.mark.asyncio
    async def test_kwargs_passed_through(self):
        """Should pass keyword args to function."""
        mock_func = AsyncMock(return_value="done")

        await retry_with_exponential_backoff(mock_func, key="value", other=123)

        mock_func.assert_called_once_with(key="value", other=123)


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Decorator should wrap async function."""
        @with_retry(max_retries=1)
        async def my_func():
            return "result"

        result = await my_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        """Decorator should preserve function name."""
        @with_retry()
        async def named_function():
            pass

        assert named_function.__name__ == "named_function"

    @pytest.mark.asyncio
    async def test_decorator_preserves_docstring(self):
        """Decorator should preserve docstring."""
        @with_retry()
        async def documented_function():
            """This is the docstring."""
            pass

        assert documented_function.__doc__ == "This is the docstring."

    @pytest.mark.asyncio
    async def test_decorator_retries_on_error(self):
        """Decorator should retry on retryable errors."""
        call_count = 0

        @with_retry(max_retries=2)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("503 Service Unavailable")
            return "success"

        with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await flaky_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_custom_config(self):
        """Decorator should use custom retry configuration."""
        @with_retry(max_retries=1, base_delay=0.1, max_delay=0.5)
        async def quick_retry_func():
            raise Exception("429")

        with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(Exception):
                await quick_retry_func()

        # Should have slept once with base_delay (may have jitter)
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_passes_args(self):
        """Decorator should pass arguments to function."""
        @with_retry()
        async def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await func_with_args("x", "y", c="z")
        assert result == "x-y-z"


class TestDefaultConstants:
    """Tests for default constant values."""

    def test_default_max_retries(self):
        """Default max retries should be 5."""
        assert DEFAULT_MAX_RETRIES == 5

    def test_default_base_delay(self):
        """Default base delay should be 1.0 seconds."""
        assert DEFAULT_BASE_DELAY == 1.0

    def test_default_max_delay(self):
        """Default max delay should be 60.0 seconds."""
        assert DEFAULT_MAX_DELAY == 60.0

    def test_default_exponential_base(self):
        """Default exponential base should be 2."""
        assert DEFAULT_EXPONENTIAL_BASE == 2


class TestRetryBudgetExhausted:
    """Tests for RetryBudgetExhausted exception."""

    def test_exception_has_required_attributes(self):
        """RetryBudgetExhausted has all 5 required attributes."""
        exc = RetryBudgetExhausted(
            attempts_made=3,
            rate_limit_hits=2,
            total_sleep_time=15.2,
            remaining_budget=2.1,
            last_error=Exception("429 Rate Limit"),
        )

        assert exc.attempts_made == 3
        assert exc.rate_limit_hits == 2
        assert exc.total_sleep_time == 15.2
        assert exc.remaining_budget == 2.1
        assert isinstance(exc.last_error, Exception)

    def test_exception_str_is_descriptive(self):
        """str(RetryBudgetExhausted) contains rate limit info."""
        exc = RetryBudgetExhausted(
            attempts_made=3,
            rate_limit_hits=2,
            total_sleep_time=15.2,
            remaining_budget=2.1,
            last_error=Exception("429 Rate Limit"),
        )

        msg = str(exc)
        assert "budget" in msg.lower() or "exhausted" in msg.lower()
        assert "2" in msg  # rate_limit_hits
        assert "15.2" in msg  # total_sleep_time

    def test_exception_is_exception_subclass(self):
        """RetryBudgetExhausted is a subclass of Exception."""
        exc = RetryBudgetExhausted(
            attempts_made=1,
            rate_limit_hits=1,
            total_sleep_time=1.0,
            remaining_budget=0.5,
            last_error=Exception("429"),
        )
        assert isinstance(exc, Exception)


class TestRetryWithDeadline:
    """Tests for deadline parameter in retry_with_exponential_backoff."""

    @pytest.mark.asyncio
    async def test_deadline_raises_when_sleep_exceeds_remaining(self):
        """Should raise RetryBudgetExhausted when next sleep would exceed deadline."""
        mock_func = AsyncMock(side_effect=Exception("429 Rate Limit"))

        # Simulate: current time=100, deadline=102 (2s remaining), delay=3s
        monotonic_values = iter([100.0, 100.0, 100.0])

        with patch("dokumen.providers.retry.time.monotonic", side_effect=monotonic_values):
            with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RetryBudgetExhausted) as exc_info:
                    await retry_with_exponential_backoff(
                        mock_func,
                        max_retries=5,
                        base_delay=3.0,
                        jitter=False,
                        deadline=102.0,
                    )

        assert exc_info.value.rate_limit_hits >= 1
        assert exc_info.value.attempts_made >= 1

    @pytest.mark.asyncio
    async def test_deadline_allows_retry_when_sufficient_time(self):
        """Should succeed when deadline has sufficient remaining time."""
        mock_func = AsyncMock(side_effect=[
            Exception("429 Rate Limit"),
            "success",
        ])

        # Simulate: plenty of time remaining
        monotonic_values = iter([100.0, 100.5, 101.0, 101.5])

        with patch("dokumen.providers.retry.time.monotonic", side_effect=monotonic_values):
            with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
                result = await retry_with_exponential_backoff(
                    mock_func,
                    max_retries=5,
                    base_delay=1.0,
                    jitter=False,
                    deadline=200.0,  # far in the future
                )

        assert result == "success"

    @pytest.mark.asyncio
    async def test_deadline_none_preserves_original_behavior(self):
        """deadline=None should behave identically to original (no budget check)."""
        mock_func = AsyncMock(side_effect=[
            Exception("429 Rate Limit"),
            "success",
        ])

        with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_exponential_backoff(
                mock_func,
                max_retries=5,
                deadline=None,
            )

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_deadline_expired_before_first_retry(self):
        """Should raise immediately when deadline is already passed."""
        mock_func = AsyncMock(side_effect=Exception("429 Rate Limit"))

        # Deadline already in the past
        with patch("dokumen.providers.retry.time.monotonic", return_value=200.0):
            with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RetryBudgetExhausted) as exc_info:
                    await retry_with_exponential_backoff(
                        mock_func,
                        max_retries=5,
                        deadline=100.0,  # already expired
                    )

        assert exc_info.value.remaining_budget <= 0

    @pytest.mark.asyncio
    async def test_deadline_tracks_rate_limit_vs_other_errors(self):
        """Should track rate_limit_hits separately from total attempts."""
        mock_func = AsyncMock(side_effect=[
            Exception("503 Service Unavailable"),  # retryable, not rate limit
            Exception("429 Rate Limit"),            # rate limit
            Exception("429 Rate Limit"),            # rate limit - should trigger budget exceeded
        ])

        # Enough time for first retry but not enough for the second retry's sleep
        monotonic_calls = iter([100.0, 100.5, 101.0, 101.5, 102.0])

        with patch("dokumen.providers.retry.time.monotonic", side_effect=monotonic_calls):
            with patch("dokumen.providers.retry.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RetryBudgetExhausted) as exc_info:
                    await retry_with_exponential_backoff(
                        mock_func,
                        max_retries=5,
                        base_delay=1.0,
                        exponential_base=2,
                        jitter=False,
                        deadline=103.0,  # 3s budget total
                    )

        # At least 1 rate limit hit counted
        assert exc_info.value.rate_limit_hits >= 1
        # Total attempts includes the non-rate-limit error too
        assert exc_info.value.attempts_made >= 2
