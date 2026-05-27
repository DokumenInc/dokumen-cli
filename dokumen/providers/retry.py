"""
Retry utilities with exponential backoff for API rate limiting.
"""
import asyncio
import random
import logging
import time
from functools import wraps
from typing import Callable, Optional, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Default retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2


class RetryBudgetExhausted(Exception):
    """Raised when retry backoff would exceed the remaining deadline budget.

    This exception provides detailed context about why the retry loop stopped,
    making it possible for callers (like the explore agent) to produce verbose,
    actionable error messages instead of generic timeout errors.
    """

    def __init__(
        self,
        attempts_made: int,
        rate_limit_hits: int,
        total_sleep_time: float,
        remaining_budget: float,
        last_error: Exception,
    ):
        self.attempts_made = attempts_made
        self.rate_limit_hits = rate_limit_hits
        self.total_sleep_time = total_sleep_time
        self.remaining_budget = remaining_budget
        self.last_error = last_error
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"Retry budget exhausted: {self.rate_limit_hits} rate limit retries "
            f"consumed {self.total_sleep_time:.1f}s, only {self.remaining_budget:.1f}s "
            f"remaining ({self.attempts_made} total attempts)"
        )


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    error_str = str(exception).lower()

    # Check for common rate limit indicators
    if '429' in str(exception) or 'rate_limit' in error_str or 'rate limit' in error_str:
        return True

    # Check for specific exception types
    exception_type = type(exception).__name__.lower()
    if 'ratelimit' in exception_type:
        return True

    return False


def is_retryable_error(exception: Exception) -> bool:
    """Check if an exception is retryable (rate limit, timeout, or temporary server error)."""
    if is_rate_limit_error(exception):
        return True

    error_str = str(exception).lower()

    # Also retry on temporary server errors (5xx)
    if '500' in error_str or '502' in error_str or '503' in error_str or '504' in error_str:
        return True

    # Check for overloaded errors
    if 'overloaded' in error_str:
        return True

    # Check for timeout errors (network timeouts, API timeouts)
    if 'timed out' in error_str or 'timeout' in error_str:
        return True

    # Check for connection errors that are usually temporary
    if 'connection' in error_str and ('reset' in error_str or 'refused' in error_str or 'aborted' in error_str):
        return True

    # Check for interrupted requests
    if 'interrupted' in error_str:
        return True

    return False


async def retry_with_exponential_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    jitter: bool = True,
    deadline: Optional[float] = None,
    **kwargs
) -> T:
    """
    Execute an async function with exponential backoff on rate limit errors.

    Args:
        func: Async function to execute
        *args: Positional arguments to pass to func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        deadline: Optional absolute time.monotonic() timestamp. When set,
                  the retry loop will raise RetryBudgetExhausted instead of
                  sleeping if the next delay would exceed the remaining budget.
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries are exhausted
        RetryBudgetExhausted if deadline would be exceeded by next sleep
    """
    last_exception = None
    rate_limit_hits = 0
    total_sleep_time = 0.0

    logger.debug(f"[retry] Starting retry loop for {func.__name__} (max {max_retries} retries)")

    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"[retry] Attempt {attempt + 1}/{max_retries + 1} for {func.__name__}")
            result = await func(*args, **kwargs)
            logger.debug(f"[retry] {func.__name__} succeeded on attempt {attempt + 1}")
            return result
        except Exception as e:
            last_exception = e

            if not is_retryable_error(e):
                # Not a retryable error, raise immediately
                raise

            # Classify the error
            if is_rate_limit_error(e):
                error_class = "rate_limit"
                rate_limit_hits += 1
            elif 'timeout' in str(e).lower() or 'timed out' in str(e).lower():
                error_class = "timeout"
            elif any(code in str(e) for code in ('500', '502', '503', '504')):
                error_class = "server_error"
            else:
                error_class = "other"

            if attempt == max_retries:
                # Exhausted all retries
                logger.warning(
                    "Retry exhausted",
                    extra={
                        "func": func.__name__,
                        "max_retries": max_retries,
                        "error_class": error_class,
                        "error": str(e),
                    },
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random())

            # Check deadline budget before sleeping
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or delay >= remaining:
                    logger.warning(
                        "retry.budget_will_exceed_deadline",
                        extra={
                            "delay": delay,
                            "remaining": remaining,
                            "attempts_made": attempt + 1,
                            "rate_limit_hits": rate_limit_hits,
                            "total_sleep_time": total_sleep_time,
                        }
                    )
                    raise RetryBudgetExhausted(
                        attempts_made=attempt + 1,
                        rate_limit_hits=rate_limit_hits,
                        total_sleep_time=total_sleep_time,
                        remaining_budget=remaining,
                        last_error=e,
                    )

            logger.info(
                "Retrying after error",
                extra={
                    "func": func.__name__,
                    "error_class": error_class,
                    "attempt": attempt + 1,
                    "max_retries": max_retries + 1,
                    "wait_seconds": round(delay, 1),
                    "error": str(e),
                },
            )

            await asyncio.sleep(delay)
            total_sleep_time += delay

    # Should never reach here, but just in case
    raise last_exception


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    jitter: bool = True
):
    """
    Decorator to add exponential backoff retry to an async function.

    Usage:
        @with_retry(max_retries=5)
        async def my_api_call():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_with_exponential_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                **kwargs
            )
        return wrapper
    return decorator
