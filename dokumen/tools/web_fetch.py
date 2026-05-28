"""
Upgraded web_fetch tool with HTML→markdown, caching, summarization, and SSRF protection.

Replaces the bare-bones create_http_request_tool for forward-looking usage.
The old function in tools_object.py stays for backward compat.
"""

from __future__ import annotations

import asyncio
import importlib.util
import ipaddress
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

from ..logging_config import get_logger
from .types import ToolDefinition, ToolResult

if TYPE_CHECKING:
    from ..sandbox import Sandbox

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# preapproved dev/docs domains — skip summarization if content already small
# ---------------------------------------------------------------------------
DEFAULT_PREAPPROVED_HOSTS: Set[str] = {
    "docs.python.org",
    "docs.djangoproject.com",
    "fastapi.tiangolo.com",
    "flask.palletsprojects.com",
    "docs.sqlalchemy.org",
    "pydantic-docs.helpmanual.io",
    "docs.pydantic.dev",
    "packaging.python.org",
    "setuptools.pypa.io",
    "pip.pypa.io",
    "react.dev",
    "reactjs.org",
    "nextjs.org",
    "vuejs.org",
    "svelte.dev",
    "developer.mozilla.org",
    "html.spec.whatwg.org",
    "w3.org",
    "www.w3.org",
    "docs.npmjs.com",
    "nodejs.org",
    "deno.land",
    "bun.sh",
    "vitejs.dev",
    "esbuild.github.io",
    "typescriptlang.org",
    "www.typescriptlang.org",
    "jestjs.io",
    "vitest.dev",
    "testing-library.com",
    "docs.github.com",
    "kubernetes.io",
    "docs.docker.com",
    "docs.aws.amazon.com",
    "cloud.google.com",
    "docs.microsoft.com",
    "learn.microsoft.com",
    "developer.apple.com",
    "dev.to",
    "stackoverflow.com",
    "en.wikipedia.org",
}

# ---------------------------------------------------------------------------
# SSRF blocklist (mirrors tools_object.py)
# ---------------------------------------------------------------------------
_BLOCKED_HOSTNAMES: Set[str] = {
    "localhost",
    "localhost.localdomain",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "169.254.169.254",  # aws/gcp/azure metadata
    "metadata.google.internal",
    "metadata.google",
    "169.254.170.2",  # aws ecs metadata
}


# ---------------------------------------------------------------------------
# config dataclass
# ---------------------------------------------------------------------------
@dataclass
class WebFetchConfig:
    """Runtime configuration for WebFetcher."""

    cache_ttl: int = 900  # 15 minutes in seconds
    max_cache_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_content_chars: int = 100_000
    max_url_length: int = 2000
    timeout: float = 30.0
    max_redirects: int = 10
    preapproved_hosts: Set[str] = field(default_factory=lambda: set(DEFAULT_PREAPPROVED_HOSTS))


# ---------------------------------------------------------------------------
# protocol: content summarizer
# ---------------------------------------------------------------------------
class ContentSummarizer:
    """Protocol for summarizing fetched content against a prompt.

    Implement this and pass it to WebFetcher to enable prompt-on-content.
    The interface is intentionally thin so a Redis-backed or remote version
    can be swapped in later (rule 2.6).
    """

    async def summarize(self, content: str, prompt: str) -> str:
        raise NotImplementedError


class ProviderSummarizer(ContentSummarizer):
    """model-agnostic summarizer using any dokumen Provider.

    uses the existing Provider ABC (agent_object.py) so it works with
    anthropic, mock, or any future provider. defaults to haiku but
    accepts any model string.

    usage:
        from dokumen.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
        summarizer = ProviderSummarizer(provider)
        fetcher = WebFetcher(summarizer=summarizer)
    """

    SYSTEM_PROMPT = (
        "you are a web content extraction agent. given a web page's content "
        "and a user prompt, return ONLY the information the prompt asks for. "
        "be concise. no preamble. if the content doesn't contain what's asked, "
        "say so briefly."
    )

    def __init__(self, provider: Any, max_content_chars: int = 80_000) -> None:
        self._provider = provider
        self._max_chars = max_content_chars

    async def summarize(self, content: str, prompt: str) -> str:
        # truncate content before sending to model
        if len(content) > self._max_chars:
            content = content[: self._max_chars] + "\n\n[truncated]"

        user_msg = f"web page content:\n---\n{content}\n---\n\n{prompt}"

        logger.info(
            "provider_summarizer.call",
            extra={"prompt_len": len(prompt), "content_len": len(content)},
        )

        result = await self._provider.complete(
            messages=[{"role": "user", "content": user_msg}],
            system_prompt=self.SYSTEM_PROMPT,
        )

        # provider.complete returns dict with 'content' key
        if isinstance(result, dict):
            return result.get("content", str(result))
        return str(result)


# ---------------------------------------------------------------------------
# result dataclass
# ---------------------------------------------------------------------------
@dataclass
class WebFetchResult:
    """Structured result from a web fetch operation."""

    url: str
    status_code: int
    content: str
    content_type: str
    byte_size: int
    from_cache: bool
    duration_ms: float


# ---------------------------------------------------------------------------
# cache entry + store (protocol-backed per rule 2.6)
# ---------------------------------------------------------------------------
@dataclass
class _CacheEntry:
    url: str
    content: str
    status_code: int
    content_type: str
    byte_size: int
    fetched_at: float  # unix timestamp


class WebFetchCacheStore:
    """Protocol for the web fetch LRU+TTL cache.

    Current implementation is in-memory. Swap to Redis by subclassing and
    overriding get/set/delete.
    """

    def get(self, key: str) -> Optional[_CacheEntry]: ...
    def set(self, key: str, entry: _CacheEntry) -> None: ...
    def delete(self, key: str) -> None: ...
    def total_bytes(self) -> int: ...


class InMemoryWebFetchCache(WebFetchCacheStore):
    """In-memory LRU cache with TTL and byte-budget enforcement."""

    def __init__(self, ttl: int, max_bytes: int) -> None:
        self._ttl = ttl
        self._max_bytes = max_bytes
        # ordered dict preserves insertion order for LRU eviction
        self._store: Dict[str, _CacheEntry] = {}
        self._access_order: List[str] = []

    def _touch(self, key: str) -> None:
        # move to end (most recently used)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def get(self, key: str) -> Optional[_CacheEntry]:
        entry = self._store.get(key)
        if entry is None:
            return None
        # ttl check
        if time.monotonic() - entry.fetched_at > self._ttl:
            self.delete(key)
            return None
        self._touch(key)
        return entry

    def set(self, key: str, entry: _CacheEntry) -> None:
        # overwrite existing
        if key in self._store:
            self.delete(key)
        self._store[key] = entry
        self._access_order.append(key)
        # evict until under byte budget
        while self.total_bytes() > self._max_bytes and self._access_order:
            oldest = self._access_order[0]
            self.delete(oldest)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def total_bytes(self) -> int:
        return sum(e.byte_size for e in self._store.values())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_private_ip(ip_str: str) -> bool:
    """return true if ip is private/loopback/reserved."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
        )
    except ValueError:
        return False


def _is_blocked_host(hostname: str) -> tuple[bool, str]:
    """check if a hostname should be blocked for SSRF protection."""
    if not hostname:
        return True, "empty hostname"

    hostname_lower = hostname.lower()

    if hostname_lower in _BLOCKED_HOSTNAMES:
        return True, f"blocked hostname: {hostname}"

    if _is_private_ip(hostname):
        return True, f"private/internal IP address not allowed: {hostname}"

    try:
        resolved_ips = socket.gethostbyname_ex(hostname)[2]
        for ip in resolved_ips:
            if _is_private_ip(ip):
                return True, f"hostname resolves to private IP: {hostname} -> {ip}"
    except socket.gaierror:
        # dns failure — let the http layer handle it
        pass

    return False, ""


def _validate_url(url: str, max_length: int) -> Optional[str]:
    """validate url and return error string or None if ok."""
    if len(url) > max_length:
        return f"URL exceeds {max_length} character limit"

    try:
        parsed = urlparse(url)
    except Exception as exc:
        return f"invalid URL: {exc}"

    if parsed.scheme not in ("http", "https"):
        return f"invalid URL scheme '{parsed.scheme}': only http and https are allowed"

    if parsed.username or parsed.password:
        return "credentials in URL are not allowed"

    hostname = parsed.hostname or ""
    parts = [p for p in hostname.split(".") if p]
    if len(parts) < 2:
        return f"hostname '{hostname}' must have at least two parts"

    return None


def _upgrade_to_https(url: str) -> str:
    """silently upgrade http → https."""
    parsed = urlparse(url)
    if parsed.scheme == "http":
        upgraded = parsed._replace(scheme="https")
        return urlunparse(upgraded)
    return url


def _same_host(host_a: str, host_b: str) -> bool:
    """true if the two hostnames are the same modulo a www. prefix."""
    a = host_a.lower().lstrip("www.")
    b = host_b.lower().lstrip("www.")
    return a == b


def _html_to_markdown(html: str) -> str:
    """convert html to markdown, lazy-loading markdownify."""
    try:
        from markdownify import markdownify as md  # type: ignore

        return md(html, heading_style="ATX", strip=["script", "style"])
    except ImportError:
        logger.debug("markdownify not installed, returning raw html")
        return html


def _truncate(content: str, max_chars: int) -> str:
    """cap content at max_chars, appending a notice if truncated."""
    if len(content) <= max_chars:
        return content
    original_len = len(content)
    return content[:max_chars] + f"\n\n[content truncated from {original_len} chars]"


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------
class WebFetcher:
    """Fetches URLs with caching, html→markdown, optional summarization, and SSRF protection."""

    def __init__(
        self,
        config: Optional[WebFetchConfig] = None,
        cache: Optional[WebFetchCacheStore] = None,
        summarizer: Optional[ContentSummarizer] = None,
    ) -> None:
        self._config = config or WebFetchConfig()
        self._cache = cache or InMemoryWebFetchCache(
            ttl=self._config.cache_ttl,
            max_bytes=self._config.max_cache_bytes,
        )
        self._summarizer = summarizer
        logger.info(
            "WebFetcher initialised",
            extra={
                "cache_ttl": self._config.cache_ttl,
                "max_content_chars": self._config.max_content_chars,
                "has_summarizer": summarizer is not None,
            },
        )

    async def fetch(
        self,
        url: str,
        *,
        prompt: Optional[str] = None,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
    ) -> WebFetchResult:
        """fetch a url and return a WebFetchResult.

        if prompt is given and a summarizer is attached, the markdown content
        is summarized before returning, reducing token waste significantly.
        """
        logger.info("fetch.start", extra={"url": url, "method": method, "has_prompt": bool(prompt)})

        # --- upgrade scheme ---
        url = _upgrade_to_https(url)

        # --- validate ---
        err = _validate_url(url, self._config.max_url_length)
        if err:
            logger.warning("fetch.validation_failed", extra={"url": url, "reason": err})
            raise ValueError(err)

        parsed = urlparse(url)
        blocked, reason = _is_blocked_host(parsed.hostname or "")
        if blocked:
            logger.warning("fetch.ssrf_blocked", extra={"url": url, "reason": reason})
            raise ValueError(f"request blocked for security: {reason}")

        # --- cache lookup (only for GET with no body) ---
        cache_key = url
        if method.upper() == "GET" and not body:
            cached = self._cache.get(cache_key)
            if cached:
                logger.info("fetch.cache_hit", extra={"url": url})
                content = _truncate(cached.content, self._config.max_content_chars)
                if prompt:
                    content = await self._summarize(content, prompt, url)
                return WebFetchResult(
                    url=url,
                    status_code=cached.status_code,
                    content=content,
                    content_type=cached.content_type,
                    byte_size=cached.byte_size,
                    from_cache=True,
                    duration_ms=0.0,
                )

        # --- do the actual request ---
        start = time.monotonic()
        raw_content, status_code, content_type, final_url = await self._do_request(
            url, method=method, headers=headers or {}, body=body
        )
        duration_ms = (time.monotonic() - start) * 1000

        logger.info(
            "fetch.complete",
            extra={
                "url": url,
                "final_url": final_url,
                "status_code": status_code,
                "content_type": content_type,
                "raw_bytes": len(raw_content.encode("utf-8", errors="replace")),
                "duration_ms": round(duration_ms, 1),
            },
        )

        # --- html → markdown ---
        ct_lower = content_type.lower()
        if "html" in ct_lower:
            content = _html_to_markdown(raw_content)
        else:
            content = raw_content

        byte_size = len(content.encode("utf-8", errors="replace"))

        # --- store in cache ---
        if method.upper() == "GET" and not body and 200 <= status_code < 400:
            self._cache.set(
                cache_key,
                _CacheEntry(
                    url=url,
                    content=content,
                    status_code=status_code,
                    content_type=content_type,
                    byte_size=byte_size,
                    fetched_at=time.monotonic(),
                ),
            )

        # --- truncate ---
        content = _truncate(content, self._config.max_content_chars)

        # --- preapproved fast-path: skip summarization if already small ---
        hostname = (parsed.hostname or "").lower()
        is_preapproved = hostname in self._config.preapproved_hosts
        already_markdown = "markdown" in ct_lower or "text/plain" in ct_lower
        if is_preapproved and already_markdown and len(content) < self._config.max_content_chars:
            logger.debug("fetch.preapproved_fast_path", extra={"hostname": hostname})
        elif prompt:
            content = await self._summarize(content, prompt, url)

        return WebFetchResult(
            url=final_url,
            status_code=status_code,
            content=content,
            content_type=content_type,
            byte_size=byte_size,
            from_cache=False,
            duration_ms=duration_ms,
        )

    async def _summarize(self, content: str, prompt: str, url: str) -> str:
        """delegate to the summarizer if one is attached."""
        if self._summarizer is None:
            logger.debug("fetch.no_summarizer", extra={"url": url})
            return content
        logger.info("fetch.summarizing", extra={"url": url, "prompt_len": len(prompt)})
        try:
            result = await self._summarizer.summarize(content, prompt)
            logger.info(
                "fetch.summarized",
                extra={"url": url, "before": len(content), "after": len(result)},
            )
            return result
        except Exception as exc:
            logger.error(
                "fetch.summarize_error",
                extra={"url": url, "error": str(exc)},
                exc_info=True,
            )
            # fall back to full content
            return content

    async def _do_request(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[str],
    ) -> tuple[str, int, str, str]:
        """perform the actual http request, following same-host redirects.

        returns (content, status_code, content_type, final_url).
        cross-host redirects raise a ValueError telling the caller the new url.
        """
        if importlib.util.find_spec("aiohttp") is not None:
            return await self._do_request_aiohttp(url, method, headers, body)

        logger.debug("aiohttp not available, falling back to urllib")
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._do_request_urllib_sync,
            url,
            method,
            headers,
            body,
        )

    async def _do_request_aiohttp(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[str],
    ) -> tuple[str, int, str, str]:
        """aiohttp path with manual redirect handling."""
        import aiohttp

        original_host = urlparse(url).hostname or ""
        current_url = url
        hops = 0

        connector = aiohttp.TCPConnector(ssl=True)
        timeout = aiohttp.ClientTimeout(total=self._config.timeout)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            # disable auto-redirect so we can inspect location headers ourselves
        ) as session:
            while hops <= self._config.max_redirects:
                logger.debug("fetch.hop", extra={"hop": hops, "url": current_url})
                async with session.request(
                    method,
                    current_url,
                    headers=headers,
                    data=body,
                    allow_redirects=False,
                ) as response:
                    status = response.status
                    ct = response.headers.get("Content-Type", "text/plain")

                    if status in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location", "")
                        if not location:
                            # no location header — just read the body and stop
                            content = await response.text(errors="replace")
                            return content, status, ct, current_url

                        redirect_host = urlparse(location).hostname or ""
                        if _same_host(original_host, redirect_host):
                            hops += 1
                            current_url = location
                            # for POST→GET on 303
                            if status == 303:
                                method = "GET"
                                body = None
                            continue
                        else:
                            # cross-host redirect — tell the agent to retry
                            raise ValueError(f"cross-host redirect detected: retry with {location}")

                    content = await response.text(errors="replace")
                    return content, status, ct, current_url

        raise ValueError(f"too many redirects (>{self._config.max_redirects})")

    def _do_request_urllib_sync(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[str],
    ) -> tuple[str, int, str, str]:
        """synchronous urllib fallback (run in executor)."""
        import urllib.request
        import urllib.error

        original_host = urlparse(url).hostname or ""
        current_url = url
        hops = 0

        while hops <= self._config.max_redirects:
            req = urllib.request.Request(current_url, method=method)
            for k, v in headers.items():
                req.add_header(k, v)
            data = body.encode("utf-8") if body else None

            try:
                with urllib.request.urlopen(req, data=data, timeout=self._config.timeout) as resp:
                    ct = resp.headers.get("Content-Type", "text/plain")
                    content = resp.read().decode("utf-8", errors="replace")
                    return content, resp.status, ct, current_url
            except urllib.error.HTTPError as exc:
                if exc.code in (301, 302, 303, 307, 308):
                    location = exc.headers.get("Location", "")
                    if not location:
                        content = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                        return content, exc.code, "", current_url
                    redirect_host = urlparse(location).hostname or ""
                    if _same_host(original_host, redirect_host):
                        hops += 1
                        current_url = location
                        if exc.code == 303:
                            method = "GET"
                            body = None
                        continue
                    else:
                        raise ValueError(
                            f"cross-host redirect detected: retry with {location}"
                        ) from exc
                # non-redirect error
                ct = exc.headers.get("Content-Type", "text/plain") if exc.headers else "text/plain"
                content = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                return content, exc.code, ct, current_url

        raise ValueError(f"too many redirects (>{self._config.max_redirects})")


# ---------------------------------------------------------------------------
# tool factory
# ---------------------------------------------------------------------------
def create_web_fetch_tool(
    fetcher: WebFetcher,
    sandbox: Optional["Sandbox"] = None,
) -> ToolDefinition:
    """create a ToolDefinition that wraps WebFetcher, compatible with tools_object.py.

    sandbox is accepted for api symmetry with create_http_request_tool but is not
    currently used — fetcher handles its own transport.
    """

    async def handler(params: Dict[str, Any]) -> ToolResult:
        url = params.get("url")
        prompt = params.get("prompt") or None
        method = (params.get("method") or "GET").upper()
        headers = params.get("headers") or {}
        body = params.get("body") or None

        logger.info(
            "web_fetch_tool.called",
            extra={"url": url, "method": method, "has_prompt": bool(prompt)},
        )

        if not url:
            return ToolResult(success=False, output=None, error="missing 'url' parameter")

        try:
            result = await fetcher.fetch(
                url,
                prompt=prompt,
                method=method,
                headers=headers,
                body=body,
            )
        except ValueError as exc:
            msg = str(exc)
            # cross-host redirect — surface it clearly so the agent can retry
            if "cross-host redirect" in msg:
                logger.info("web_fetch_tool.cross_host_redirect", extra={"url": url, "detail": msg})
                return ToolResult(
                    success=False,
                    output={"redirect_message": msg},
                    error=msg,
                )
            logger.warning("web_fetch_tool.validation_error", extra={"url": url, "error": msg})
            return ToolResult(success=False, output=None, error=msg)
        except Exception as exc:
            logger.error(
                "web_fetch_tool.unexpected_error",
                extra={"url": url, "error": str(exc)},
                exc_info=True,
            )
            return ToolResult(success=False, output=None, error=str(exc))

        logger.info(
            "web_fetch_tool.success",
            extra={
                "url": result.url,
                "status_code": result.status_code,
                "from_cache": result.from_cache,
                "content_len": len(result.content),
                "duration_ms": round(result.duration_ms, 1),
            },
        )

        return ToolResult(
            success=200 <= result.status_code < 400,
            output={
                "url": result.url,
                "status_code": result.status_code,
                "content_type": result.content_type,
                "byte_size": result.byte_size,
                "from_cache": result.from_cache,
                "duration_ms": round(result.duration_ms, 1),
                "content": result.content,
            },
        )

    return ToolDefinition(
        name="web_fetch",
        description=(
            "Fetch a URL and return its content as markdown. "
            "Optionally provide a 'prompt' to extract only what you need, "
            "which significantly reduces token usage. "
            "Responses are cached for 15 minutes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "the URL to fetch",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "what to extract from the page (optional). "
                        "when provided, only the relevant section is returned, "
                        "reducing token usage significantly"
                    ),
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (default: GET)",
                    "default": "GET",
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP headers as key-value pairs",
                    "additionalProperties": {"type": "string"},
                },
                "body": {
                    "type": "string",
                    "description": "request body for POST/PUT",
                },
            },
            "required": ["url"],
        },
        handler=handler,
    )
