#!/usr/bin/env python3
"""standalone tests for improved web_fetch and compaction archive.

run: python3 tests/scripts/test_web_fetch_archive.py
"""
import sys
import os
import asyncio
import tempfile
import shutil
import time
import importlib.util

_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)

passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}")


def _load(name, rel_path):
    path = os.path.join(_root, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── load modules in dependency order ──

# logging_config needed by web_fetch
_log_mod = _load("dokumen.logging_config", "dokumen/logging_config.py")

# tools_object for ToolDefinition/ToolResult — has many deps, load it carefully
# we only need the dataclasses, so load types directly
_debug_mod = _load("dokumen.debug", "dokumen/debug.py")

# load tools_object
_tools_mod = _load("dokumen.tools_object", "dokumen/tools_object.py")

# load web_fetch
_wf_mod = _load("dokumen.tools.web_fetch", "dokumen/tools/web_fetch.py")

# load archive
_arch_mod = _load("dokumen.context.archive", "dokumen/context/archive.py")

# load compactor (needs archive)
_comp_mod = _load("dokumen.context.compactor", "dokumen/context/compactor.py")

loop = asyncio.new_event_loop()

# ═══════════════════════════════════════════
# web_fetch: config
# ═══════════════════════════════════════════

print("\n── web_fetch config ──")

WebFetchConfig = _wf_mod.WebFetchConfig
config = WebFetchConfig()
test("default cache_ttl is 900", config.cache_ttl == 900)
test("default max_content_chars is 100000", config.max_content_chars == 100_000)
test("default max_url_length is 2000", config.max_url_length == 2000)
test("default timeout is 30", config.timeout == 30.0)
test("default max_redirects is 10", config.max_redirects == 10)
test("preapproved hosts not empty", len(config.preapproved_hosts) > 20)

custom = WebFetchConfig(cache_ttl=60, max_content_chars=5000)
test("custom cache_ttl", custom.cache_ttl == 60)
test("custom max_content_chars", custom.max_content_chars == 5000)

# ═══════════════════════════════════════════
# web_fetch: URL validation
# ═══════════════════════════════════════════

print("\n── url validation ──")

_validate = _wf_mod._validate_url

test("valid https url", _validate("https://example.com/path", 2000) is None)
test("valid http url", _validate("http://example.com", 2000) is None)
test("url too long", _validate("https://example.com/" + "a" * 2000, 2000) is not None)
test("ftp scheme rejected", _validate("ftp://example.com", 2000) is not None)
test("no hostname parts", _validate("https://localhost/path", 2000) is not None)
test("credentials rejected", _validate("https://user:pass@example.com", 2000) is not None)
test("empty scheme rejected", _validate("://example.com", 2000) is not None)

# ═══════════════════════════════════════════
# web_fetch: SSRF protection
# ═══════════════════════════════════════════

print("\n── ssrf protection ──")

_is_blocked = _wf_mod._is_blocked_host

blocked, _ = _is_blocked("localhost")
test("localhost blocked", blocked)

blocked, _ = _is_blocked("127.0.0.1")
test("127.0.0.1 blocked", blocked)

blocked, _ = _is_blocked("169.254.169.254")
test("metadata endpoint blocked", blocked)

blocked, _ = _is_blocked("")
test("empty hostname blocked", blocked)

blocked, _ = _is_blocked("example.com")
test("example.com not blocked", not blocked)

blocked, _ = _is_blocked("10.0.0.1")
test("private ip blocked", blocked)

# ═══════════════════════════════════════════
# web_fetch: https upgrade
# ═══════════════════════════════════════════

print("\n── https upgrade ──")

_upgrade = _wf_mod._upgrade_to_https

test("http upgraded", _upgrade("http://example.com").startswith("https://"))
test("https unchanged", _upgrade("https://example.com") == "https://example.com")

# ═══════════════════════════════════════════
# web_fetch: same host check
# ═══════════════════════════════════════════

print("\n── same host check ──")

_same = _wf_mod._same_host

test("same host", _same("example.com", "example.com"))
test("www prefix", _same("www.example.com", "example.com"))
test("www both sides", _same("www.example.com", "www.example.com"))
test("different hosts", not _same("example.com", "other.com"))

# ═══════════════════════════════════════════
# web_fetch: html to markdown
# ═══════════════════════════════════════════

print("\n── html to markdown ──")

_h2m = _wf_mod._html_to_markdown

# this works whether or not markdownify is installed
result = _h2m("<p>hello world</p>")
test("html converted to something", len(result) > 0)
test("result contains hello", "hello" in result.lower())

# ═══════════════════════════════════════════
# web_fetch: truncation
# ═══════════════════════════════════════════

print("\n── truncation ──")

_trunc = _wf_mod._truncate

test("short content unchanged", _trunc("hello", 100) == "hello")
test("long content truncated", len(_trunc("a" * 200, 50)) < 200)
test("truncation notice", "truncated" in _trunc("a" * 200, 50))
test("truncation at limit", _trunc("a" * 100, 100) == "a" * 100)

# ═══════════════════════════════════════════
# web_fetch: LRU cache
# ═══════════════════════════════════════════

print("\n── lru cache ──")

CacheEntry = _wf_mod._CacheEntry
InMemoryCache = _wf_mod.InMemoryWebFetchCache

cache = InMemoryCache(ttl=10, max_bytes=1000)
entry = CacheEntry(url="https://x.com", content="hi", status_code=200, content_type="text/html", byte_size=2, fetched_at=time.monotonic())
cache.set("https://x.com", entry)

test("cache get returns entry", cache.get("https://x.com") is not None)
test("cache get content", cache.get("https://x.com").content == "hi")
test("cache miss", cache.get("https://nothere.com") is None)
test("cache total bytes", cache.total_bytes() == 2)

# eviction
big = CacheEntry(url="big", content="x" * 500, status_code=200, content_type="text/html", byte_size=500, fetched_at=time.monotonic())
cache.set("big1", big)
cache.set("big2", big)
# big1 + big2 = 1000, x.com should have been evicted
test("lru eviction happened", cache.total_bytes() <= 1000)

# ttl expiry
expired_cache = InMemoryCache(ttl=0, max_bytes=10000)
expired_entry = CacheEntry(url="e", content="old", status_code=200, content_type="t", byte_size=3, fetched_at=time.monotonic() - 1)
expired_cache._store["e"] = expired_entry
expired_cache._access_order.append("e")
test("expired entry returns None", expired_cache.get("e") is None)

# ═══════════════════════════════════════════
# web_fetch: WebFetcher class
# ═══════════════════════════════════════════

print("\n── WebFetcher ──")

WebFetcher = _wf_mod.WebFetcher
WebFetchResult = _wf_mod.WebFetchResult

fetcher = WebFetcher()
test("fetcher created", fetcher is not None)
test("fetcher has config", fetcher._config is not None)
test("fetcher has cache", fetcher._cache is not None)

# test with a real url would need network — test the validation path instead
try:
    loop.run_until_complete(fetcher.fetch("ftp://bad.com"))
    test("bad scheme raises", False)
except ValueError:
    test("bad scheme raises", True)

try:
    loop.run_until_complete(fetcher.fetch("https://localhost/path"))
    test("localhost raises", False)
except ValueError:
    test("localhost raises", True)

try:
    loop.run_until_complete(fetcher.fetch("https://127.0.0.1/x"))
    test("loopback raises", False)
except ValueError:
    test("loopback raises", True)

# ═══════════════════════════════════════════
# web_fetch: result dataclass
# ═══════════════════════════════════════════

print("\n── WebFetchResult ──")

r = WebFetchResult(url="https://x.com", status_code=200, content="hi", content_type="text/html", byte_size=2, from_cache=False, duration_ms=123.4)
test("result url", r.url == "https://x.com")
test("result status", r.status_code == 200)
test("result content", r.content == "hi")
test("result from_cache", r.from_cache is False)
test("result duration", r.duration_ms == 123.4)

# ═══════════════════════════════════════════
# web_fetch: tool factory
# ═══════════════════════════════════════════

print("\n── tool factory ──")

create_web_fetch_tool = _wf_mod.create_web_fetch_tool
ToolDefinition = _tools_mod.ToolDefinition

tool = create_web_fetch_tool(fetcher)
test("tool is ToolDefinition", isinstance(tool, ToolDefinition))
test("tool name is web_fetch", tool.name == "web_fetch")
test("tool has url param", "url" in tool.parameters["properties"])
test("tool has prompt param", "prompt" in tool.parameters["properties"])
test("tool has method param", "method" in tool.parameters["properties"])
test("url is required", "url" in tool.parameters["required"])

# call with missing url
res = loop.run_until_complete(tool.handler({}))
test("missing url returns error", not res.success)

# call with blocked url
res = loop.run_until_complete(tool.handler({"url": "https://127.0.0.1/x"}))
test("blocked url returns error", not res.success)

# ═══════════════════════════════════════════
# web_fetch: ContentSummarizer protocol
# ═══════════════════════════════════════════

print("\n── summarizer protocol ──")

ContentSummarizer = _wf_mod.ContentSummarizer


class MockSummarizer(ContentSummarizer):
    async def summarize(self, content, prompt):
        return f"summary of: {prompt}"


summarizer = MockSummarizer()
result = loop.run_until_complete(summarizer.summarize("content", "find links"))
test("mock summarizer works", result == "summary of: find links")

# fetcher with summarizer
fetcher_with_sum = WebFetcher(summarizer=summarizer)
test("fetcher has summarizer", fetcher_with_sum._summarizer is not None)

# ═══════════════════════════════════════════
# web_fetch: preapproved hosts
# ═══════════════════════════════════════════

print("\n── preapproved hosts ──")

hosts = _wf_mod.DEFAULT_PREAPPROVED_HOSTS
test("python docs preapproved", "docs.python.org" in hosts)
test("react.dev preapproved", "react.dev" in hosts)
test("MDN preapproved", "developer.mozilla.org" in hosts)
test("fastapi preapproved", "fastapi.tiangolo.com" in hosts)
test("random.com not preapproved", "random.com" not in hosts)

# ═══════════════════════════════════════════
# archive: InMemoryArchiveStore
# ═══════════════════════════════════════════

print("\n── archive: in-memory store ──")

InMemoryArchiveStore = _arch_mod.InMemoryArchiveStore
ArchiveEntry = _arch_mod.ArchiveEntry

store = InMemoryArchiveStore()
test("empty store has no sessions", store.list_sessions() == [])
test("empty load returns empty", store.load("nonexistent") == [])

entry = store.save(
    session_id="sess-1",
    turns=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
    summary="user said hello, assistant responded",
)
test("save returns ArchiveEntry", isinstance(entry, ArchiveEntry))
test("entry session_id", entry.session_id == "sess-1")
test("entry turn_count", entry.turn_count == 2)
test("entry has turns", len(entry.turns) == 2)
test("entry has summary", "hello" in entry.summary)
test("entry token_estimate > 0", entry.token_estimate > 0)

loaded = store.load("sess-1")
test("load returns 1 entry", len(loaded) == 1)
test("loaded matches saved", loaded[0].session_id == "sess-1")

store.save("sess-1", [{"role": "user", "content": "more"}], "second compaction")
loaded2 = store.load("sess-1")
test("load returns 2 entries after second save", len(loaded2) == 2)

store.save("sess-2", [{"role": "user", "content": "other"}], "other session")
test("list_sessions returns both", sorted(store.list_sessions()) == ["sess-1", "sess-2"])

# ═══════════════════════════════════════════
# archive: FileArchiveStore
# ═══════════════════════════════════════════

print("\n── archive: file store ──")

FileArchiveStore = _arch_mod.FileArchiveStore

tmpdir = tempfile.mkdtemp(prefix="archive_test_")
fstore = FileArchiveStore(archive_dir=tmpdir)

fentry = fstore.save(
    session_id="file-sess",
    turns=[{"role": "user", "content": "test turn"}],
    summary="test summary",
    metadata={"extra": "data"},
)
test("file save returns entry", isinstance(fentry, ArchiveEntry))
test("file entry session_id", fentry.session_id == "file-sess")

# check file exists
import json
files = os.listdir(tmpdir)
test("archive file created", len(files) == 1)
test("file is json", files[0].endswith(".json"))

# load back
floaded = fstore.load("file-sess")
test("file load returns 1 entry", len(floaded) == 1)
test("file loaded session_id", floaded[0].session_id == "file-sess")
test("file loaded turns", len(floaded[0].turns) == 1)
test("file loaded metadata", floaded[0].metadata.get("extra") == "data")

# list sessions
test("file list_sessions", fstore.list_sessions() == ["file-sess"])

# nonexistent session
test("file load nonexistent", fstore.load("nope") == [])

# multiple entries
fstore.save("file-sess", [{"role": "assistant", "content": "second"}], "second")
fstore.save("other-sess", [{"role": "user", "content": "other"}], "other")
test("file list_sessions multiple", sorted(fstore.list_sessions()) == ["file-sess", "other-sess"])
test("file load multiple", len(fstore.load("file-sess")) == 2)

shutil.rmtree(tmpdir)

# ═══════════════════════════════════════════
# archive: ArchiveEntry roundtrip
# ═══════════════════════════════════════════

print("\n── archive entry roundtrip ──")

ae = ArchiveEntry(
    session_id="rt",
    timestamp=12345.0,
    turns=[{"role": "user", "content": "hi"}],
    summary="hi",
    turn_count=1,
    token_estimate=1,
    metadata={"key": "val"},
)
d = ae.to_dict()
test("to_dict has session_id", d["session_id"] == "rt")
test("to_dict has timestamp", d["timestamp"] == 12345.0)

ae2 = ArchiveEntry.from_dict(d)
test("roundtrip session_id", ae2.session_id == "rt")
test("roundtrip turns", ae2.turns == [{"role": "user", "content": "hi"}])
test("roundtrip metadata", ae2.metadata == {"key": "val"})

# ═══════════════════════════════════════════
# compactor + archive integration
# ═══════════════════════════════════════════

print("\n── compactor + archive integration ──")

ContextCompactor = _comp_mod.ContextCompactor

mem_archive = InMemoryArchiveStore()
compactor = ContextCompactor(
    max_tokens=100,
    compact_threshold=0.5,
    keep_recent=2,
    archive=mem_archive,
    session_id="test-session",
)

# add enough turns to trigger compaction
for i in range(10):
    compactor.add_turn("user", f"message {i} " + "x" * 100)

test("needs compaction", compactor.needs_compaction)

result = loop.run_until_complete(compactor.compact())
test("compaction happened", result.turns_removed > 0)

archived = mem_archive.load("test-session")
test("turns were archived", len(archived) == 1)
test("archived turn count matches removed", archived[0].turn_count == result.turns_removed)
test("archived has summary", len(archived[0].summary) > 0)
test("archived turns preserved", len(archived[0].turns) == result.turns_removed)

# compactor without archive still works
compactor_no_arch = ContextCompactor(max_tokens=100, compact_threshold=0.5, keep_recent=2)
for i in range(10):
    compactor_no_arch.add_turn("user", f"msg {i} " + "y" * 100)
result2 = loop.run_until_complete(compactor_no_arch.compact())
test("compaction works without archive", result2.turns_removed > 0)


# ═══════════════════════════════════════════
print(f"\n{'=' * 50}")
print(f"  web_fetch + archive tests: {passed} passed, {failed} failed")
print(f"{'=' * 50}")

sys.exit(0 if failed == 0 else 1)
