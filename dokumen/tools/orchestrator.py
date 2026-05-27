"""
tool orchestrator — concurrency-safe tool execution.

partitions tool calls into batches: consecutive read-only tools batch
together and run concurrently (up to max_concurrent), write tools run
one at a time. this prevents lost updates while maximizing throughput.

inspired by claude code's tool orchestration architecture.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# default max concurrent read-only tools
DEFAULT_MAX_CONCURRENT = 10


class ToolConcurrencyMode(Enum):
    """whether a tool is safe to run concurrently."""
    READ_ONLY = "read_only"    # safe to batch with other read-only tools
    WRITE = "write"            # must run serially
    UNKNOWN = "unknown"        # treat as write (safe default)


# known read-only tools — these never modify state
READ_ONLY_TOOLS = frozenset({
    "read_file",
    "list_directory",
    "list_files",
    "glob",
    "search_file_content",
    "search_files",
    "web_fetch",
    "web_search",
    "anthropic_web_search",
    # browser observation tools
    "browser_snapshot",
    "browser_screenshot",
    "browser_take_screenshot",
    "browser_console_messages",
    "browser_network_requests",
})

# known write tools — these modify state
WRITE_TOOLS = frozenset({
    "run_shell_command",
    "write_file",
    "delete_file",
    "create_test",
    # browser action tools
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_evaluate",
    "browser_wait",
    "browser_close",
})


def classify_tool(tool_name: str) -> ToolConcurrencyMode:
    """classify a tool as read-only or write.

    known tools are classified from the hardcoded sets.
    unknown tools default to WRITE (safe).
    """
    if tool_name in READ_ONLY_TOOLS:
        return ToolConcurrencyMode.READ_ONLY
    if tool_name in WRITE_TOOLS:
        return ToolConcurrencyMode.WRITE
    return ToolConcurrencyMode.UNKNOWN


@dataclass
class ToolCall:
    """a pending tool call."""
    tool_name: str
    args: Dict[str, Any]
    call_id: str = ""
    mode: ToolConcurrencyMode = field(init=False)

    def __post_init__(self):
        self.mode = classify_tool(self.tool_name)


@dataclass
class ToolResult:
    """result from executing a tool."""
    call_id: str
    tool_name: str
    output: str
    success: bool = True
    error: Optional[str] = None
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "duration": self.duration,
        }


@dataclass
class ToolBatch:
    """a batch of tool calls to execute together."""
    calls: List[ToolCall] = field(default_factory=list)
    concurrent: bool = False

    @property
    def tool_names(self) -> List[str]:
        return [c.tool_name for c in self.calls]


# type for tool executor function
ToolExecutor = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, str]]


class ToolOrchestrator:
    """orchestrates tool execution with concurrency safety.

    partitions tool calls into batches:
    - consecutive read-only tools → concurrent batch (up to max_concurrent)
    - write tools → serial batch (single tool)

    usage:
        orchestrator = ToolOrchestrator(executor=my_tool_executor)
        results = await orchestrator.execute([
            ToolCall(tool_name="read_file", args={"path": "a.py"}),
            ToolCall(tool_name="glob", args={"pattern": "*.py"}),
            ToolCall(tool_name="write_file", args={"path": "b.py", "content": "..."}),
            ToolCall(tool_name="read_file", args={"path": "c.py"}),
        ])
        # read_file + glob run concurrently
        # write_file runs alone
        # second read_file runs alone (new batch after write)
    """

    def __init__(
        self,
        executor: Optional[ToolExecutor] = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ):
        self._executor = executor
        self._max_concurrent = max_concurrent
        self._total_calls = 0
        self._total_batches = 0

        logger.info(
            "tool orchestrator initialized",
            extra={"max_concurrent": max_concurrent},
        )

    def partition(self, calls: List[ToolCall]) -> List[ToolBatch]:
        """partition tool calls into batches.

        consecutive read-only calls batch together.
        write/unknown calls always start a new single-item batch.
        """
        if not calls:
            return []

        batches: List[ToolBatch] = []
        current_batch: Optional[ToolBatch] = None

        for call in calls:
            is_readonly = call.mode == ToolConcurrencyMode.READ_ONLY

            if is_readonly:
                if current_batch and current_batch.concurrent:
                    # extend existing read-only batch
                    current_batch.calls.append(call)
                else:
                    # start new read-only batch
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = ToolBatch(calls=[call], concurrent=True)
            else:
                # write tool — flush current batch, create serial batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = ToolBatch(calls=[call], concurrent=False)

        if current_batch:
            batches.append(current_batch)

        logger.debug(
            "partitioned tool calls",
            extra={
                "total_calls": len(calls),
                "batches": len(batches),
                "concurrent_batches": sum(1 for b in batches if b.concurrent),
            },
        )

        return batches

    async def execute(self, calls: List[ToolCall]) -> List[ToolResult]:
        """execute tool calls with proper batching.

        returns results in the same order as input calls.
        """
        if self._executor is None:
            raise ValueError("no executor provided — pass executor to __init__ or use partition() only")

        batches = self.partition(calls)
        all_results: List[ToolResult] = []

        for batch in batches:
            self._total_batches += 1

            if batch.concurrent and len(batch.calls) > 1:
                results = await self._run_concurrent(batch)
            else:
                results = await self._run_serial(batch)

            all_results.extend(results)
            self._total_calls += len(batch.calls)

        return all_results

    async def _run_concurrent(self, batch: ToolBatch) -> List[ToolResult]:
        """run a batch of read-only tools concurrently."""
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _run_one(call: ToolCall) -> ToolResult:
            async with sem:
                return await self._execute_one(call)

        logger.debug(
            "running concurrent batch",
            extra={"tools": batch.tool_names, "count": len(batch.calls)},
        )

        tasks = [_run_one(call) for call in batch.calls]
        return list(await asyncio.gather(*tasks))

    async def _run_serial(self, batch: ToolBatch) -> List[ToolResult]:
        """run tools one at a time."""
        results = []
        for call in batch.calls:
            result = await self._execute_one(call)
            results.append(result)
        return results

    async def _execute_one(self, call: ToolCall) -> ToolResult:
        """execute a single tool call."""
        start = time.time()
        try:
            output = await self._executor(call.tool_name, call.args)
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output=output,
                success=True,
                duration=time.time() - start,
            )
        except Exception as e:
            logger.warning(
                "tool execution failed",
                extra={"tool": call.tool_name, "error": str(e)},
            )
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                output="",
                success=False,
                error=str(e),
                duration=time.time() - start,
            )

    def stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self._total_calls,
            "total_batches": self._total_batches,
            "max_concurrent": self._max_concurrent,
        }
