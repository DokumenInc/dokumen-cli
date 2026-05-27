"""
Debug utilities for dokumen.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import random
import string

logger = logging.getLogger(__name__)

# Global debug flag
_debug_enabled = False
_debug_session: Optional['DebugSession'] = None

# Truncation limits for debug trace output to prevent oversized artifacts
MAX_DEBUG_TOOL_RESULT_CHARS = 2000
MAX_DEBUG_MESSAGE_CONTENT_CHARS = 5000


def set_debug(enabled: bool) -> None:
    """Enable or disable debug output."""
    global _debug_enabled
    _debug_enabled = enabled


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return _debug_enabled


def debug(message: str) -> None:
    """Print a debug message if debug mode is enabled."""
    if _debug_enabled:
        print(message)


@dataclass
class DebugSession:
    """Tracks a debug session with file output."""
    command: str
    output_dir: str = ".dokumen-cache/debug-traces"
    started_at: datetime = field(default_factory=datetime.now)
    meta: Dict[str, Any] = field(default_factory=dict)
    tests: List[Dict[str, Any]] = field(default_factory=list)
    analyzers: List[Dict[str, Any]] = field(default_factory=list)
    scaffold_generation: Optional[Dict[str, Any]] = field(default=None)
    _current_test: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _current_executor: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _current_judge: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _current_analyzer: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _session_id: str = field(default_factory=lambda: ''.join(
        random.choices(string.ascii_lowercase + string.digits, k=6)))

    def get_output_path(self) -> Path:
        """Generate unique output file path using session ID."""
        timestamp = self.started_at.strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{self.command}_{timestamp}_{self._session_id}.json"
        return Path(self.output_dir) / filename

    # ==========================================================================
    # Test tracking (for 'run' command)
    # ==========================================================================

    def start_test(self, test_id: str) -> None:
        """Start tracking a new test."""
        self._current_test = {
            "test_id": test_id,
            "started_at": datetime.now().isoformat(),
            "executor": {"iterations": [], "output": None},
            "judges": [],
            "result": None
        }

    def start_executor(self) -> None:
        """Start tracking executor for current test."""
        if self._current_test:
            self._current_executor = self._current_test["executor"]

    def add_executor_iteration(self, iteration: int, messages: List,
                                response: Any, tool_calls: List) -> None:
        """Record an executor iteration with truncated content for size control."""
        if self._current_executor is not None:
            serialized_messages = _truncate_message_content(_serialize_messages(messages))
            self._current_executor["iterations"].append({
                "iteration": iteration,
                "messages_sent": serialized_messages,
                "response": _serialize_response(response),
                "tool_calls": _truncate_tool_calls_for_debug(tool_calls)
            })

    def finish_executor(self, output: Dict) -> None:
        """Record executor output."""
        if self._current_test:
            self._current_test["executor"]["output"] = output
        self._current_executor = None

    def start_judge(self, judge_id: str) -> None:
        """Start tracking a judge."""
        self._current_judge = {
            "judge_id": judge_id,
            "iterations": [],
            "result": None
        }

    def add_judge_iteration(self, iteration: int, messages: List,
                            response: Any, tool_calls: List) -> None:
        """Record a judge iteration with truncated content for size control."""
        if self._current_judge:
            serialized_messages = _truncate_message_content(_serialize_messages(messages))
            self._current_judge["iterations"].append({
                "iteration": iteration,
                "messages_sent": serialized_messages,
                "response": _serialize_response(response),
                "tool_calls": _truncate_tool_calls_for_debug(tool_calls)
            })

    def finish_judge(self, result: Dict) -> None:
        """Record judge result."""
        if self._current_judge and self._current_test:
            self._current_judge["result"] = result
            self._current_test["judges"].append(self._current_judge)
            self._current_judge = None

    def finish_test(self, result: Dict) -> None:
        """Finish tracking current test."""
        if self._current_test:
            self._current_test["completed_at"] = datetime.now().isoformat()
            self._current_test["result"] = result
            self.tests.append(self._current_test)
            self._current_test = None

    def flush_test(self, test_id: str) -> None:
        """
        Write completed test data to debug file incrementally.

        Args:
            test_id: The test that just completed (must already be in self.tests)
        """
        if not _debug_enabled:
            return

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use session-based filename for consistent incremental writes
        timestamp = self.started_at.strftime("%Y-%m-%dT%H-%M-%S")
        debug_file = output_dir / f"{self.command}_{timestamp}_{self._session_id}.json"

        # Load existing or create new
        if debug_file.exists():
            with open(debug_file, 'r') as f:
                data = json.load(f)
        else:
            data = {
                "meta": {
                    "command": self.command,
                    "started_at": self.started_at.isoformat(),
                    **self.meta
                },
                "tests": [],
                "analyzers": []
            }

        # Find the just-completed test and append if not already present
        existing_test_ids = {t.get("test_id") for t in data.get("tests", [])}
        for test in self.tests:
            if test.get("test_id") == test_id and test_id not in existing_test_ids:
                data["tests"].append(test)
                break

        with open(debug_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    # ==========================================================================
    # Analyzer tracking (for 'analyze' command)
    # ==========================================================================

    def start_analyzer(self, analyzer_name: str) -> None:
        """Start tracking an analyzer."""
        self._current_analyzer = {
            "analyzer_name": analyzer_name,
            "started_at": datetime.now().isoformat(),
            "iterations": [],
            "problems": [],
            "result": None
        }

    def add_analyzer_iteration(self, iteration: int, messages: List,
                                response: Any, tool_calls: List) -> None:
        """Record an analyzer iteration with truncated content for size control."""
        if self._current_analyzer:
            serialized_messages = _truncate_message_content(_serialize_messages(messages))
            self._current_analyzer["iterations"].append({
                "iteration": iteration,
                "messages_sent": serialized_messages,
                "response": _serialize_response(response),
                "tool_calls": _truncate_tool_calls_for_debug(tool_calls)
            })

    def add_analyzer_problem(self, problem: Dict) -> None:
        """Record a problem found by analyzer."""
        if self._current_analyzer:
            self._current_analyzer["problems"].append(problem)

    def finish_analyzer(self, result: Optional[Dict] = None) -> None:
        """Finish tracking current analyzer."""
        if self._current_analyzer:
            self._current_analyzer["completed_at"] = datetime.now().isoformat()
            self._current_analyzer["result"] = result
            self.analyzers.append(self._current_analyzer)
            self._current_analyzer = None

    # ==========================================================================
    # Scaffold generation tracking (for 'add' command)
    # ==========================================================================

    def start_scaffold_generation(self, doc_path: str, name: Optional[str] = None) -> None:
        """Start tracking scaffold generation."""
        self.scaffold_generation = {
            "doc_path": doc_path,
            "name": name,
            "started_at": datetime.now().isoformat(),
            "iterations": [],
            "result": None
        }

    def add_scaffold_iteration(self, iteration: int, messages: List,
                                response: Any, tool_calls: List) -> None:
        """Record a scaffold generation iteration with truncated content for size control."""
        if self.scaffold_generation:
            serialized_messages = _truncate_message_content(_serialize_messages(messages))
            self.scaffold_generation["iterations"].append({
                "iteration": iteration,
                "messages_sent": serialized_messages,
                "response": _serialize_response(response),
                "tool_calls": _truncate_tool_calls_for_debug(tool_calls)
            })

    def finish_scaffold_generation(self, result: Optional[Dict] = None) -> None:
        """Finish tracking scaffold generation."""
        if self.scaffold_generation:
            self.scaffold_generation["completed_at"] = datetime.now().isoformat()
            self.scaffold_generation["result"] = result

    # ==========================================================================
    # Output
    # ==========================================================================

    def write(self) -> Path:
        """Write session to file."""
        output_path = self.get_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "meta": {
                "command": self.command,
                "started_at": self.started_at.isoformat(),
                "completed_at": datetime.now().isoformat(),
                **self.meta
            },
            "tests": self.tests if self.tests else None,
            "analyzers": self.analyzers if self.analyzers else None,
            "scaffold_generation": self.scaffold_generation
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return output_path


def _truncate_tool_calls_for_debug(tool_calls: List) -> List:
    """Truncate tool call results for debug output to prevent oversized traces.

    Args:
        tool_calls: List of tool call dicts (or other items).

    Returns:
        New list with long string results truncated. Original list is not mutated.
    """
    truncated = []
    for tc in tool_calls:
        if not isinstance(tc, dict) or "result" not in tc:
            truncated.append(tc)
            continue
        result = tc["result"]
        if isinstance(result, str) and len(result) > MAX_DEBUG_TOOL_RESULT_CHARS:
            tc_copy = {**tc, "result": result[:MAX_DEBUG_TOOL_RESULT_CHARS] + f"... [truncated from {len(result)} chars]"}
            logger.debug("Truncated tool result for debug trace", extra={"tool": tc.get("name"), "original_len": len(result), "truncated_to": MAX_DEBUG_TOOL_RESULT_CHARS})
            truncated.append(tc_copy)
        else:
            truncated.append(tc)
    return truncated


def _truncate_message_content(messages: List) -> List:
    """Truncate long message content for debug output.

    Args:
        messages: List of message dicts (or other items).

    Returns:
        New list with long string content truncated. Original list is not mutated.
    """
    truncated = []
    for msg in messages:
        if not isinstance(msg, dict) or "content" not in msg:
            truncated.append(msg)
            continue
        content = msg["content"]
        if isinstance(content, str) and len(content) > MAX_DEBUG_MESSAGE_CONTENT_CHARS:
            msg_copy = {**msg, "content": content[:MAX_DEBUG_MESSAGE_CONTENT_CHARS] + f"... [truncated from {len(content)} chars]"}
            truncated.append(msg_copy)
        else:
            truncated.append(msg)
    return truncated


def _serialize_messages(messages: List) -> List[Dict]:
    """Serialize messages for JSON output."""
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
        elif hasattr(msg, 'to_dict'):
            result.append(msg.to_dict())
        elif hasattr(msg, '__dict__'):
            result.append(msg.__dict__)
        else:
            result.append({"content": str(msg)})
    return result


def _serialize_response(response: Any) -> Any:
    """Serialize LLM response for JSON output."""
    if response is None:
        return None
    if isinstance(response, dict):
        return response
    if hasattr(response, 'to_dict'):
        return response.to_dict()
    if hasattr(response, '__dict__'):
        return response.__dict__
    return str(response)


# =============================================================================
# Session management functions
# =============================================================================

def start_debug_session(command: str, meta: Dict[str, Any] = None) -> DebugSession:
    """Start a new debug session."""
    global _debug_session
    _debug_session = DebugSession(command=command, meta=meta or {})
    return _debug_session


def get_debug_session() -> Optional[DebugSession]:
    """Get the current debug session."""
    return _debug_session


def end_debug_session() -> Optional[Path]:
    """End the current debug session and write output."""
    global _debug_session
    if _debug_session:
        path = _debug_session.write()
        _debug_session = None
        return path
    return None
