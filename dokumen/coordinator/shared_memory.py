"""
shared memory — agent-namespaced key/value store for coordinator mode.

agents write findings to shared memory during execution. other agents
(and the coordinator) can read them. task results are auto-persisted
so downstream tasks in the DAG can access upstream outputs.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class SharedMemoryStore(Protocol):
    """abstract shared memory store — swappable backend (rule 2.6)."""

    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> None: ...
    def delete(self, key: str) -> bool: ...
    def keys(self) -> List[str]: ...


class InMemorySharedStore:
    """in-memory implementation of shared memory."""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}  # key -> {value, metadata, created_at, updated_at}

    def get(self, key: str) -> Optional[str]:
        entry = self._data.get(key)
        return entry["value"] if entry else None

    def set(self, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        existing = self._data.get(key)
        self._data[key] = {
            "value": value,
            "metadata": metadata or {},
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
        }

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def get_entry(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._data)


class SharedMemory:
    """agent-namespaced shared memory.

    keys are stored as `agent_name/key` to track provenance.
    agents can read from any namespace but only write to their own.

    usage:
        mem = SharedMemory()
        mem.write("worker-1", "findings", "found 3 api endpoints")
        mem.write("worker-2", "findings", "auth module uses JWT")

        # read specific
        val = mem.read("worker-1", "findings")

        # read all from an agent
        entries = mem.list_by_agent("worker-1")

        # get summary for prompt injection
        summary = mem.get_summary()
    """

    def __init__(self, store: Optional[SharedMemoryStore] = None):
        self._store = store or InMemorySharedStore()

    def write(self, agent_name: str, key: str, value: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """write a value under an agent's namespace."""
        namespaced_key = f"{agent_name}/{key}"
        meta = {"agent": agent_name}
        if metadata:
            meta.update(metadata)
        self._store.set(namespaced_key, value, meta)

        logger.debug(
            "shared memory write",
            extra={"agent": agent_name, "key": key, "length": len(value)},
        )

    def read(self, agent_name: str, key: str) -> Optional[str]:
        """read a value from an agent's namespace."""
        return self._store.get(f"{agent_name}/{key}")

    def read_any(self, key: str) -> Optional[str]:
        """read a value by full namespaced key (e.g. 'worker-1/findings')."""
        return self._store.get(key)

    def list_by_agent(self, agent_name: str) -> Dict[str, str]:
        """get all entries written by an agent."""
        prefix = f"{agent_name}/"
        result = {}
        for key in self._store.keys():
            if key.startswith(prefix):
                short_key = key[len(prefix):]
                val = self._store.get(key)
                if val is not None:
                    result[short_key] = val
        return result

    def write_task_result(self, task_id: str, agent_name: str, result: str) -> None:
        """persist a task's result so downstream tasks can access it."""
        self.write(agent_name, f"task:{task_id}:result", result)

    def get_task_result(self, task_id: str) -> Optional[str]:
        """get a task's result from any agent that wrote it."""
        for key in self._store.keys():
            if f"task:{task_id}:result" in key:
                return self._store.get(key)
        return None

    def get_summary(self, max_value_chars: int = 200) -> str:
        """get a markdown summary of all shared memory for prompt injection."""
        all_keys = self._store.keys()
        if not all_keys:
            return ""

        # group by agent
        by_agent: Dict[str, List[tuple]] = {}
        for key in sorted(all_keys):
            parts = key.split("/", 1)
            agent = parts[0] if len(parts) > 1 else "unknown"
            short_key = parts[1] if len(parts) > 1 else key
            val = self._store.get(key) or ""
            if len(val) > max_value_chars:
                val = val[:max_value_chars - 3] + "..."
            by_agent.setdefault(agent, []).append((short_key, val))

        lines = ["## shared team memory\n"]
        for agent, entries in by_agent.items():
            lines.append(f"### {agent}")
            for k, v in entries:
                lines.append(f"- **{k}**: {v}")

        return "\n".join(lines)

    def clear(self) -> None:
        """clear all shared memory."""
        for key in list(self._store.keys()):
            self._store.delete(key)

    @property
    def size(self) -> int:
        return len(self._store.keys())
