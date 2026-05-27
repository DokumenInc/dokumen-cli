"""
message bus — inter-agent communication for coordinator mode.

agents can send point-to-point messages or broadcast to all.
messages are tracked per-agent with read state so agents
can retrieve unread messages between turns.
"""
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

BROADCAST = "*"


@dataclass
class Message:
    """a message between agents."""
    id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:8]}")
    sender: str = ""
    recipient: str = ""  # agent name or "*" for broadcast
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_broadcast(self) -> bool:
        return self.recipient == BROADCAST

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        return cls(
            id=d.get("id", f"msg-{uuid.uuid4().hex[:8]}"),
            sender=d.get("sender", ""),
            recipient=d.get("recipient", ""),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )


MessageCallback = Callable[[Message], None]


class MessageBus:
    """inter-agent message bus with per-agent read tracking.

    usage:
        bus = MessageBus()
        bus.send("worker-1", "worker-2", "found relevant file: api.md")
        bus.broadcast("coordinator", "all workers: focus on the auth module")

        unread = bus.get_unread("worker-2")
        # returns the message from worker-1

        bus.subscribe("worker-2", lambda msg: print(f"got: {msg.content}"))
    """

    def __init__(self):
        self._messages: List[Message] = []
        self._read_state: Dict[str, Set[str]] = {}  # agent_name -> set of read message ids
        self._subscribers: Dict[str, List[MessageCallback]] = {}  # agent_name -> callbacks

    def send(self, sender: str, recipient: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """send a point-to-point message."""
        msg = Message(
            sender=sender,
            recipient=recipient,
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(msg)

        logger.info(
            "message sent",
            extra={"msg_id": msg.id, "from": sender, "to": recipient, "length": len(content)},
        )

        self._notify(msg)
        return msg

    def broadcast(self, sender: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """broadcast a message to all agents except the sender."""
        return self.send(sender, BROADCAST, content, metadata)

    def get_unread(self, agent_name: str) -> List[Message]:
        """get all unread messages addressed to this agent."""
        read_ids = self._read_state.get(agent_name, set())
        unread = []

        for msg in self._messages:
            if msg.id in read_ids:
                continue
            if self._is_addressed_to(msg, agent_name):
                unread.append(msg)

        return unread

    def mark_read(self, agent_name: str, message_ids: Optional[List[str]] = None) -> None:
        """mark messages as read for an agent. if no ids given, marks all unread."""
        if agent_name not in self._read_state:
            self._read_state[agent_name] = set()

        if message_ids is None:
            # mark all addressed messages as read
            for msg in self._messages:
                if self._is_addressed_to(msg, agent_name):
                    self._read_state[agent_name].add(msg.id)
        else:
            self._read_state[agent_name].update(message_ids)

    def get_all(self, agent_name: Optional[str] = None) -> List[Message]:
        """get all messages, optionally filtered to those addressed to an agent."""
        if agent_name is None:
            return list(self._messages)
        return [m for m in self._messages if self._is_addressed_to(m, agent_name)]

    def get_conversation(self, agent_a: str, agent_b: str) -> List[Message]:
        """get all messages between two agents in either direction."""
        return [
            m for m in self._messages
            if (m.sender == agent_a and m.recipient == agent_b)
            or (m.sender == agent_b and m.recipient == agent_a)
        ]

    def subscribe(self, agent_name: str, callback: MessageCallback) -> Callable:
        """subscribe to messages for an agent. returns unsubscribe function."""
        if agent_name not in self._subscribers:
            self._subscribers[agent_name] = []
        self._subscribers[agent_name].append(callback)

        def unsubscribe():
            try:
                self._subscribers[agent_name].remove(callback)
            except (ValueError, KeyError):
                pass

        return unsubscribe

    def get_summary(self, max_chars: int = 200) -> str:
        """get a markdown summary of all messages for prompt injection."""
        if not self._messages:
            return ""

        # group by sender
        by_sender: Dict[str, List[Message]] = {}
        for msg in self._messages:
            by_sender.setdefault(msg.sender, []).append(msg)

        parts = ["## inter-agent messages\n"]
        for sender, msgs in by_sender.items():
            parts.append(f"### from {sender}")
            for msg in msgs[-5:]:  # last 5 per sender
                to = "all" if msg.is_broadcast else msg.recipient
                content = msg.content[:max_chars] + "..." if len(msg.content) > max_chars else msg.content
                parts.append(f"- → {to}: {content}")

        return "\n".join(parts)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        """clear all messages and read state."""
        self._messages.clear()
        self._read_state.clear()

    # ── internal ──

    def _is_addressed_to(self, msg: Message, agent_name: str) -> bool:
        """check if a message is addressed to an agent (direct or broadcast)."""
        if msg.recipient == agent_name:
            return True
        if msg.is_broadcast and msg.sender != agent_name:
            return True
        return False

    def _notify(self, msg: Message) -> None:
        """notify subscribers about a new message."""
        if msg.is_broadcast:
            # notify all subscribers except sender
            for agent_name, callbacks in self._subscribers.items():
                if agent_name == msg.sender:
                    continue
                for cb in callbacks:
                    try:
                        cb(msg)
                    except Exception as e:
                        logger.warning("subscriber callback error", extra={"agent": agent_name, "error": str(e)})
        else:
            # notify direct recipient
            for cb in self._subscribers.get(msg.recipient, []):
                try:
                    cb(msg)
                except Exception as e:
                    logger.warning("subscriber callback error", extra={"agent": msg.recipient, "error": str(e)})
