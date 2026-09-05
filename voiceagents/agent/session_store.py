"""TTL-based in-memory session store.

Maps a session id to arbitrary session state (typically a ConversationMemory
instance). Lazy expiry: entries aren't proactively swept, they're just
treated as absent once their TTL has elapsed and the next access notices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class SessionStore(Generic[T]):
    def __init__(self, ttl_s: float = 3600.0, clock=None) -> None:
        self._ttl_s = ttl_s
        self._entries: dict[str, _Entry[T]] = {}
        # Injectable clock (callable returning epoch seconds) for deterministic tests.
        self._clock = clock or _default_clock

    def set(self, session_id: str, value: T) -> None:
        self._entries[session_id] = _Entry(value=value, expires_at=self._clock() + self._ttl_s)

    def get(self, session_id: str) -> T | None:
        entry = self._entries.get(session_id)
        if entry is None:
            return None
        if self._clock() >= entry.expires_at:
            del self._entries[session_id]
            return None
        return entry.value

    def touch(self, session_id: str) -> bool:
        """Reset the TTL without changing the value. Returns False if the
        session doesn't exist or already expired."""
        entry = self._entries.get(session_id)
        if entry is None or self._clock() >= entry.expires_at:
            self._entries.pop(session_id, None)
            return False
        entry.expires_at = self._clock() + self._ttl_s
        return True

    def update(self, session_id: str, value: T) -> None:
        self.set(session_id, value)

    def delete(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        return self.get(session_id) is not None

    def __len__(self) -> int:
        # Note: may include not-yet-lazily-expired entries.
        return len(self._entries)


def _default_clock() -> float:
    import time

    return time.time()
