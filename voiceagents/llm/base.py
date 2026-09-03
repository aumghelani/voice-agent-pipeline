"""LLM provider interface.

A provider turns a list of chat messages into a reply, either all at once
(`complete`) or as an incremental token/text-fragment stream (`stream`) for
low-latency pipelines where TTS starts speaking before generation finishes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[Message]) -> str:
        """Generate a full reply for the given conversation."""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Default streaming implementation: yield the full reply as one
        chunk. Real streaming backends should override this."""
        reply = self.complete(messages)
        if reply:
            yield reply
