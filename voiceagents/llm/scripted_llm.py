"""Deterministic LLM providers for tests and offline demos: no network, no
model weights, fully predictable output."""

from __future__ import annotations

from typing import Iterator

from voiceagents.llm.base import LLMProvider, Message


class ScriptedLLM(LLMProvider):
    """Returns replies from a fixed script, in order, regardless of input."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies or [])
        self._call_index = 0
        self.received_messages: list[list[Message]] = []

    def queue_reply(self, text: str) -> None:
        self._replies.append(text)

    def complete(self, messages: list[Message]) -> str:
        self.received_messages.append(list(messages))
        if self._call_index < len(self._replies):
            reply = self._replies[self._call_index]
        else:
            reply = ""
        self._call_index += 1
        return reply

    def stream(self, messages: list[Message]) -> Iterator[str]:
        reply = self.complete(messages)
        # Simulate token-by-token streaming by splitting on whitespace,
        # preserving the space so a downstream sentence chunker sees the
        # same text it would from a real token stream.
        words = reply.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else f" {word}"


class EchoLLM(LLMProvider):
    """Echoes the latest user message, prefixed. Useful for smoke-testing
    the full pipeline without scripting exact replies."""

    def __init__(self, prefix: str = "You said: ") -> None:
        self._prefix = prefix

    def complete(self, messages: list[Message]) -> str:
        user_messages = [m for m in messages if m.role == "user"]
        last = user_messages[-1].content if user_messages else ""
        return f"{self._prefix}{last}"
