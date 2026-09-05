"""Conversation memory as a real multi-turn chat message list.

Unlike a single-string-with-injected-context design, this keeps each turn as
its own Message so an LLMProvider can use native multi-turn chat semantics.
A sliding window bounds prompt size for long conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from voiceagents.llm.base import Message


@dataclass
class ConversationMemory:
    system_prompt: str = ""
    max_turns: int = 20
    _turns: list[Message] = field(default_factory=list)

    def add_user_turn(self, text: str) -> None:
        self._turns.append(Message(role="user", content=text))
        self._trim()

    def add_assistant_turn(self, text: str) -> None:
        self._turns.append(Message(role="assistant", content=text))
        self._trim()

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def to_messages(self) -> list[Message]:
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.extend(self._turns)
        return messages

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
