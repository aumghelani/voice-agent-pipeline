"""Real LLM backend using the Anthropic Messages API."""

from __future__ import annotations

from typing import Iterator

from voiceagents.llm.base import LLMProvider, Message


class AnthropicLLM(LLMProvider):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
        client: object | None = None,
        api_key: str | None = None,
    ) -> None:
        """`client` allows injecting a pre-built anthropic.Anthropic client
        (or a test double) instead of constructing one from an API key."""
        self._model = model
        self._max_tokens = max_tokens
        self._client = client
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "AnthropicLLM requires the 'anthropic' extra: pip install "
                    "'voice-agents-lab[anthropic]'"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[dict]]:
        system_parts = [m.content for m in messages if m.role == "system"]
        system = "\n".join(system_parts) if system_parts else None
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        return system, turns

    def complete(self, messages: list[Message]) -> str:
        client = self._get_client()
        system, turns = self._split_system(messages)
        kwargs = {"model": self._model, "max_tokens": self._max_tokens, "messages": turns}
        if system is not None:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        client = self._get_client()
        system, turns = self._split_system(messages)
        kwargs = {"model": self._model, "max_tokens": self._max_tokens, "messages": turns}
        if system is not None:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream
