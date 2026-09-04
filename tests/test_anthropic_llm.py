import pytest

from voiceagents.llm import Message
from voiceagents.llm.anthropic_llm import AnthropicLLM


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, texts: list[str]) -> None:
        self.content = [_FakeTextBlock(t) for t in texts]


class _FakeMessages:
    def __init__(self) -> None:
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _FakeResponse(["hello ", "world"])


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_complete_joins_text_blocks_and_sends_system_separately():
    client = _FakeClient()
    llm = AnthropicLLM(client=client, model="claude-sonnet-5")

    reply = llm.complete(
        [Message("system", "Be terse."), Message("user", "hi")]
    )

    assert reply == "hello world"
    call = client.messages.create_calls[0]
    assert call["system"] == "Be terse."
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["model"] == "claude-sonnet-5"


def test_complete_omits_system_kwarg_when_no_system_message():
    client = _FakeClient()
    llm = AnthropicLLM(client=client)

    llm.complete([Message("user", "hi")])

    call = client.messages.create_calls[0]
    assert "system" not in call


def test_get_client_raises_clear_error_without_dependency():
    import sys
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("no module named anthropic")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    sys.modules.pop("anthropic", None)
    try:
        llm = AnthropicLLM()
        with pytest.raises(ImportError, match="voice-agents-lab\\[anthropic\\]"):
            llm.complete([Message("user", "hi")])
    finally:
        builtins.__import__ = real_import
