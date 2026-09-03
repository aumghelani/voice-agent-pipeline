from voiceagents.llm.base import LLMProvider, Message, Role
from voiceagents.llm.scripted_llm import EchoLLM, ScriptedLLM

__all__ = ["LLMProvider", "Message", "Role", "EchoLLM", "ScriptedLLM", "AnthropicLLM"]


def __getattr__(name: str):
    if name == "AnthropicLLM":
        from voiceagents.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
