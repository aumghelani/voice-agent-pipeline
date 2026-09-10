"""Interactive CLI entrypoint.

`demo` runs a fully mocked end-to-end turn so the pipeline shape is visible
with no mic, speakers, or downloaded models required. `chat` runs a live
text-only REPL against a real LLM backend (Anthropic) when an API key is
available, useful for exercising ConversationMemory + tool loop without
audio hardware at all.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console

from voiceagents.agent.core import VoiceAgent
from voiceagents.agent.memory import ConversationMemory
from voiceagents.audio import MockAudioBackend
from voiceagents.llm import EchoLLM, Message
from voiceagents.stt import MockTranscriber
from voiceagents.tts import MockSynthesizer

app = typer.Typer(help="voice-agents-lab: from-scratch voice agent pipeline")
console = Console()


@app.command()
def demo() -> None:
    """Run one fully-mocked pipeline turn end-to-end (no hardware/models)."""
    agent = VoiceAgent(
        audio=MockAudioBackend(),
        stt=MockTranscriber(scripted_transcripts=["Hello, what can you do?"]),
        llm=EchoLLM(prefix="I heard you say: "),
        tts=MockSynthesizer(),
        memory=ConversationMemory(system_prompt="You are a helpful voice assistant."),
    )
    result = agent.run_turn_blocking()
    console.print(f"[bold cyan]User:[/bold cyan] {result.user_text}")
    console.print(f"[bold green]Assistant:[/bold green] {result.assistant_text}")


@app.command()
def chat() -> None:
    """Text-only multi-turn REPL against a real LLM (requires ANTHROPIC_API_KEY)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[yellow]ANTHROPIC_API_KEY is not set; falling back to the EchoLLM mock "
            "so this still runs end-to-end.[/yellow]"
        )
        llm = EchoLLM()
    else:
        from voiceagents.llm import AnthropicLLM

        llm = AnthropicLLM()

    memory = ConversationMemory(system_prompt="You are a helpful, concise voice assistant.")
    console.print("[bold]Type 'exit' to quit.[/bold]")
    while True:
        try:
            user_text = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.strip().lower() in {"exit", "quit"}:
            break
        memory.add_user_turn(user_text)
        reply = llm.complete(memory.to_messages())
        memory.add_assistant_turn(reply)
        console.print(f"[bold green]Assistant:[/bold green] {reply}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
