# voice-agents-lab

A from-scratch, **testable** voice agent pipeline: microphone -> speech-to-text -> LLM -> text-to-speech -> speaker, with tool calling, memory, and turn-taking/interruption handling.

Every stage of the pipeline (audio I/O, STT, TTS, LLM) is defined as a small interface with:

- a **mock implementation** with no hardware or network dependency, so the full pipeline can be built, run, and unit-tested in CI or a sandbox, and
- at least one **real backend** you can swap in for actual microphone/speaker/model access.

This is a personal learning project, structured as incremental commits, inspired by the general shape of build-a-voice-agent tutorials (mic -> STT -> LLM -> TTS -> speaker; streaming; tool calls; turn-taking). No code was copied from any other project.

## Status

105 tests, ~89% statement coverage on `voiceagents/`. See [docs/architecture.md](docs/architecture.md)
for design notes and `tests/` for behavior specs.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run the mock end-to-end pipeline (no mic/speaker/models required):

```bash
python -m voiceagents.cli demo
```

Run a text-only multi-turn chat REPL (uses a real Claude model if `ANTHROPIC_API_KEY`
is set, otherwise falls back to a mock so it still runs):

```bash
python -m voiceagents.cli chat
```

Run the tool-calling example (router -> tool call -> summarizer, fully mocked):

```bash
python -m examples.assistant_with_tools
```

### Optional extras

Real backends are opt-in via extras so the base install stays lightweight:

```bash
pip install -e ".[audio]"      # sounddevice/PortAudio for real mic+speaker
pip install -e ".[whisper]"    # faster-whisper for real STT
pip install -e ".[anthropic]"  # Anthropic client for a real LLM backend
```

## Layout

| Path | Role |
|------|------|
| `voiceagents/audio/` | Mic input / speaker output interfaces + mock and `sounddevice` backends |
| `voiceagents/stt/` | Speech-to-text interface + mock and `faster-whisper` backends |
| `voiceagents/tts/` | Text-to-speech interface + mock backend, sentence-chunked streaming |
| `voiceagents/llm/` | LLM provider interface + scripted/mock provider and Anthropic backend |
| `voiceagents/agent/` | Conversation loop, memory, session store |
| `voiceagents/tools/` | Tool registry and LLM tool-calling loop |
| `tests/` | pytest suite covering the pipeline via mocks |
| `examples/` | Small runnable scripts, chapter-style |
| `docs/` | Design notes |
