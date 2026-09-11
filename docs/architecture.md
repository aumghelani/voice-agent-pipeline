# Architecture

## Pipeline

```
mic --> AudioBackend.record --> Transcriber.transcribe --> LLMProvider.complete/stream
                                                                    |
speaker <-- AudioBackend.play <-- Synthesizer.synthesize/stream <--+
```

Every stage is an abstract interface (`AudioBackend`, `Transcriber`, `LLMProvider`,
`Synthesizer`) with a mock implementation with no hardware/network dependency, and
at least one real backend:

| Stage | Interface | Mock | Real |
|---|---|---|---|
| Audio I/O | `voiceagents.audio.base.AudioBackend` | `MockAudioBackend` | `SoundDeviceBackend` (PortAudio) |
| STT | `voiceagents.stt.base.Transcriber` | `MockTranscriber` | `WhisperTranscriber` (faster-whisper) |
| LLM | `voiceagents.llm.base.LLMProvider` | `ScriptedLLM`, `EchoLLM` | `AnthropicLLM` |
| TTS | `voiceagents.tts.base.Synthesizer` | `MockSynthesizer` | *(bring your own; interface only)* |

The interchange format is always mono float32 numpy PCM for audio, and plain
strings/`Message` objects for text — no backend-specific types leak across
stage boundaries.

## Blocking vs streaming

`VoiceAgent.run_turn_blocking()` runs each stage to completion before the
next starts: simplest to reason about, highest latency (full LLM reply
generated before any audio plays).

`VoiceAgent.run_turn_streaming()` streams LLM output through
`chunk_sentences()` (sentence-boundary splitting, hard length cap as a
fallback) and synthesizes+plays each sentence as it becomes available,
so playback can start well before the full reply is generated.

## Turn-taking and interruption

`TurnTakingController` is a small state machine (`IDLE -> LISTENING ->
THINKING -> SPEAKING -> ...`) with an explicit allow-list of legal
transitions; anything else raises `InvalidTransition` rather than silently
leaving the agent in an inconsistent state.

Each `SPEAKING` phase gets a fresh `CancelToken` (a `threading.Event`
wrapper). `InterruptibleVoiceAgent.run_turn()` checks a caller-supplied
`should_interrupt()` predicate between sentence chunks; if triggered, it
calls `barge_in()` (cancels the token, returns to `LISTENING`) and stops
before playing the rest of the reply.

**Known limitation:** interruption is sentence-granular, not sample-accurate.
Stopping mid-playback of a single audio chunk requires a callback-driven
streaming audio backend (checking the cancel token every audio block), which
`SoundDeviceBackend`'s current blocking `play()` doesn't implement. A future
`play_cancellable_stream()` on `AudioBackend` would close this gap.

`EnergyVAD` (RMS-over-threshold with a consecutive-block requirement) is a
simple building block for driving `should_interrupt()` from live mic input,
independent of the ML-based VAD models real deployments might swap in later.

## Memory

`ConversationMemory` stores each turn as a `Message` (system/user/assistant),
not a single string with injected context — so `LLMProvider`s can use native
multi-turn chat semantics. A sliding window (`max_turns`) bounds prompt size.

`SessionStore` is a separate, decoupled TTL cache for keying arbitrary state
(typically a `ConversationMemory` instance) by session id, for multi-user
scenarios (e.g. a future web server front-end).

## Tools

`ToolRegistry` maps a tool name to a Pydantic argument model and callable,
and exports JSON Schema per tool. `run_tool_loop()` implements prompt-based
tool calling for providers without native function-calling: a router pass
asks the LLM for one `{"tool": ..., "arguments": {...}}` JSON object, then
(if a tool was picked) a summarizer pass turns the raw tool result into a
natural-language reply. Router misfires (bad JSON, unknown tool, invalid
arguments) degrade to a plain reply rather than raising.

## Testing philosophy

Every stage's mock is a first-class implementation of the same interface as
its real counterpart, not a testing-only shim bolted on the side — so the
full pipeline (`VoiceAgent`, `InterruptibleVoiceAgent`, the tool loop) is
exercised in CI with zero hardware, zero model downloads, and zero network
access, while still using the exact same code paths a real deployment would.
