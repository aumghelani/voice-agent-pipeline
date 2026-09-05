"""Wires audio, STT, LLM, and TTS into a single conversational turn.

Two execution modes are provided:
- `run_turn_blocking`: record -> transcribe -> complete -> synthesize -> play,
  each stage fully finishing before the next starts. Simplest to reason
  about; highest latency.
- `run_turn_streaming`: record -> transcribe -> stream LLM tokens -> chunk
  into sentences -> synthesize+play each sentence as it's ready. Lower
  latency to first audio, at the cost of more moving parts.
"""

from __future__ import annotations

from dataclasses import dataclass

from voiceagents.agent.memory import ConversationMemory
from voiceagents.audio.base import AudioBackend, AudioConfig
from voiceagents.llm.base import LLMProvider
from voiceagents.stt.base import Transcriber
from voiceagents.tts.base import Synthesizer
from voiceagents.tts.sentence_chunker import chunk_sentences


@dataclass(frozen=True)
class TurnResult:
    user_text: str
    assistant_text: str


class VoiceAgent:
    def __init__(
        self,
        audio: AudioBackend,
        stt: Transcriber,
        llm: LLMProvider,
        tts: Synthesizer,
        memory: ConversationMemory | None = None,
        audio_config: AudioConfig | None = None,
    ) -> None:
        self.audio = audio
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.memory = memory if memory is not None else ConversationMemory()
        self.audio_config = audio_config or AudioConfig()

    def run_turn_blocking(self, *, record_seconds: float = 5.0) -> TurnResult:
        samples = self.audio.record(record_seconds, config=self.audio_config)
        user_text = self.stt.transcribe(samples, sample_rate=self.audio_config.sample_rate)

        self.memory.add_user_turn(user_text)
        assistant_text = self.llm.complete(self.memory.to_messages())
        self.memory.add_assistant_turn(assistant_text)

        audio_out = self.tts.synthesize(assistant_text)
        self.audio.play(audio_out, config=AudioConfig(sample_rate=self.tts.sample_rate))

        return TurnResult(user_text=user_text, assistant_text=assistant_text)

    def run_turn_streaming(self, *, record_seconds: float = 5.0) -> TurnResult:
        samples = self.audio.record(record_seconds, config=self.audio_config)
        user_text = self.stt.transcribe(samples, sample_rate=self.audio_config.sample_rate)

        self.memory.add_user_turn(user_text)
        token_stream = self.llm.stream(self.memory.to_messages())

        assistant_parts: list[str] = []

        def _tap(fragments):
            for fragment in fragments:
                assistant_parts.append(fragment)
                yield fragment

        for sentence in chunk_sentences(_tap(token_stream)):
            audio_out = self.tts.synthesize(sentence)
            self.audio.play(audio_out, config=AudioConfig(sample_rate=self.tts.sample_rate))

        assistant_text = "".join(assistant_parts)
        self.memory.add_assistant_turn(assistant_text)

        return TurnResult(user_text=user_text, assistant_text=assistant_text)
