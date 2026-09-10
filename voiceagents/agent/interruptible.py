"""An interruptible, streaming voice agent: wires VoiceAgent's streaming
turn together with TurnTakingController so a caller can signal barge-in
between sentence chunks.

True sample-accurate barge-in (stopping mid-playback of a single audio
chunk) requires a streaming audio backend with a callback-driven cancel
check, which SoundDeviceBackend does not yet implement (its `play` is
blocking). This class approximates it at sentence granularity: after each
synthesized sentence is played, it checks a caller-supplied
`should_interrupt` predicate before starting the next one. This is coarser
than block-level cancellation but requires no changes to AudioBackend and is
fully testable with MockAudioBackend.
"""

from __future__ import annotations

from typing import Callable

from voiceagents.agent.core import TurnResult, VoiceAgent
from voiceagents.agent.turn_taking import TurnTakingController
from voiceagents.audio.base import AudioConfig
from voiceagents.tts.sentence_chunker import chunk_sentences


class InterruptibleVoiceAgent(VoiceAgent):
    def __init__(self, *args, controller: TurnTakingController | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.controller = controller or TurnTakingController()

    def run_turn(
        self,
        *,
        record_seconds: float = 5.0,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> TurnResult:
        should_interrupt = should_interrupt or (lambda: False)

        self.controller.start_listening()
        samples = self.audio.record(record_seconds, config=self.audio_config)
        user_text = self.stt.transcribe(samples, sample_rate=self.audio_config.sample_rate)

        self.controller.start_thinking()
        self.memory.add_user_turn(user_text)
        token_stream = self.llm.stream(self.memory.to_messages())

        assistant_parts: list[str] = []

        def _tap(fragments):
            for fragment in fragments:
                assistant_parts.append(fragment)
                yield fragment

        cancel_token = self.controller.start_speaking()
        interrupted = False

        for sentence in chunk_sentences(_tap(token_stream)):
            if cancel_token.is_cancelled() or should_interrupt():
                self.controller.barge_in()
                interrupted = True
                break
            audio_out = self.tts.synthesize(sentence)
            self.audio.play(audio_out, config=AudioConfig(sample_rate=self.tts.sample_rate))

        assistant_text = "".join(assistant_parts)
        self.memory.add_assistant_turn(assistant_text)

        if not interrupted:
            self.controller.go_idle()

        return TurnResult(user_text=user_text, assistant_text=assistant_text)
