"""Deterministic STT for tests: maps audio identity (by array id / content
hash) to a scripted transcript, so pipeline tests don't need real models."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from voiceagents.stt.base import Transcriber, TranscriptEvent


class MockTranscriber(Transcriber):
    def __init__(self, scripted_transcripts: list[str] | None = None) -> None:
        self._transcripts = list(scripted_transcripts or [])
        self._call_index = 0

    def queue_transcript(self, text: str) -> None:
        self._transcripts.append(text)

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16_000) -> str:
        if self._call_index < len(self._transcripts):
            text = self._transcripts[self._call_index]
        else:
            text = ""
        self._call_index += 1
        return text

    def stream(
        self, chunks: Iterator[np.ndarray], sample_rate: int = 16_000
    ) -> Iterator[TranscriptEvent]:
        # Consume the chunk iterator (mirrors real backends pulling audio),
        # then emit growing partials word-by-word followed by a final event.
        for _ in chunks:
            pass

        text = self._transcripts[self._call_index] if self._call_index < len(self._transcripts) else ""
        self._call_index += 1

        words = text.split()
        partial = ""
        for word in words:
            partial = f"{partial} {word}".strip()
            yield TranscriptEvent(text=partial, is_final=False)
        yield TranscriptEvent(text=text, is_final=True)
