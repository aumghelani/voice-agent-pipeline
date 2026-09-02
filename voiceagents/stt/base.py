"""Speech-to-text interface.

A transcriber turns mono float32 PCM samples into text. `transcribe` is the
one-shot batch path used after end-of-turn silence detection. `stream` models
incremental transcription as a generator of TranscriptEvent, so callers can
react to partial hypotheses before the final one (e.g. live captions, early
barge-in cancellation of TTS).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    is_final: bool


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, samples: np.ndarray, sample_rate: int = 16_000) -> str:
        """Transcribe a complete audio clip and return the final text."""

    def stream(
        self, chunks: Iterator[np.ndarray], sample_rate: int = 16_000
    ) -> Iterator[TranscriptEvent]:
        """Default streaming implementation: buffer everything and emit one
        final event. Real streaming backends should override this to emit
        partial events as chunks arrive."""
        buffer = [c for c in chunks]
        samples = np.concatenate(buffer) if buffer else np.zeros(0, dtype=np.float32)
        text = self.transcribe(samples, sample_rate=sample_rate)
        yield TranscriptEvent(text=text, is_final=True)
