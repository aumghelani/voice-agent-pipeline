"""Deterministic TTS for tests: produces silence of a length proportional to
the text (approximating "reading time"), so pipeline timing/ordering can be
tested without a real voice model."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from voiceagents.tts.base import Synthesizer
from voiceagents.tts.sentence_chunker import chunk_sentences


class MockSynthesizer(Synthesizer):
    def __init__(self, sample_rate: int = 24_000, seconds_per_char: float = 0.03) -> None:
        self._sample_rate = sample_rate
        self._seconds_per_char = seconds_per_char
        self.synthesized_texts: list[str] = []

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        self.synthesized_texts.append(text)
        n_samples = max(1, int(len(text) * self._seconds_per_char * self._sample_rate))
        return np.zeros(n_samples, dtype=np.float32)

    def stream(self, text_chunks: Iterator[str]) -> Iterator[np.ndarray]:
        for sentence in chunk_sentences(text_chunks):
            yield self.synthesize(sentence)
