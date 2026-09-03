"""Text-to-speech interface.

`synthesize` turns a complete string into mono float32 PCM. `stream` turns an
iterator of text fragments (e.g. LLM token stream) into an iterator of audio
chunks, so playback can start before the full reply is generated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class Synthesizer(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize a complete utterance and return mono float32 samples."""

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    def stream(self, text_chunks: Iterator[str]) -> Iterator[np.ndarray]:
        """Default streaming implementation: buffer all text, synthesize once.
        Real streaming backends should override this to synthesize per
        sentence as text arrives, reducing time-to-first-audio."""
        full_text = "".join(text_chunks)
        if full_text.strip():
            yield self.synthesize(full_text)
