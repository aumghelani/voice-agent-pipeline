"""Audio I/O interface.

All audio in this project is represented as mono float32 numpy arrays in
[-1.0, 1.0], at a configurable sample rate (16 kHz is the default for STT
input; TTS output may use a different rate). Keeping one in-memory
representation everywhere means every other stage (STT, TTS, VAD) can stay
backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    dtype: str = "float32"


class AudioBackend(ABC):
    """A recording + playback device. Implementations may be real hardware
    (sounddevice) or in-memory fakes for testing."""

    @abstractmethod
    def record(self, duration_s: float, config: AudioConfig | None = None) -> np.ndarray:
        """Record `duration_s` seconds of audio and return mono float32 samples."""

    @abstractmethod
    def record_until_silence(
        self,
        config: AudioConfig | None = None,
        *,
        max_duration_s: float = 30.0,
        silence_threshold: float = 0.02,
        silence_duration_s: float = 0.8,
        chunk_s: float = 0.05,
    ) -> np.ndarray:
        """Record until `silence_duration_s` seconds of near-silence (simple
        energy-based end-pointing), or `max_duration_s` is hit."""

    @abstractmethod
    def play(self, samples: np.ndarray, config: AudioConfig | None = None) -> None:
        """Play mono float32 samples, blocking until playback finishes."""


def rms(samples: np.ndarray) -> float:
    """Root-mean-square energy of a signal, used as a cheap voice-activity proxy."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
