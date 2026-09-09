"""Simple energy-based voice activity detection.

Not ML-based (no Silero/WebRTC VAD model) - just RMS-over-threshold with a
consecutive-block requirement, which is enough to demonstrate barge-in
detection and end-of-turn silence detection without extra model downloads.
A real deployment would likely swap this for a trained VAD model behind the
same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voiceagents.audio.base import rms


@dataclass(frozen=True)
class VadConfig:
    threshold: float = 0.02
    sustain_blocks: int = 3


class EnergyVAD:
    """Stateful detector: feed it blocks one at a time via `process_block`;
    it reports speech only after `sustain_blocks` consecutive loud blocks,
    to avoid false triggers on a single noise spike."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        self._consecutive_loud = 0

    def process_block(self, block: np.ndarray) -> bool:
        """Feed one audio block; returns True the instant sustained speech
        is confirmed (i.e. only on the triggering call, not every call
        after)."""
        if rms(block) >= self.config.threshold:
            self._consecutive_loud += 1
        else:
            self._consecutive_loud = 0
        return self._consecutive_loud == self.config.sustain_blocks

    def reset(self) -> None:
        self._consecutive_loud = 0
