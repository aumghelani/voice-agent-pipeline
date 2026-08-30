"""In-memory audio backend for tests and headless demos.

Recording returns pre-loaded scripted clips instead of touching a real
microphone; playback appends to a log instead of touching real speakers.
This lets the full pipeline run in CI/sandboxes with no hardware.
"""

from __future__ import annotations

import numpy as np

from voiceagents.audio.base import AudioBackend, AudioConfig, rms


class MockAudioBackend(AudioBackend):
    def __init__(self, scripted_clips: list[np.ndarray] | None = None) -> None:
        self._clips = list(scripted_clips or [])
        self._clip_index = 0
        self.playback_log: list[np.ndarray] = []

    def queue_clip(self, samples: np.ndarray) -> None:
        self._clips.append(samples)

    def record(self, duration_s: float, config: AudioConfig | None = None) -> np.ndarray:
        cfg = config or AudioConfig()
        return self._next_clip(cfg, fallback_duration_s=duration_s)

    def record_until_silence(
        self,
        config: AudioConfig | None = None,
        *,
        max_duration_s: float = 30.0,
        silence_threshold: float = 0.02,
        silence_duration_s: float = 0.8,
        chunk_s: float = 0.05,
    ) -> np.ndarray:
        cfg = config or AudioConfig()
        clip = self._next_clip(cfg, fallback_duration_s=1.0)
        max_samples = int(max_duration_s * cfg.sample_rate)
        return clip[:max_samples]

    def play(self, samples: np.ndarray, config: AudioConfig | None = None) -> None:
        self.playback_log.append(np.asarray(samples, dtype=np.float32))

    def _next_clip(self, cfg: AudioConfig, *, fallback_duration_s: float) -> np.ndarray:
        if self._clip_index < len(self._clips):
            clip = self._clips[self._clip_index]
            self._clip_index += 1
            return np.asarray(clip, dtype=np.float32)
        return np.zeros(int(fallback_duration_s * cfg.sample_rate), dtype=np.float32)

    def last_played_rms(self) -> float:
        if not self.playback_log:
            return 0.0
        return rms(self.playback_log[-1])
