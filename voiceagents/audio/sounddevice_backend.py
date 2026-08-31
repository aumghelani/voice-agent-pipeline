"""Real microphone/speaker backend using PortAudio via `sounddevice`.

Only imported when actually instantiated, so the rest of the package has
no hard dependency on PortAudio being installed/available.
"""

from __future__ import annotations

import numpy as np

from voiceagents.audio.base import AudioBackend, AudioConfig, rms


class SoundDeviceBackend(AudioBackend):
    def __init__(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError(
                "SoundDeviceBackend requires the 'audio' extra: pip install "
                "'voice-agents-lab[audio]'"
            ) from exc
        self._sd = sd

    def record(self, duration_s: float, config: AudioConfig | None = None) -> np.ndarray:
        cfg = config or AudioConfig()
        n_samples = int(duration_s * cfg.sample_rate)
        recording = self._sd.rec(
            n_samples, samplerate=cfg.sample_rate, channels=cfg.channels, dtype=cfg.dtype
        )
        self._sd.wait()
        return recording.reshape(-1)

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
        chunk_samples = max(1, int(chunk_s * cfg.sample_rate))
        needed_silent_chunks = max(1, int(silence_duration_s / chunk_s))
        max_chunks = max(1, int(max_duration_s / chunk_s))

        chunks: list[np.ndarray] = []
        silent_run = 0

        with self._sd.InputStream(
            samplerate=cfg.sample_rate, channels=cfg.channels, dtype=cfg.dtype
        ) as stream:
            for _ in range(max_chunks):
                block, _overflowed = stream.read(chunk_samples)
                block = block.reshape(-1)
                chunks.append(block)
                if rms(block) < silence_threshold:
                    silent_run += 1
                    if silent_run >= needed_silent_chunks and chunks:
                        break
                else:
                    silent_run = 0

        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def play(self, samples: np.ndarray, config: AudioConfig | None = None) -> None:
        cfg = config or AudioConfig()
        cleaned = _fade_edges(_remove_dc_offset(np.asarray(samples, dtype=np.float32)))
        self._sd.play(cleaned, samplerate=cfg.sample_rate, blocksize=2048)
        self._sd.wait()


def _remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples
    return samples - float(np.mean(samples))


def _fade_edges(samples: np.ndarray, fade_in_ms: float = 8.0, fade_out_ms: float = 40.0,
                 sample_rate: int = 16_000) -> np.ndarray:
    """Cosine fade in/out to avoid audible clicks/pops at clip boundaries."""
    if samples.size == 0:
        return samples
    out = samples.copy()
    fade_in_n = min(int(fade_in_ms / 1000 * sample_rate), out.size // 2)
    fade_out_n = min(int(fade_out_ms / 1000 * sample_rate), out.size // 2)

    if fade_in_n > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_in_n)))
        out[:fade_in_n] *= ramp
    if fade_out_n > 0:
        ramp = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_out_n)))
        out[-fade_out_n:] *= ramp
    return out
