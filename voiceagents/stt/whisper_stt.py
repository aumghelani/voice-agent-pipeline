"""Real STT backend using faster-whisper (CTranslate2 Whisper inference)."""

from __future__ import annotations

import numpy as np

from voiceagents.stt.base import Transcriber


class WhisperTranscriber(Transcriber):
    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        model: object | None = None,
    ) -> None:
        """`model` allows injecting an already-loaded faster_whisper.WhisperModel
        to avoid reload latency when reused across turns."""
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = model

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError(
                    "WhisperTranscriber requires the 'whisper' extra: pip install "
                    "'voice-agents-lab[whisper]'"
                ) from exc
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16_000) -> str:
        if sample_rate != 16_000:
            raise ValueError(
                f"WhisperTranscriber expects 16kHz audio, got {sample_rate}Hz. "
                "Resample before calling transcribe()."
            )
        model = self._get_model()
        segments, _info = model.transcribe(samples.astype(np.float32), language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()
