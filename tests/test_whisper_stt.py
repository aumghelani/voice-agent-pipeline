import numpy as np
import pytest

from voiceagents.stt.whisper_stt import WhisperTranscriber


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    def __init__(self, segments_text: list[str]) -> None:
        self._segments_text = segments_text
        self.transcribe_calls = []

    def transcribe(self, samples, language="en"):
        self.transcribe_calls.append((samples, language))
        return [_FakeSegment(t) for t in self._segments_text], object()


def test_transcribe_joins_segments_with_injected_model():
    fake_model = _FakeWhisperModel([" hello ", "world "])
    stt = WhisperTranscriber(model=fake_model)

    text = stt.transcribe(np.zeros(1600, dtype=np.float32))

    assert text == "hello world"
    assert len(fake_model.transcribe_calls) == 1


def test_transcribe_rejects_non_16khz_audio():
    stt = WhisperTranscriber(model=_FakeWhisperModel([]))

    with pytest.raises(ValueError, match="16kHz"):
        stt.transcribe(np.zeros(100, dtype=np.float32), sample_rate=8_000)


def test_get_model_raises_clear_error_without_dependency():
    import sys
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("no module named faster_whisper")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    sys.modules.pop("faster_whisper", None)
    try:
        stt = WhisperTranscriber()
        with pytest.raises(ImportError, match="voice-agents-lab\\[whisper\\]"):
            stt.transcribe(np.zeros(100, dtype=np.float32))
    finally:
        builtins.__import__ = real_import
