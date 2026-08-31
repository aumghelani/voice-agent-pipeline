import numpy as np
import pytest

from voiceagents.audio.sounddevice_backend import _fade_edges, _remove_dc_offset


def test_remove_dc_offset_centers_signal():
    samples = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    result = _remove_dc_offset(samples)

    assert np.allclose(result, 0.0)


def test_remove_dc_offset_handles_empty():
    assert _remove_dc_offset(np.array([], dtype=np.float32)).size == 0


def test_fade_edges_zeroes_first_and_last_sample():
    samples = np.ones(1600, dtype=np.float32)

    result = _fade_edges(samples, fade_in_ms=8.0, fade_out_ms=40.0, sample_rate=16_000)

    assert result[0] == pytest.approx(0.0, abs=1e-6)
    assert result[-1] == pytest.approx(0.0, abs=1e-6)
    assert result[len(result) // 2] == pytest.approx(1.0, abs=1e-6)


def test_fade_edges_handles_empty():
    assert _fade_edges(np.array([], dtype=np.float32)).size == 0


def test_sounddevice_backend_raises_clear_error_without_dependency():
    import sys
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("no module named sounddevice")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    sys.modules.pop("sounddevice", None)
    try:
        from voiceagents.audio.sounddevice_backend import SoundDeviceBackend

        with pytest.raises(ImportError, match="voice-agents-lab\\[audio\\]"):
            SoundDeviceBackend()
    finally:
        builtins.__import__ = real_import
