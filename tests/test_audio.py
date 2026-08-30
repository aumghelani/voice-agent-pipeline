import numpy as np

from voiceagents.audio import AudioConfig, MockAudioBackend
from voiceagents.audio.base import rms


def test_record_returns_scripted_clip():
    clip = np.ones(1600, dtype=np.float32) * 0.5
    backend = MockAudioBackend(scripted_clips=[clip])

    recorded = backend.record(duration_s=0.1)

    assert np.array_equal(recorded, clip)


def test_record_falls_back_to_silence_when_no_clip_queued():
    backend = MockAudioBackend()
    cfg = AudioConfig(sample_rate=16_000)

    recorded = backend.record(duration_s=0.5, config=cfg)

    assert recorded.shape == (8_000,)
    assert np.all(recorded == 0.0)


def test_record_until_silence_truncates_to_max_duration():
    long_clip = np.ones(32_000, dtype=np.float32)
    backend = MockAudioBackend(scripted_clips=[long_clip])

    recorded = backend.record_until_silence(max_duration_s=1.0)

    assert recorded.shape[0] == 16_000


def test_play_appends_to_log_and_rms_is_computed():
    backend = MockAudioBackend()
    loud = np.ones(100, dtype=np.float32)

    backend.play(loud)

    assert len(backend.playback_log) == 1
    assert backend.last_played_rms() == 1.0


def test_rms_of_empty_signal_is_zero():
    assert rms(np.array([], dtype=np.float32)) == 0.0
