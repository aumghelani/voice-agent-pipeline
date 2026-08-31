from voiceagents.audio.base import AudioBackend, AudioConfig
from voiceagents.audio.mock_backend import MockAudioBackend

__all__ = ["AudioBackend", "AudioConfig", "MockAudioBackend", "SoundDeviceBackend"]


def __getattr__(name: str):
    # Lazy import so `sounddevice` is only required if this backend is used.
    if name == "SoundDeviceBackend":
        from voiceagents.audio.sounddevice_backend import SoundDeviceBackend

        return SoundDeviceBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
