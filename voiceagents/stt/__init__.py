from voiceagents.stt.base import Transcriber, TranscriptEvent
from voiceagents.stt.mock_stt import MockTranscriber

__all__ = ["Transcriber", "TranscriptEvent", "MockTranscriber", "WhisperTranscriber"]


def __getattr__(name: str):
    if name == "WhisperTranscriber":
        from voiceagents.stt.whisper_stt import WhisperTranscriber

        return WhisperTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
