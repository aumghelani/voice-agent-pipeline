import numpy as np

from voiceagents.stt import MockTranscriber, TranscriptEvent


def test_transcribe_returns_scripted_text():
    stt = MockTranscriber(scripted_transcripts=["hello world"])

    text = stt.transcribe(np.zeros(1600, dtype=np.float32))

    assert text == "hello world"


def test_transcribe_returns_empty_string_when_no_script_queued():
    stt = MockTranscriber()

    text = stt.transcribe(np.zeros(1600, dtype=np.float32))

    assert text == ""


def test_transcribe_advances_through_multiple_scripted_calls():
    stt = MockTranscriber(scripted_transcripts=["first", "second"])

    assert stt.transcribe(np.zeros(10, dtype=np.float32)) == "first"
    assert stt.transcribe(np.zeros(10, dtype=np.float32)) == "second"


def test_stream_yields_growing_partials_then_final():
    stt = MockTranscriber(scripted_transcripts=["hello world"])
    chunks = iter([np.zeros(160, dtype=np.float32)])

    events = list(stt.stream(chunks))

    assert events == [
        TranscriptEvent(text="hello", is_final=False),
        TranscriptEvent(text="hello world", is_final=False),
        TranscriptEvent(text="hello world", is_final=True),
    ]


def test_stream_consumes_the_chunk_iterator():
    stt = MockTranscriber(scripted_transcripts=["hi"])
    consumed = []

    def chunks():
        for i in range(3):
            consumed.append(i)
            yield np.zeros(10, dtype=np.float32)

    list(stt.stream(chunks()))

    assert consumed == [0, 1, 2]
