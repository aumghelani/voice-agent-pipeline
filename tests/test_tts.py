import numpy as np

from voiceagents.tts import MockSynthesizer


def test_synthesize_returns_silence_proportional_to_text_length():
    tts = MockSynthesizer(sample_rate=24_000, seconds_per_char=0.03)

    short = tts.synthesize("hi")
    long = tts.synthesize("hello there, this is much longer text")

    assert long.shape[0] > short.shape[0]
    assert np.all(short == 0.0)


def test_synthesize_records_calls():
    tts = MockSynthesizer()

    tts.synthesize("one")
    tts.synthesize("two")

    assert tts.synthesized_texts == ["one", "two"]


def test_stream_yields_one_chunk_per_sentence():
    tts = MockSynthesizer()
    text_chunks = iter(["Hello there. ", "How are you?"])

    chunks = list(tts.stream(text_chunks))

    assert len(chunks) == 2
    assert tts.synthesized_texts == ["Hello there.", "How are you?"]


def test_sample_rate_property():
    tts = MockSynthesizer(sample_rate=22_050)

    assert tts.sample_rate == 22_050
