from voiceagents.tts.sentence_chunker import chunk_sentences


def test_splits_on_sentence_boundaries():
    fragments = iter(["Hello there. How are you? I am fine!"])

    chunks = list(chunk_sentences(fragments))

    assert chunks == ["Hello there.", "How are you?", "I am fine!"]


def test_handles_fragments_arriving_token_by_token():
    fragments = iter(["Hel", "lo the", "re. How ", "are you?"])

    chunks = list(chunk_sentences(fragments))

    assert chunks == ["Hello there.", "How are you?"]


def test_flushes_trailing_text_without_terminal_punctuation():
    fragments = iter(["No punctuation here"])

    chunks = list(chunk_sentences(fragments))

    assert chunks == ["No punctuation here"]


def test_hard_flush_on_long_sentence_without_punctuation():
    long_text = "word " * 100  # 500 chars, no sentence-ending punctuation
    fragments = iter([long_text])

    chunks = list(chunk_sentences(fragments, max_chars=50))

    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks[:-1])
    assert "".join(chunks).replace(" ", "") == long_text.replace(" ", "")


def test_empty_input_yields_nothing():
    assert list(chunk_sentences(iter([]))) == []


def test_ignores_blank_fragments():
    fragments = iter(["", "  ", "Hi.", ""])

    chunks = list(chunk_sentences(fragments))

    assert chunks == ["Hi."]
