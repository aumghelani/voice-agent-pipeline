"""Splits a stream of text fragments into sentence-ish chunks for TTS.

Waiting for a full LLM reply before speaking adds seconds of latency.
Splitting on every fragment (e.g. every token) produces choppy, unnatural
audio. Splitting on sentence boundaries is the middle ground: low latency to
first audio, without garbling short phrases across a fixed char boundary.
A hard length cap guards against a single very long sentence (or reply with
no punctuation) stalling audio indefinitely.
"""

from __future__ import annotations

import re
from typing import Iterator

_SENTENCE_END = re.compile(r"([.!?])(?:\s+|$)")


def chunk_sentences(text_fragments: Iterator[str], *, max_chars: int = 200) -> Iterator[str]:
    buffer = ""
    for fragment in text_fragments:
        buffer += fragment
        while True:
            match = _SENTENCE_END.search(buffer)
            if match:
                end = match.end()
                sentence = buffer[:end].strip()
                buffer = buffer[end:]
                if sentence:
                    yield sentence
                continue
            if len(buffer) >= max_chars:
                # No sentence boundary yet, but we've buffered enough text:
                # flush on the last whitespace to avoid splitting a word.
                split_at = buffer.rfind(" ", 0, max_chars)
                if split_at <= 0:
                    split_at = max_chars
                chunk, buffer = buffer[:split_at].strip(), buffer[split_at:]
                if chunk:
                    yield chunk
                continue
            break

    tail = buffer.strip()
    if tail:
        yield tail
