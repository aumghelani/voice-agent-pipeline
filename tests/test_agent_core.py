import numpy as np

from voiceagents.agent.core import VoiceAgent
from voiceagents.agent.memory import ConversationMemory
from voiceagents.audio import MockAudioBackend
from voiceagents.llm import ScriptedLLM
from voiceagents.stt import MockTranscriber
from voiceagents.tts import MockSynthesizer


def _make_agent(**overrides):
    defaults = dict(
        audio=MockAudioBackend(),
        stt=MockTranscriber(scripted_transcripts=["hello agent"]),
        llm=ScriptedLLM(replies=["Hi there. How can I help?"]),
        tts=MockSynthesizer(),
        memory=ConversationMemory(system_prompt="Be helpful."),
    )
    defaults.update(overrides)
    return VoiceAgent(**defaults)


def test_run_turn_blocking_returns_transcribed_and_generated_text():
    agent = _make_agent()

    result = agent.run_turn_blocking()

    assert result.user_text == "hello agent"
    assert result.assistant_text == "Hi there. How can I help?"


def test_run_turn_blocking_updates_memory():
    agent = _make_agent()

    agent.run_turn_blocking()

    messages = agent.memory.to_messages()
    assert messages[-2].role == "user"
    assert messages[-2].content == "hello agent"
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "Hi there. How can I help?"


def test_run_turn_blocking_plays_synthesized_audio():
    audio = MockAudioBackend()
    agent = _make_agent(audio=audio)

    agent.run_turn_blocking()

    assert len(audio.playback_log) == 1


def test_run_turn_blocking_passes_full_conversation_to_llm():
    llm = ScriptedLLM(replies=["ok"])
    agent = _make_agent(llm=llm, memory=ConversationMemory(system_prompt="sys"))

    agent.run_turn_blocking()

    sent = llm.received_messages[0]
    assert sent[0].role == "system"
    assert sent[0].content == "sys"
    assert sent[-1].content == "hello agent"


def test_run_turn_streaming_plays_one_chunk_per_sentence():
    audio = MockAudioBackend()
    stt = MockTranscriber(scripted_transcripts=["hello"])
    llm = ScriptedLLM(replies=["Hi there. How can I help?"])
    agent = _make_agent(audio=audio, stt=stt, llm=llm)

    result = agent.run_turn_streaming()

    assert result.assistant_text == "Hi there. How can I help?"
    assert len(audio.playback_log) == 2  # two sentences


def test_run_turn_streaming_updates_memory_with_full_reconstructed_text():
    agent = _make_agent()

    agent.run_turn_streaming()

    messages = agent.memory.to_messages()
    assert messages[-1].content == "Hi there. How can I help?"


def test_multi_turn_conversation_accumulates_memory():
    stt = MockTranscriber(scripted_transcripts=["first question", "second question"])
    llm = ScriptedLLM(replies=["first answer", "second answer"])
    agent = _make_agent(stt=stt, llm=llm)

    agent.run_turn_blocking()
    agent.run_turn_blocking()

    messages = agent.memory.to_messages()
    contents = [m.content for m in messages]
    assert contents == [
        "Be helpful.",
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]


def test_audio_recorded_at_configured_sample_rate():
    stt = MockTranscriber(scripted_transcripts=["hi"])
    audio = MockAudioBackend(scripted_clips=[np.zeros(8000, dtype=np.float32)])
    agent = _make_agent(audio=audio, stt=stt)

    agent.run_turn_blocking(record_seconds=0.5)

    assert agent.audio_config.sample_rate == 16_000
