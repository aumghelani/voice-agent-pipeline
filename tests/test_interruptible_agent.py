from voiceagents.agent.interruptible import InterruptibleVoiceAgent
from voiceagents.agent.memory import ConversationMemory
from voiceagents.agent.turn_taking import TurnState
from voiceagents.audio import MockAudioBackend
from voiceagents.llm import ScriptedLLM
from voiceagents.stt import MockTranscriber
from voiceagents.tts import MockSynthesizer


def _make_agent(**overrides):
    defaults = dict(
        audio=MockAudioBackend(),
        stt=MockTranscriber(scripted_transcripts=["hello agent"]),
        llm=ScriptedLLM(replies=["Sentence one. Sentence two. Sentence three."]),
        tts=MockSynthesizer(),
        memory=ConversationMemory(),
    )
    defaults.update(overrides)
    return InterruptibleVoiceAgent(**defaults)


def test_uninterrupted_turn_plays_all_sentences_and_ends_idle():
    audio = MockAudioBackend()
    agent = _make_agent(audio=audio)

    result = agent.run_turn()

    assert len(audio.playback_log) == 3
    assert agent.controller.state == TurnState.IDLE
    assert result.assistant_text == "Sentence one. Sentence two. Sentence three."


def test_should_interrupt_stops_after_first_sentence():
    audio = MockAudioBackend()
    call_count = {"n": 0}

    def should_interrupt():
        call_count["n"] += 1
        return call_count["n"] > 1  # allow first sentence through, then interrupt

    agent = _make_agent(audio=audio)

    agent.run_turn(should_interrupt=should_interrupt)

    assert len(audio.playback_log) == 1
    assert agent.controller.state == TurnState.LISTENING


def test_interrupted_turn_records_only_what_was_actually_generated_before_the_break():
    # chunk_sentences() is a generator pulled lazily by the for-loop; once we
    # `break` out on interruption, it stops advancing the underlying LLM
    # token stream, so any sentence not yet yielded (here: "Sentence three.")
    # was never pulled from the stream and never lands in assistant_parts.
    agent = _make_agent()
    call_count = {"n": 0}

    def should_interrupt():
        call_count["n"] += 1
        return call_count["n"] > 1

    agent.run_turn(should_interrupt=should_interrupt)

    messages = agent.memory.to_messages()
    assert messages[-1].content == "Sentence one. Sentence two."


def test_immediate_interrupt_plays_nothing():
    audio = MockAudioBackend()
    agent = _make_agent(audio=audio)

    agent.run_turn(should_interrupt=lambda: True)

    assert len(audio.playback_log) == 0
    assert agent.controller.state == TurnState.LISTENING


def test_controller_passes_through_listening_and_thinking_states():
    agent = _make_agent()
    seen_states = []

    class RecordingController:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if name in ("start_listening", "start_thinking", "start_speaking", "go_idle"):
                def wrapped(*a, **kw):
                    result = attr(*a, **kw)
                    seen_states.append(self._inner.state)
                    return result
                return wrapped
            return attr

    from voiceagents.agent.turn_taking import TurnTakingController

    agent.controller = RecordingController(TurnTakingController())

    agent.run_turn()

    assert seen_states == [
        TurnState.LISTENING,
        TurnState.THINKING,
        TurnState.SPEAKING,
        TurnState.IDLE,
    ]
