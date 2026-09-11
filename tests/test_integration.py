"""End-to-end tests exercising the full mocked pipeline together, as
opposed to each unit test file's narrower per-module scope."""

import json

import numpy as np

from voiceagents.agent.core import VoiceAgent
from voiceagents.agent.interruptible import InterruptibleVoiceAgent
from voiceagents.agent.memory import ConversationMemory
from voiceagents.agent.turn_taking import TurnState
from voiceagents.audio import MockAudioBackend
from voiceagents.llm import Message, ScriptedLLM
from voiceagents.stt import MockTranscriber
from voiceagents.tts import MockSynthesizer
from voiceagents.tools import ToolRegistry, run_tool_loop
from voiceagents.tools.builtin import register_builtin_tools


def test_multi_turn_conversation_with_streaming_and_barge_in():
    audio = MockAudioBackend(
        scripted_clips=[np.zeros(8_000, dtype=np.float32), np.zeros(8_000, dtype=np.float32)]
    )
    stt = MockTranscriber(scripted_transcripts=["What's the weather like?", "Never mind, thanks"])
    llm = ScriptedLLM(
        replies=[
            "It's sunny today. Perfect for a walk. Let me know if you need anything else.",
            "No problem, have a great day!",
        ]
    )
    tts = MockSynthesizer()
    agent = InterruptibleVoiceAgent(
        audio=audio,
        stt=stt,
        llm=llm,
        tts=tts,
        memory=ConversationMemory(system_prompt="You are a helpful assistant."),
    )

    # Turn 1: user gets interrupted after the first sentence.
    interrupt_after_first = {"n": 0}

    def interrupt_once():
        interrupt_after_first["n"] += 1
        return interrupt_after_first["n"] > 1

    result1 = agent.run_turn(should_interrupt=interrupt_once)
    assert result1.user_text == "What's the weather like?"
    assert agent.controller.state == TurnState.LISTENING  # interrupted, not idle
    assert len(audio.playback_log) == 1  # only first sentence played

    # Turn 2: runs to completion.
    result2 = agent.run_turn()
    assert result2.user_text == "Never mind, thanks"
    assert agent.controller.state == TurnState.IDLE
    assert len(audio.playback_log) == 2  # one more sentence played (single-sentence reply)

    # Memory carries both turns.
    contents = [m.content for m in agent.memory.to_messages()]
    assert contents[0] == "You are a helpful assistant."
    assert "What's the weather like?" in contents
    assert "Never mind, thanks" in contents


def test_pipeline_with_tool_call_updates_memory_correctly():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    memory = ConversationMemory(system_prompt="You have access to tools.")
    memory.add_user_turn("What is 6 times 9?")

    llm = ScriptedLLM(
        replies=[
            json.dumps({"tool": "calculator", "arguments": {"expression": "6 * 9"}}),
            "6 times 9 is 54.",
        ]
    )

    result = run_tool_loop(llm, registry, memory.to_messages())
    memory.add_assistant_turn(result.reply)

    assert result.tool_result == 54
    assert memory.to_messages()[-1].content == "6 times 9 is 54."


def test_blocking_and_streaming_agents_produce_equivalent_transcripts():
    stt_text = "hello there"
    reply_text = "Hi! How can I help you today?"

    blocking_agent = VoiceAgent(
        audio=MockAudioBackend(),
        stt=MockTranscriber(scripted_transcripts=[stt_text]),
        llm=ScriptedLLM(replies=[reply_text]),
        tts=MockSynthesizer(),
    )
    streaming_agent = VoiceAgent(
        audio=MockAudioBackend(),
        stt=MockTranscriber(scripted_transcripts=[stt_text]),
        llm=ScriptedLLM(replies=[reply_text]),
        tts=MockSynthesizer(),
    )

    blocking_result = blocking_agent.run_turn_blocking()
    streaming_result = streaming_agent.run_turn_streaming()

    assert blocking_result.user_text == streaming_result.user_text
    assert blocking_result.assistant_text == streaming_result.assistant_text
