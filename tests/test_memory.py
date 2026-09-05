from voiceagents.agent.memory import ConversationMemory
from voiceagents.llm.base import Message


def test_to_messages_includes_system_prompt_first():
    memory = ConversationMemory(system_prompt="Be helpful.")
    memory.add_user_turn("hi")

    messages = memory.to_messages()

    assert messages[0] == Message("system", "Be helpful.")
    assert messages[1] == Message("user", "hi")


def test_to_messages_omits_system_when_empty():
    memory = ConversationMemory(system_prompt="")
    memory.add_user_turn("hi")

    messages = memory.to_messages()

    assert messages == [Message("user", "hi")]


def test_records_alternating_turns_in_order():
    memory = ConversationMemory()
    memory.add_user_turn("hello")
    memory.add_assistant_turn("hi there")
    memory.add_user_turn("how are you")

    messages = memory.to_messages()

    assert [m.content for m in messages] == ["hello", "hi there", "how are you"]


def test_sliding_window_trims_oldest_turns():
    memory = ConversationMemory(max_turns=2)
    for i in range(5):
        memory.add_user_turn(f"user {i}")
        memory.add_assistant_turn(f"assistant {i}")

    messages = memory.to_messages()

    assert len(messages) == 4  # max_turns=2 -> 4 messages (2 user + 2 assistant)
    assert messages[0].content == "user 3"
    assert messages[-1].content == "assistant 4"


def test_clear_removes_all_turns():
    memory = ConversationMemory(system_prompt="sys")
    memory.add_user_turn("hi")

    memory.clear()

    assert len(memory) == 0
    assert memory.to_messages() == [Message("system", "sys")]


def test_len_reflects_turn_count_not_including_system():
    memory = ConversationMemory(system_prompt="sys")
    memory.add_user_turn("hi")
    memory.add_assistant_turn("hello")

    assert len(memory) == 2
