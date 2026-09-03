from voiceagents.llm import EchoLLM, Message, ScriptedLLM


def test_scripted_llm_returns_replies_in_order():
    llm = ScriptedLLM(replies=["first", "second"])

    assert llm.complete([Message("user", "hi")]) == "first"
    assert llm.complete([Message("user", "hi again")]) == "second"


def test_scripted_llm_returns_empty_string_past_end_of_script():
    llm = ScriptedLLM(replies=["only"])

    llm.complete([])
    assert llm.complete([]) == ""


def test_scripted_llm_records_received_messages():
    llm = ScriptedLLM(replies=["ok"])
    messages = [Message("system", "sys"), Message("user", "hello")]

    llm.complete(messages)

    assert llm.received_messages == [messages]


def test_scripted_llm_stream_reconstructs_full_reply():
    llm = ScriptedLLM(replies=["hello there friend"])

    chunks = list(llm.stream([Message("user", "hi")]))

    assert "".join(chunks) == "hello there friend"
    assert len(chunks) == 3


def test_echo_llm_echoes_last_user_message():
    llm = EchoLLM(prefix="Echo: ")

    reply = llm.complete(
        [Message("system", "sys"), Message("user", "first"), Message("user", "second")]
    )

    assert reply == "Echo: second"


def test_echo_llm_handles_no_user_messages():
    llm = EchoLLM()

    reply = llm.complete([Message("system", "sys")])

    assert reply == "You said: "
