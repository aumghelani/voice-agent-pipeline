from typer.testing import CliRunner

from voiceagents.cli import app

runner = CliRunner()


def test_demo_command_runs_end_to_end_and_prints_turn():
    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0
    assert "User:" in result.stdout
    assert "Assistant:" in result.stdout
    assert "Hello, what can you do?" in result.stdout


def test_chat_command_falls_back_to_echo_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["chat"], input="hi there\nexit\n")

    assert result.exit_code == 0
    assert "ANTHROPIC_API_KEY is not set" in result.stdout
    assert "You said: hi there" in result.stdout


def test_chat_command_exits_cleanly_on_quit():
    result = runner.invoke(app, ["chat"], input="quit\n")

    assert result.exit_code == 0
