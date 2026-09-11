import pytest

from voiceagents.agent.turn_taking import (
    CancelToken,
    InvalidTransition,
    TurnState,
    TurnTakingController,
)


def test_starts_idle():
    controller = TurnTakingController()

    assert controller.state == TurnState.IDLE


def test_full_happy_path_cycle():
    controller = TurnTakingController()

    controller.start_listening()
    assert controller.state == TurnState.LISTENING

    controller.start_thinking()
    assert controller.state == TurnState.THINKING

    token = controller.start_speaking()
    assert controller.state == TurnState.SPEAKING
    assert isinstance(token, CancelToken)

    controller.go_idle()
    assert controller.state == TurnState.IDLE


def test_thinking_can_go_directly_back_to_listening_without_speaking():
    controller = TurnTakingController()
    controller.start_listening()
    controller.start_thinking()

    controller.start_listening()

    assert controller.state == TurnState.LISTENING


def test_listening_can_restart_listening_after_a_barge_in():
    # Regression: after barge_in() returns to LISTENING, the next turn
    # re-arms the mic by calling start_listening() again while already
    # LISTENING. This must be legal, not an InvalidTransition.
    controller = TurnTakingController()
    controller.start_listening()
    controller.start_thinking()
    controller.start_speaking()
    controller.barge_in()
    assert controller.state == TurnState.LISTENING

    controller.start_listening()

    assert controller.state == TurnState.LISTENING


def test_invalid_transition_raises():
    controller = TurnTakingController()  # IDLE

    with pytest.raises(InvalidTransition):
        controller.start_speaking()


def test_cannot_skip_thinking_from_listening_to_speaking():
    controller = TurnTakingController()
    controller.start_listening()

    with pytest.raises(InvalidTransition):
        controller.transition(TurnState.SPEAKING)


def test_history_records_prior_states():
    controller = TurnTakingController()

    controller.start_listening()
    controller.start_thinking()

    assert controller.history == [TurnState.IDLE, TurnState.LISTENING]


def test_barge_in_cancels_token_and_returns_to_listening():
    controller = TurnTakingController()
    controller.start_listening()
    controller.start_thinking()
    token = controller.start_speaking()

    result = controller.barge_in()

    assert result is True
    assert token.is_cancelled() is True
    assert controller.state == TurnState.LISTENING


def test_barge_in_is_noop_when_not_speaking():
    controller = TurnTakingController()
    controller.start_listening()

    result = controller.barge_in()

    assert result is False
    assert controller.state == TurnState.LISTENING


def test_each_speaking_phase_gets_a_fresh_cancel_token():
    controller = TurnTakingController()
    controller.start_listening()
    controller.start_thinking()
    first_token = controller.start_speaking()
    controller.barge_in()

    controller.start_thinking()
    second_token = controller.start_speaking()

    assert first_token is not second_token
    assert second_token.is_cancelled() is False


def test_cancel_token_starts_uncancelled():
    token = CancelToken()

    assert token.is_cancelled() is False

    token.cancel()

    assert token.is_cancelled() is True
