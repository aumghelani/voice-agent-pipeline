"""Turn-taking state machine with barge-in (interruption) support.

Three states: LISTENING (waiting for/capturing user speech), THINKING
(STT + LLM in flight), SPEAKING (TTS audio playing). A CancelToken models
cooperative cancellation: playback backends are expected to check
`is_set()` periodically (e.g. per audio block) rather than being killed at
the OS level, so partial audio can still fade out cleanly.

This module only models the state machine and cancellation primitive; it
does not depend on any concrete audio/STT/LLM/TTS implementation, so it can
be tested and reused independently of VoiceAgent.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto


class TurnState(Enum):
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    IDLE = auto()


class InvalidTransition(Exception):
    pass


# Only these transitions are legal; anything else is a bug in the caller.
_ALLOWED_TRANSITIONS: dict[TurnState, set[TurnState]] = {
    TurnState.IDLE: {TurnState.LISTENING},
    # LISTENING -> LISTENING is legal: after a barge-in returns to
    # LISTENING, the next turn starts by re-arming the mic, i.e. entering
    # LISTENING again rather than resuming a stale in-progress one.
    TurnState.LISTENING: {TurnState.LISTENING, TurnState.THINKING, TurnState.IDLE},
    TurnState.THINKING: {TurnState.SPEAKING, TurnState.LISTENING, TurnState.IDLE},
    TurnState.SPEAKING: {TurnState.LISTENING, TurnState.IDLE},
}


class CancelToken:
    """Wraps a threading.Event as a cooperative cancellation signal for
    in-progress playback. A fresh token is created per SPEAKING phase so a
    stale cancel from a previous turn can never bleed into a new one."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class TurnTakingController:
    state: TurnState = TurnState.IDLE
    history: list[TurnState] = field(default_factory=list)
    _active_cancel_token: CancelToken | None = field(default=None, repr=False)

    def transition(self, new_state: TurnState) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTransition(
                f"Cannot transition from {self.state.name} to {new_state.name}"
            )
        self.history.append(self.state)
        self.state = new_state
        if new_state == TurnState.SPEAKING:
            self._active_cancel_token = CancelToken()
        elif self.state != TurnState.SPEAKING:
            self._active_cancel_token = None

    def start_listening(self) -> None:
        self.transition(TurnState.LISTENING)

    def start_thinking(self) -> None:
        self.transition(TurnState.THINKING)

    def start_speaking(self) -> CancelToken:
        self.transition(TurnState.SPEAKING)
        assert self._active_cancel_token is not None
        return self._active_cancel_token

    def go_idle(self) -> None:
        self.transition(TurnState.IDLE)

    def barge_in(self) -> bool:
        """Cancel in-progress speech and return to LISTENING. Returns False
        (no-op) if not currently speaking."""
        if self.state != TurnState.SPEAKING:
            return False
        if self._active_cancel_token is not None:
            self._active_cancel_token.cancel()
        self.transition(TurnState.LISTENING)
        return True
