from voiceagents.agent.session_store import SessionStore


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def test_set_and_get_round_trips_value():
    store = SessionStore(ttl_s=100, clock=_FakeClock())

    store.set("s1", {"turns": 1})

    assert store.get("s1") == {"turns": 1}


def test_get_returns_none_for_unknown_session():
    store = SessionStore(clock=_FakeClock())

    assert store.get("missing") is None


def test_get_returns_none_and_evicts_after_ttl_elapses():
    clock = _FakeClock()
    store = SessionStore(ttl_s=10, clock=clock)
    store.set("s1", "value")

    clock.now = 11

    assert store.get("s1") is None
    assert len(store) == 0


def test_touch_extends_ttl():
    clock = _FakeClock()
    store = SessionStore(ttl_s=10, clock=clock)
    store.set("s1", "value")

    clock.now = 9
    assert store.touch("s1") is True

    clock.now = 15  # would have expired at t=10 without the touch at t=9
    assert store.get("s1") == "value"


def test_touch_returns_false_for_expired_session():
    clock = _FakeClock()
    store = SessionStore(ttl_s=10, clock=clock)
    store.set("s1", "value")

    clock.now = 11

    assert store.touch("s1") is False


def test_delete_removes_session():
    store = SessionStore(clock=_FakeClock())
    store.set("s1", "value")

    store.delete("s1")

    assert store.get("s1") is None


def test_contains_reflects_liveness():
    clock = _FakeClock()
    store = SessionStore(ttl_s=10, clock=clock)
    store.set("s1", "value")

    assert "s1" in store

    clock.now = 11
    assert "s1" not in store


def test_update_overwrites_value_and_resets_ttl():
    clock = _FakeClock()
    store = SessionStore(ttl_s=10, clock=clock)
    store.set("s1", "old")

    clock.now = 5
    store.update("s1", "new")

    clock.now = 14  # would be expired if TTL hadn't reset at t=5
    assert store.get("s1") == "new"
