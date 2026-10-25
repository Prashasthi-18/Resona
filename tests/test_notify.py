"""Notifier tests against a fake Telegram + ntfy (no network needed)."""
import json
import urllib.parse

import pytest

from resona import notify
from resona.notify import Notifier, NotifyError

TOKEN = "123456789:AAH_fake_token_for_tests_000000"


class FakeServices:
    def __init__(self):
        self.calls = []
        self.updates = []
        self.fail_ntfy = 0

    def __call__(self, url, data=None, headers=None, timeout=15):
        payload = None
        if data:
            try:
                payload = json.loads(data)
            except ValueError:
                payload = dict(urllib.parse.parse_qsl(data.decode()))
        self.calls.append((url, payload))
        if url.startswith(notify.TELEGRAM_API):
            method = url.rsplit("/", 1)[-1]
            if "bad" in url:
                return {"ok": False, "description": "Unauthorized"}
            if method == "getMe":
                return {"ok": True, "result": {"username": "my_kitchen_bot"}}
            if method == "getUpdates":
                off = int((payload or {}).get("offset", 0))
                return {"ok": True, "result": [u for u in self.updates if u["update_id"] >= off]}
            if method == "sendMessage":
                return {"ok": True, "result": {}}
        if self.fail_ntfy:
            self.fail_ntfy -= 1
            raise notify.urllib.error.URLError("offline")
        return {"id": "x"}

    def sent(self, kind):
        if kind == "telegram":
            return [p for u, p in self.calls if u.endswith("sendMessage")]
        return [p for u, p in self.calls if not u.startswith(notify.TELEGRAM_API)]


@pytest.fixture
def svc(monkeypatch, tmp_path):
    fake = FakeServices()
    monkeypatch.setattr(notify, "http_json", fake)
    for k in ("RESONA_TELEGRAM_TOKEN", "RESONA_TELEGRAM_CHAT", "RESONA_NTFY_TOPIC"):
        monkeypatch.delenv(k, raising=False)
    fake.path = tmp_path / "notify.json"
    return fake


def make(svc):
    n = Notifier(svc.path, start_worker=False)
    n.retry_delays = (0, 0)
    return n


def test_telegram_setup_flow(svc):
    n = make(svc)
    st = n.connect_telegram(TOKEN)
    assert st["telegram"]["bot"] == "my_kitchen_bot" and not st["telegram"]["ready"]
    assert n.link_telegram()["linked"] is False             # user hasn't pressed Start yet
    svc.updates.append({"update_id": 7, "message": {"chat": {"id": 4242, "type": "private", "first_name": "Ananya"}}})
    st = n.link_telegram()
    assert st["linked"] and st["telegram"]["ready"] and st["telegram"]["name"] == "Ananya"
    assert svc.sent("telegram")[-1]["chat_id"] == "4242"    # confirmation message
    assert json.loads(svc.path.read_text())["telegram_chat_id"] == "4242"   # persisted
    assert TOKEN not in json.dumps(n.status())              # token never echoed back


def test_rejects_malformed_token(svc):
    with pytest.raises(NotifyError):
        make(svc).connect_telegram("not-a-token")


def test_ntfy_topic_generated_and_validated(svc):
    n = make(svc)
    topic = n.set_ntfy()["ntfy"]["topic"]
    assert topic.startswith("resona-") and len(topic) == len("resona-") + 12
    with pytest.raises(NotifyError):
        n.set_ntfy("bad topic!")


def test_event_mapping(svc):
    n = make(svc)
    n.set_ntfy("kitchen-test-topic")
    ev = lambda c: {"type": "whistle_start", "count": c, "event": {"t_start": 65.0}}
    n.handle_event(ev(1), target=3)
    n.handle_event(ev(3), target=3)                    # final whistle: covered by target_reached
    n.handle_event({"type": "target_reached", "count": 3}, target=3)
    n.handle_event(ev(4), target=3)                    # past the target -> warning
    n.handle_event({"type": "error", "message": "no model"}, target=3)   # not critical: no alert
    queued = [n._q.get_nowait() for _ in range(n._q.qsize())]
    titles = [q[0] for q in queued]
    assert titles == ["Whistle 1 of 3", "3 whistles reached", "Whistle 4, past your target"]
    assert queued[0][1] == "Heard 1:05 into the session."
    assert [q[2] for q in queued] == [False, True, True]


def test_each_whistle_can_be_turned_off(svc):
    n = make(svc)
    n.set_ntfy("kitchen-test-topic")
    n.set_prefs(False)
    n.handle_event({"type": "whistle_start", "count": 1, "event": {"t_start": 1.0}}, target=3)
    assert n._q.empty()


def test_delivery_to_both_channels_with_retry(svc):
    n = make(svc)
    n.connect_telegram(TOKEN)
    svc.updates.append({"update_id": 1, "message": {"chat": {"id": 1, "type": "private"}}})
    n.link_telegram()
    n.set_ntfy("kitchen-test-topic")
    svc.fail_ntfy = 1                                    # first ntfy attempt fails, retry succeeds
    r = n.send_now("3 whistles reached", "Time to turn off the stove.", urgent=True, tags=["rotating_light"])
    assert r["results"] == {"telegram": {"ok": True, "error": None}, "ntfy": {"ok": True, "error": None}}
    msg = svc.sent("ntfy")[-1]
    assert msg["topic"] == "kitchen-test-topic" and msg["priority"] == 5
    assert "<b>3 whistles reached</b>" in svc.sent("telegram")[-1]["text"]


def test_nothing_configured(svc):
    n = make(svc)
    n.send("x", "y")
    assert n._q.empty()
    with pytest.raises(NotifyError):
        n.send_now("x", "y")
