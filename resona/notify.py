"""Phone alerts through Telegram and the ntfy app.

Two channels, both set up from the dashboard (no code changes needed):

* Telegram - you create a bot with @BotFather, paste its token, then press
  Start in your bot's chat. Resona finds your chat id automatically.
* ntfy     - free push-notification app (Android and iPhone, no account).
  Resona generates a private topic; you subscribe to it in the app.

Settings are saved to notify.json (keep it out of git: it holds your bot
token). Environment variables override the file:
    RESONA_TELEGRAM_TOKEN, RESONA_TELEGRAM_CHAT, RESONA_NTFY_TOPIC,
    RESONA_NTFY_SERVER, RESONA_NOTIFY_FILE
Sending happens on a background thread with retries, so a slow network
never blocks audio processing.
"""
from __future__ import annotations

import html
import json
import logging
import os
import queue
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

log = logging.getLogger("resona.notify")
TELEGRAM_API = "https://api.telegram.org"
DEFAULT_NTFY = "https://ntfy.sh"
TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


class NotifyError(Exception):
    """A setup or delivery problem, phrased so it can be shown to the user."""


def http_json(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 15) -> dict:
    """Tiny HTTP helper (module-level so tests can replace it)."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def fmt_time(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


@dataclass
class NotifySettings:
    telegram_token: str = ""
    telegram_chat_id: str = ""
    telegram_bot: str = ""
    telegram_name: str = ""
    ntfy_server: str = DEFAULT_NTFY
    ntfy_topic: str = ""
    each_whistle: bool = True

    @classmethod
    def load(cls, path: Path) -> "NotifySettings":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            log.warning("Ignoring unreadable %s", path)
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        try:
            os.chmod(path, 0o600)  # the file holds a bot token
        except OSError:
            pass


# --------------------------------------------------------------------- Telegram
def telegram_call(token: str, method: str, params: dict | None = None) -> dict | list:
    url = f"{TELEGRAM_API}/bot{token}/{method}"
    data = urllib.parse.urlencode(params).encode() if params else None
    try:
        res = http_json(url, data)
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):
            raise NotifyError("Telegram didn’t accept this token. Copy it again from @BotFather.") from e
        try:
            desc = json.loads(e.read()).get("description", "")
        except Exception:
            desc = ""
        raise NotifyError(f"Telegram returned an error ({e.code}). {desc}".strip()) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise NotifyError("Couldn’t reach Telegram. Check this computer’s internet connection.") from e
    if not res.get("ok"):
        raise NotifyError(res.get("description", "Telegram request failed."))
    return res["result"]


# ------------------------------------------------------------------------ core
class Notifier:
    def __init__(self, path: str | Path | None = None,
                 on_result: Callable[[dict], None] | None = None, start_worker: bool = True):
        self.path = Path(path or os.getenv("RESONA_NOTIFY_FILE", "notify.json"))
        self.s = NotifySettings.load(self.path)
        env = os.environ
        self.s.telegram_token = env.get("RESONA_TELEGRAM_TOKEN", self.s.telegram_token)
        self.s.telegram_chat_id = env.get("RESONA_TELEGRAM_CHAT", self.s.telegram_chat_id)
        self.s.ntfy_topic = env.get("RESONA_NTFY_TOPIC", self.s.ntfy_topic)
        self.s.ntfy_server = env.get("RESONA_NTFY_SERVER", self.s.ntfy_server).rstrip("/")
        self.on_result = on_result
        self.retry_delays = (1.0, 3.0)
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        if start_worker:
            threading.Thread(target=self._worker, name="resona-notify", daemon=True).start()

    # -------------------------------------------------------------- status
    @property
    def telegram_ready(self) -> bool:
        return bool(self.s.telegram_token and self.s.telegram_chat_id)

    @property
    def ntfy_ready(self) -> bool:
        return bool(self.s.ntfy_topic)

    def status(self) -> dict:
        tok = self.s.telegram_token
        return {
            "telegram": {
                "ready": self.telegram_ready,
                "has_token": bool(tok),
                "token_hint": f"…{tok[-4:]}" if tok else "",
                "bot": self.s.telegram_bot,
                "name": self.s.telegram_name,
                "link": f"https://t.me/{self.s.telegram_bot}?start=resona" if self.s.telegram_bot else "",
            },
            "ntfy": {
                "ready": self.ntfy_ready,
                "server": self.s.ntfy_server,
                "topic": self.s.ntfy_topic,
                "link": f"{self.s.ntfy_server}/{self.s.ntfy_topic}" if self.s.ntfy_topic else "",
            },
            "each_whistle": self.s.each_whistle,
        }

    def _save(self) -> None:
        self.s.save(self.path)

    # ------------------------------------------------------- Telegram setup
    def connect_telegram(self, token: str) -> dict:
        token = (token or "").strip()
        if not re.fullmatch(r"\d{5,}:[A-Za-z0-9_-]{20,}", token):
            raise NotifyError("That doesn’t look like a bot token. It should look like 123456789:AA…")
        me = telegram_call(token, "getMe")
        # Skip messages sent before linking so an old chat isn't picked up by mistake.
        pending = telegram_call(token, "getUpdates", {"timeout": 0})
        if pending:
            telegram_call(token, "getUpdates", {"offset": pending[-1]["update_id"] + 1, "timeout": 0})
        with self._lock:
            self.s.telegram_token = token
            self.s.telegram_bot = me.get("username", "")
            self.s.telegram_chat_id = ""
            self.s.telegram_name = ""
            self._save()
        return self.status()

    def link_telegram(self) -> dict:
        """Look for a message to the bot; once found, remember that chat."""
        if not self.s.telegram_token:
            raise NotifyError("Connect a bot token first.")
        updates = telegram_call(self.s.telegram_token, "getUpdates", {"timeout": 0})
        chat = None
        for u in reversed(updates):
            msg = u.get("message") or u.get("edited_message")
            if msg and msg.get("chat", {}).get("type") == "private":
                chat = msg["chat"]
                break
        if chat is None:
            return {**self.status(), "linked": False}
        if updates:
            telegram_call(self.s.telegram_token, "getUpdates",
                          {"offset": updates[-1]["update_id"] + 1, "timeout": 0})
        with self._lock:
            self.s.telegram_chat_id = str(chat["id"])
            self.s.telegram_name = chat.get("first_name") or chat.get("username") or "you"
            self._save()
        self._send_telegram("Resona is connected",
                            "You’ll get a message here when your cooker whistles.", urgent=False)
        return {**self.status(), "linked": True}

    def disconnect_telegram(self) -> dict:
        with self._lock:
            self.s.telegram_token = self.s.telegram_chat_id = ""
            self.s.telegram_bot = self.s.telegram_name = ""
            self._save()
        return self.status()

    # ----------------------------------------------------------- ntfy setup
    def set_ntfy(self, topic: str | None = None, server: str | None = None) -> dict:
        topic = (topic or "").strip() or f"resona-{secrets.token_hex(6)}"
        if not TOPIC_RE.fullmatch(topic):
            raise NotifyError("Topic names need 6–64 letters, numbers, dashes or underscores.")
        with self._lock:
            self.s.ntfy_topic = topic
            if server:
                self.s.ntfy_server = server.strip().rstrip("/")
            self._save()
        return self.status()

    def disconnect_ntfy(self) -> dict:
        with self._lock:
            self.s.ntfy_topic = ""
            self._save()
        return self.status()

    def set_prefs(self, each_whistle: bool) -> dict:
        with self._lock:
            self.s.each_whistle = bool(each_whistle)
            self._save()
        return self.status()

    # -------------------------------------------------------------- sending
    def _send_telegram(self, title: str, body: str, urgent: bool) -> None:
        text = f"<b>{html.escape(title)}</b>\n{html.escape(body)}"
        telegram_call(self.s.telegram_token, "sendMessage", {
            "chat_id": self.s.telegram_chat_id, "text": text, "parse_mode": "HTML",
            "disable_notification": "false",
        })

    def _send_ntfy(self, title: str, body: str, urgent: bool, tags: list[str]) -> None:
        payload = {"topic": self.s.ntfy_topic, "title": title, "message": body,
                   "priority": 5 if urgent else 3, "tags": tags}
        try:
            http_json(self.s.ntfy_server, json.dumps(payload).encode(),
                      {"Content-Type": "application/json"})
        except urllib.error.HTTPError as e:
            raise NotifyError(f"ntfy returned an error ({e.code}).") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise NotifyError("Couldn’t reach ntfy. Check this computer’s internet connection.") from e

    def _deliver(self, title: str, body: str, urgent: bool, tags: list[str]) -> dict:
        jobs = []
        if self.telegram_ready:
            jobs.append(("telegram", lambda: self._send_telegram(title, body, urgent)))
        if self.ntfy_ready:
            jobs.append(("ntfy", lambda: self._send_ntfy(title, body, urgent, tags)))
        results = {}
        for name, job in jobs:
            err = None
            for attempt in range(len(self.retry_delays) + 1):
                try:
                    job()
                    err = None
                    break
                except NotifyError as e:
                    err = str(e)
                    if "token" in err.lower():   # retrying a bad token won't help
                        break
                    if attempt < len(self.retry_delays):
                        time.sleep(self.retry_delays[attempt])
            results[name] = {"ok": err is None, "error": err}
            if err:
                log.warning("%s alert failed: %s", name, err)
        return {"title": title, "results": results}

    def send(self, title: str, body: str, urgent: bool = False, tags: list[str] | None = None) -> None:
        """Queue an alert; returns immediately."""
        if self.telegram_ready or self.ntfy_ready:
            self._q.put((title, body, urgent, tags or []))

    def send_now(self, title: str, body: str, urgent: bool = False, tags: list[str] | None = None) -> dict:
        """Send synchronously (used by the 'Send test alert' button)."""
        if not (self.telegram_ready or self.ntfy_ready):
            raise NotifyError("Set up Telegram or the ntfy app first.")
        return self._deliver(title, body, urgent, tags or [])

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            result = self._deliver(*item)
            if self.on_result:
                try:
                    self.on_result(result)
                except Exception:
                    log.exception("on_result callback failed")

    # ------------------------------------------------- engine event mapping
    def handle_event(self, msg: dict, target: int) -> None:
        kind = msg.get("type")
        if kind == "whistle_start":
            n, t = msg["count"], msg["event"]["t_start"]
            if n > target:
                self.send(f"Whistle {n}, past your target",
                          f"You asked for {target}. The stove may still be on.", urgent=True,
                          tags=["warning"])
            elif n < target and self.s.each_whistle:
                self.send(f"Whistle {n} of {target}", f"Heard {fmt_time(t)} into the session.",
                          tags=["hot_springs"])
        elif kind == "target_reached":
            n = msg["count"]
            self.send(f"{n} whistle{'s' if n != 1 else ''} reached",
                      "Time to turn off the stove.", urgent=True, tags=["rotating_light"])
        elif kind == "error" and msg.get("critical"):
            self.send("Resona needs attention", msg.get("message", "Listening stopped."),
                      urgent=True, tags=["warning"])
