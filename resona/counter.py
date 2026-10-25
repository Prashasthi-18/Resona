"""Turns a noisy per-frame probability into clean, countable whistle events.

State machine with hysteresis:

    IDLE --p>=on--> CANDIDATE --lasts min_duration--> ACTIVE (count += 1)
                        |                               |
                     p<off                            p<off
                        v                               v
                      IDLE        <--gap>=merge_gap-- RELEASING --p>=on--> ACTIVE

* Hysteresis (on > off) stops flicker around a single threshold.
* min_duration rejects clicks, clangs and short beeps.
* merge_gap keeps one whistle that briefly dips (steam sputter) as ONE count.
* The count increments as soon as a whistle is confirmed, not when it
  ends, which keeps alert latency low.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WhistleEvent:
    index: int
    t_start: float
    t_end: float | None = None
    peak: float = 0.0

    @property
    def duration(self) -> float | None:
        return None if self.t_end is None else self.t_end - self.t_start

    def to_dict(self) -> dict:
        return {"index": self.index, "t_start": round(self.t_start, 3),
                "t_end": None if self.t_end is None else round(self.t_end, 3),
                "duration": None if self.duration is None else round(self.duration, 3),
                "peak": round(self.peak, 3)}


@dataclass
class WhistleCounter:
    on_threshold: float = 0.6
    off_threshold: float = 0.4
    min_duration: float = 0.6
    merge_gap: float = 0.8
    state: str = "idle"
    count: int = 0
    events: list[WhistleEvent] = field(default_factory=list)
    _start: float = 0.0
    _peak: float = 0.0
    _release_t: float = 0.0

    def reset(self) -> None:
        self.state, self.count, self.events = "idle", 0, []

    @property
    def active(self) -> bool:
        return self.state in ("active", "releasing")

    def update(self, p: float, t: float) -> list[tuple[str, WhistleEvent]]:
        """Feed one frame. Returns ('start'|'end', event) notifications."""
        out: list[tuple[str, WhistleEvent]] = []
        s = self.state
        if s == "idle":
            if p >= self.on_threshold:
                self.state, self._start, self._peak = "candidate", t, p
        elif s == "candidate":
            if p < self.off_threshold:
                self.state = "idle"
            else:
                self._peak = max(self._peak, p)
                if t - self._start >= self.min_duration:
                    self.count += 1
                    ev = WhistleEvent(self.count, self._start, peak=self._peak)
                    self.events.append(ev)
                    self.state = "active"
                    out.append(("start", ev))
        elif s == "active":
            self.events[-1].peak = max(self.events[-1].peak, p)
            if p < self.off_threshold:
                self.state, self._release_t = "releasing", t
        elif s == "releasing":
            if p >= self.on_threshold:
                self.state = "active"
            elif t - self._release_t >= self.merge_gap:
                ev = self.events[-1]
                ev.t_end = self._release_t
                self.state = "idle"
                out.append(("end", ev))
        return out

    def flush(self, t: float) -> list[tuple[str, WhistleEvent]]:
        """Close any open whistle at end of stream."""
        if self.active:
            ev = self.events[-1]
            ev.t_end = self._release_t if self.state == "releasing" else t
            self.state = "idle"
            return [("end", ev)]
        self.state = "idle"
        return []
