"""The real-time pipeline.

    audio source -> Framer -> FeatureExtractor -> Detector -> WhistleCounter
                                     |                |            |
                                     +------> UI frames + events <-+

Runs on a background thread and reports everything through a single
`emit(message: dict)` callback, so it can drive a WebSocket server, a CLI,
or a test harness without changes.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from .config import Settings
from .counter import WhistleCounter
from .detectors import build_detector
from .features import FeatureExtractor, Framer

log = logging.getLogger("resona.engine")


class Engine:
    def __init__(self, settings: Settings, source_factory: Callable[[], object],
                 emit: Callable[[dict], None], detector: str = "dsp", ui_decimate: int = 2):
        self.s = settings
        self.source_factory = source_factory
        self.emit = emit
        self.ui_decimate = ui_decimate
        self.fx = FeatureExtractor(settings.audio)
        self.framer = Framer(settings.audio.n_fft, settings.audio.hop)
        self.detector = build_detector(detector, settings)
        on, off = settings.effective_thresholds()
        c = settings.counter
        self.counter = WhistleCounter(on, off, c.min_duration, c.merge_gap)
        self.source = None
        self.source_name = "none"
        self.listening = False
        self.frame_idx = 0
        self.session_started = time.time()
        self.target_announced = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self.listening:
            return
        if self.source is None:
            self.source = self.source_factory()
            self.source_name = getattr(self.source, "name", "audio")
        self.listening = True
        self._thread = threading.Thread(target=self._run, name="resona-engine", daemon=True)
        self._thread.start()
        self.emit_state()

    def stop(self) -> None:
        self.listening = False
        if self._thread:
            self._thread.join(timeout=2)
        self.emit_state()

    def shutdown(self) -> None:
        self.stop()
        if self.source is not None:
            self.source.close()

    def reset(self) -> None:
        with self._lock:
            self.counter.reset()
            self.detector.reset()
            self.framer.reset()
            self.frame_idx = 0
            self.session_started = time.time()
            self.target_announced = False
        self.emit({"type": "reset"})
        self.emit_state()

    def set_target(self, n: int) -> None:
        self.s.target_count = max(1, min(20, int(n)))
        self.target_announced = self.counter.count >= self.s.target_count
        self.emit_state()

    def set_sensitivity(self, v: float) -> None:
        self.s.sensitivity = float(np.clip(v, 0.0, 1.0))
        on, off = self.s.effective_thresholds()
        with self._lock:
            self.counter.on_threshold, self.counter.off_threshold = on, off
        self.emit_state()

    def set_detector(self, name: str) -> None:
        try:
            det = build_detector(name, self.s)
        except Exception as exc:  # e.g. no trained model yet
            self.emit({"type": "error", "message": str(exc)})
            return
        with self._lock:
            self.detector = det
        self.emit_state()

    def state(self) -> dict:
        on, off = self.s.effective_thresholds()
        return {
            "type": "state",
            "listening": self.listening,
            "source": self.source_name,
            "detector": self.detector.name,
            "count": self.counter.count,
            "target": self.s.target_count,
            "sensitivity": self.s.sensitivity,
            "on_threshold": on,
            "off_threshold": off,
            "events": [e.to_dict() for e in self.counter.events],
            "t": self.frame_idx / self.s.audio.frame_rate,
            "n_mels": self.s.audio.n_mels,
            "frame_rate": self.s.audio.frame_rate / self.ui_decimate,
        }

    def emit_state(self) -> None:
        self.emit(self.state())

    # --------------------------------------------------------------------- loop
    def _run(self) -> None:
        rate = self.s.audio.frame_rate
        spec_acc = None
        prob_acc = 0.0
        try:
            for block in self.source.blocks(lambda: self.listening):
                level_db = 20 * np.log10(np.sqrt(np.mean(block ** 2)) + 1e-9)
                with self._lock:
                    for frame in self.framer.push(block):
                        feats = self.fx.frame(frame)
                        p = self.detector.process(feats)
                        t = self.frame_idx / rate
                        self.frame_idx += 1
                        for kind, ev in self.counter.update(p, t):
                            self._on_event(kind, ev)

                        col = self.fx.normalise_for_display(feats.logmel)
                        spec_acc = col if spec_acc is None else np.maximum(spec_acc, col)
                        prob_acc = max(prob_acc, p)
                        if self.frame_idx % self.ui_decimate == 0:
                            self.emit({
                                "type": "frame",
                                "t": round(t, 3),
                                "spec": np.round(spec_acc, 3).tolist(),
                                "prob": round(prob_acc, 4),
                                "level": round(float(level_db), 1),
                                "active": self.counter.active,
                            })
                            spec_acc, prob_acc = None, 0.0
        except Exception:
            log.exception("Engine crashed")
            self.emit({"type": "error", "message": "Audio engine stopped unexpectedly. Check the server log.",
                       "critical": True})
        finally:
            if self.listening:  # source ran out (end of file)
                for kind, ev in self.counter.flush(self.frame_idx / rate):
                    self._on_event(kind, ev)
                self.listening = False
                self.emit({"type": "source_ended"})
                self.emit_state()

    def _on_event(self, kind: str, ev) -> None:
        self.emit({"type": f"whistle_{kind}", "count": self.counter.count, "event": ev.to_dict()})
        if kind == "start" and not self.target_announced and self.counter.count >= self.s.target_count:
            self.target_announced = True
            self.emit({"type": "target_reached", "count": self.counter.count,
                       "target": self.s.target_count})
