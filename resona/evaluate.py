"""Offline evaluation on labeled sessions (WAV + Audacity label file).

Reports what an interviewer or examiner will ask about:
  * event precision / recall / F1   (did we find each whistle, and only whistles?)
  * false alarms per hour            (how often would it annoy a real user?)
  * count accuracy                   (sessions where the final count is exactly right)
  * detection latency                (whistle start -> count increment)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .counter import WhistleCounter
from .detectors import build_detector
from .features import FeatureExtractor, Framer
from .audio import load_wav
from .synth import read_labels


@dataclass
class SessionResult:
    name: str
    duration_s: float
    truth: list
    predicted: list          # (t_start, t_end, t_confirmed)
    tp: int
    fp: int
    fn: int
    latencies: list
    probs: np.ndarray


def run_detector(y: np.ndarray, settings: Settings, detector: str = "dsp"):
    """Run the live pipeline offline. Returns (probs, predicted events)."""
    fx = FeatureExtractor(settings.audio)
    framer = Framer(settings.audio.n_fft, settings.audio.hop)
    det = build_detector(detector, settings)
    on, off = settings.effective_thresholds()
    c = settings.counter
    counter = WhistleCounter(on, off, c.min_duration, c.merge_gap)
    rate = settings.audio.frame_rate
    probs, confirmed = [], {}
    for i, frame in enumerate(framer.push(y)):
        p = det.process(fx.frame(frame))
        probs.append(p)
        for kind, ev in counter.update(p, i / rate):
            if kind == "start":
                confirmed[ev.index] = i / rate
    counter.flush(len(probs) / rate)
    preds = [(e.t_start, e.t_end, confirmed[e.index]) for e in counter.events]
    return np.array(probs), preds


def match_events(truth, preds, tolerance: float = 0.5):
    """Greedy one-to-one matching by temporal overlap (with tolerance)."""
    used, tp, lat = set(), 0, []
    for ts, te in truth:
        best, best_ov = None, 0.0
        for j, (ps, pe, _) in enumerate(preds):
            if j in used:
                continue
            ov = min(te + tolerance, pe) - max(ts - tolerance, ps)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None:
            used.add(best)
            tp += 1
            lat.append(max(0.0, preds[best][2] - ts))
    return tp, len(preds) - len(used), len(truth) - tp, lat


def evaluate_dir(data_dir: str | Path, settings: Settings, detector: str = "dsp") -> list[SessionResult]:
    results = []
    for wav in sorted(Path(data_dir).glob("*.wav")):
        y = load_wav(wav, settings.audio.sample_rate)
        truth = read_labels(wav.with_suffix(".txt"))
        probs, preds = run_detector(y, settings, detector)
        tp, fp, fn, lat = match_events(truth, preds)
        results.append(SessionResult(wav.stem, len(y) / settings.audio.sample_rate,
                                     truth, preds, tp, fp, fn, lat, probs))
    return results


def summarise(results: list[SessionResult]) -> dict:
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    hours = sum(r.duration_s for r in results) / 3600
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lats = [l for r in results for l in r.latencies]
    exact = sum(len(r.predicted) == len(r.truth) for r in results)
    return {
        "sessions": len(results),
        "hours": round(hours, 3),
        "whistles": tp + fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_hour": round(fp / hours, 2) if hours else 0.0,
        "count_accuracy": round(exact / len(results), 4) if results else 0.0,
        "latency_mean_s": round(float(np.mean(lats)), 3) if lats else None,
        "latency_p95_s": round(float(np.percentile(lats, 95)), 3) if lats else None,
    }


def print_report(results: list[SessionResult], summary: dict) -> None:
    print(f"\n{'session':<24}{'truth':>7}{'pred':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
    for r in results:
        print(f"{r.name:<24}{len(r.truth):>7}{len(r.predicted):>7}{r.tp:>5}{r.fp:>5}{r.fn:>5}")
    print("\nSummary")
    for k, v in summary.items():
        print(f"  {k:<24}{v}")
