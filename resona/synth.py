"""Synthetic kitchen sessions for development, demos and unit tests.

IMPORTANT: synthetic audio only proves the *pipeline* works end to end
before you have real recordings. Never report accuracy on synthetic data
in your thesis; train and evaluate on real cooker recordings.

Each session is a WAV file plus an Audacity-compatible label file
(start<TAB>end<TAB>label), the same format you will use to annotate
real recordings, so every downstream tool treats both identically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


def _band_noise(rng, n, sr, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    x = sosfilt(sos, rng.standard_normal(n))
    return x / (np.std(x) + 1e-9)


def _envelope(n, sr, attack=0.25, release=0.4, wobble=0.0, rng=None):
    t = np.arange(n) / sr
    env = np.minimum(1.0, t / attack) * np.minimum(1.0, (t[-1] - t) / release)
    if wobble and rng is not None:
        env *= 1.0 + wobble * np.sin(2 * np.pi * rng.uniform(3, 7) * t)
    return np.clip(env, 0, None)


def make_whistle(rng, sr, dur):
    """Turbulent steam hiss with a resonant tone that rises as pressure releases."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    hiss = _band_noise(rng, n, sr, rng.uniform(1800, 2500), rng.uniform(5500, 7500))
    f0 = rng.uniform(2600, 4200)
    sweep = f0 * (1 + 0.04 * np.minimum(1, t / 0.8)) + 25 * np.sin(2 * np.pi * 5.5 * t)
    tone = np.sin(2 * np.pi * np.cumsum(sweep) / sr)
    sig = 0.8 * hiss + rng.uniform(0.3, 0.9) * tone
    return sig * _envelope(n, sr, 0.2, 0.5, wobble=0.15, rng=rng)


def make_distractor(rng, sr, kind, dur):
    n = int(dur * sr)
    t = np.arange(n) / sr
    if kind == "mixer":         # loud, motor-dominated, mostly low/mid band
        motor = sum(np.sin(2 * np.pi * k * 290 * t) / k for k in range(1, 6))
        sig = 0.6 * motor + 0.8 * _band_noise(rng, n, sr, 120, 2500)
        return 1.4 * sig * _envelope(n, sr, 0.3, 0.3)
    if kind == "speech":        # syllable-rate modulated mid-band noise
        syll = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(3, 5) * t)) ** 2
        return 0.5 * _band_noise(rng, n, sr, 250, 3200) * syll * _envelope(n, sr, 0.1, 0.1)
    if kind == "sizzle":        # tadka: crackly, high band but quiet and impulsive
        crackle = (rng.random(n) < 0.004).astype(float)
        crackle = np.convolve(crackle, np.hanning(64), mode="same")
        return 0.25 * _band_noise(rng, n, sr, 2000, 7000) * (0.3 + 3 * crackle) * _envelope(n, sr, 0.5, 0.5)
    if kind == "clang":         # utensil hit: short and bright
        k = min(n, int(0.35 * sr))
        hit = np.zeros(n)
        hit[:k] = np.sin(2 * np.pi * rng.uniform(2500, 4500) * t[:k]) * np.exp(-t[:k] * 14)
        return 1.2 * hit
    raise ValueError(kind)


def make_session(seed: int = 0, minutes: float = 3.0, sr: int = 16_000,
                 n_whistles: int | None = None):
    """Returns (audio float32, list of (start, end, label))."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * sr)
    t = np.arange(n) / sr
    # Room tone: exhaust fan hum + broadband floor.
    bg = 0.05 * _band_noise(rng, n, sr, 60, 4000) + 0.03 * np.sin(2 * np.pi * 100 * t)
    y = bg.copy()
    labels = []

    n_whistles = n_whistles if n_whistles is not None else int(rng.integers(2, 6))
    slots = np.sort(rng.choice(np.arange(15, minutes * 60 - 10, 12), size=n_whistles, replace=False))
    for start in slots:
        start = float(start + rng.uniform(0, 4))
        dur = float(rng.uniform(2.0, 5.0))
        w = make_whistle(rng, sr, dur) * rng.uniform(0.35, 0.7)
        i = int(start * sr)
        y[i:i + len(w)] += w[: max(0, min(len(w), n - i))]
        labels.append((start, start + dur, "whistle"))

    for kind in ["mixer", "speech", "speech", "sizzle", "clang", "clang", "mixer"]:
        dur = {"mixer": rng.uniform(4, 9), "speech": rng.uniform(3, 8),
               "sizzle": rng.uniform(4, 10), "clang": 0.5}[kind]
        start = float(rng.uniform(2, minutes * 60 - dur - 1))
        d = make_distractor(rng, sr, kind, dur) * rng.uniform(0.2, 0.5)
        i = int(start * sr)
        y[i:i + len(d)] += d[: max(0, min(len(d), n - i))]

    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    return y.astype(np.float32), labels


def write_session(out_dir: str | Path, name: str, y: np.ndarray, labels, sr: int = 16_000) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wav = out / f"{name}.wav"
    wavfile.write(wav, sr, (y * 32767).astype(np.int16))
    (out / f"{name}.txt").write_text("".join(f"{s:.3f}\t{e:.3f}\t{l}\n" for s, e, l in labels))
    return wav


def read_labels(path: str | Path, label: str = "whistle"):
    """Read an Audacity label file -> list of (start, end)."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3 and parts[2].strip().lower() == label:
            out.append((float(parts[0]), float(parts[1])))
    return out
