"""Streaming spectral features.

The exact same Framer + FeatureExtractor is used offline (training,
evaluation) and online (live server). This avoids train/serve skew, a
common reason audio models work in notebooks but fail in real time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import AudioConfig


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, center, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (freqs - lo) / max(center - lo, 1e-9)
        falling = (hi - freqs) / max(hi - center, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))
    # Slaney-style area normalisation keeps bands comparable.
    enorm = 2.0 / (hz_points[2:] - hz_points[:-2])
    fb *= enorm[:, None]
    return fb


class Framer:
    """Turns an arbitrary stream of sample blocks into overlapping frames."""

    def __init__(self, n_fft: int, hop: int):
        self.n_fft, self.hop = n_fft, hop
        self._buf = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> Iterator[np.ndarray]:
        self._buf = np.concatenate([self._buf, block.astype(np.float32, copy=False)])
        while len(self._buf) >= self.n_fft:
            yield self._buf[: self.n_fft]
            self._buf = self._buf[self.hop:]


@dataclass
class FrameFeatures:
    power: np.ndarray    # linear power spectrum, (n_fft // 2 + 1,)
    logmel: np.ndarray   # log-mel energies in dB, (n_mels,)


class FeatureExtractor:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.window = np.hanning(cfg.n_fft).astype(np.float32)
        self.win_norm = float(np.sum(self.window ** 2))
        self.mel_fb = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
        self.freqs = np.linspace(0, cfg.sample_rate / 2, cfg.n_fft // 2 + 1)

    def frame(self, frame: np.ndarray) -> FrameFeatures:
        spec = np.fft.rfft(frame * self.window)
        power = (spec.real ** 2 + spec.imag ** 2) / self.win_norm
        mel = self.mel_fb @ power
        logmel = 10.0 * np.log10(mel + 1e-10)
        return FrameFeatures(power=power.astype(np.float32), logmel=logmel.astype(np.float32))

    def normalise_for_display(self, logmel: np.ndarray) -> np.ndarray:
        lo, hi = self.cfg.db_floor, self.cfg.db_ceil
        return np.clip((logmel - lo) / (hi - lo), 0.0, 1.0)


def logmel_spectrogram(y: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Offline log-mel spectrogram, shape (frames, n_mels), identical to the live path."""
    fx = FeatureExtractor(cfg)
    framer = Framer(cfg.n_fft, cfg.hop)
    frames = [fx.frame(f).logmel for f in framer.push(y)]
    if not frames:
        return np.zeros((0, cfg.n_mels), dtype=np.float32)
    return np.stack(frames)
