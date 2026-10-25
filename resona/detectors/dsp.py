"""Rule-based baseline detector.

Idea: a cooker whistle is loud, and most of its energy sits in a
high-frequency band. We score each frame on two features:

  1. band ratio  - share of spectral energy inside [band_low, band_high]
  2. band SNR    - band level above an adaptive noise floor, in dB

Each feature passes through a logistic curve and the two are multiplied,
so a frame must be BOTH loud and high-pitched to score high. A mixer
grinder is loud but mostly low-frequency; distant chatter is mid-band but
quiet. The noise floor adapts to the room so the detector works in quiet
and noisy kitchens alike.
"""
from __future__ import annotations

import numpy as np

from ..features import FeatureExtractor, FrameFeatures
from .base import Detector


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class DSPDetector(Detector):
    name = "dsp"

    def __init__(self, settings):
        self.cfg = settings.dsp
        freqs = FeatureExtractor(settings.audio).freqs
        self.band = (freqs >= self.cfg.band_low) & (freqs <= self.cfg.band_high)
        self.reset()

    def reset(self) -> None:
        self.noise_floor: float | None = None
        self.score = 0.0
        self.last = {"ratio": 0.0, "snr": 0.0}

    def process(self, feats: FrameFeatures) -> float:
        p = feats.power
        total = float(p.sum()) + 1e-12
        band_energy = float(p[self.band].sum()) + 1e-12
        ratio = band_energy / total
        band_db = 10.0 * np.log10(band_energy)

        if self.noise_floor is None:
            self.noise_floor = band_db
        # Minimum-tracking floor: falls quickly, rises slowly, frozen while a whistle sounds.
        if band_db < self.noise_floor:
            self.noise_floor += 0.2 * (band_db - self.noise_floor)
        elif self.score < 0.3:
            self.noise_floor += 0.0008 * (band_db - self.noise_floor)
        snr = band_db - self.noise_floor

        raw = _sigmoid(self.cfg.ratio_slope * (ratio - self.cfg.ratio_center)) * \
              _sigmoid(self.cfg.snr_slope * (snr - self.cfg.snr_center))
        a = self.cfg.smoothing
        self.score = (1 - a) * self.score + a * raw
        self.last = {"ratio": ratio, "snr": snr}
        return self.score
