"""Central configuration for Resona.

Every tunable number lives here so experiments are reproducible and the
same settings are shared by training, evaluation and the live server.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    sample_rate: int = 16_000      # Hz; whistle energy sits well below the 8 kHz Nyquist limit
    n_fft: int = 1024              # 64 ms analysis window
    hop: int = 320                 # 20 ms hop -> 50 frames per second
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 8000.0
    db_floor: float = -85.0        # display / normalisation range for log-mel values
    db_ceil: float = -15.0

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop


@dataclass
class DSPConfig:
    """Rule-based baseline. Calibrate band_low/band_high from your own spectrograms."""
    band_low: float = 1500.0
    band_high: float = 7000.0
    ratio_center: float = 0.55     # fraction of energy that must sit in the whistle band
    ratio_slope: float = 14.0
    snr_center: float = 12.0       # dB above the adaptive noise floor
    snr_slope: float = 0.35
    smoothing: float = 0.3         # EMA factor applied to the per-frame score


@dataclass
class CounterConfig:
    on_threshold: float = 0.6      # probability needed to start a whistle
    off_threshold: float = 0.4     # probability below which it may end (hysteresis)
    min_duration: float = 0.6      # seconds a candidate must last to count
    merge_gap: float = 0.8         # dips shorter than this do not split one whistle


@dataclass
class ModelConfig:
    window_frames: int = 50        # 1.0 s of context per prediction
    infer_every: int = 5           # run the CNN every 5 frames (100 ms)
    weights_path: str = "models/resona_cnn.pt"


@dataclass
class Settings:
    audio: AudioConfig = field(default_factory=AudioConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    counter: CounterConfig = field(default_factory=CounterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    target_count: int = 3
    sensitivity: float = 0.5       # 0 = strict, 1 = eager; shifts counter thresholds

    def effective_thresholds(self) -> tuple[float, float]:
        shift = (0.5 - self.sensitivity) * 0.4
        on = min(0.95, max(0.2, self.counter.on_threshold + shift))
        off = min(on - 0.05, max(0.1, self.counter.off_threshold + shift))
        return on, off

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path | None) -> "Settings":
        if path is None or not Path(path).exists():
            return cls()
        raw = json.loads(Path(path).read_text())
        return cls(
            audio=AudioConfig(**raw.get("audio", {})),
            dsp=DSPConfig(**raw.get("dsp", {})),
            counter=CounterConfig(**raw.get("counter", {})),
            model=ModelConfig(**raw.get("model", {})),
            target_count=raw.get("target_count", 3),
            sensitivity=raw.get("sensitivity", 0.5),
        )
