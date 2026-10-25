"""CNN detector: runs the trained WhistleNet on a sliding 1-second window."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from ..features import FrameFeatures
from .base import Detector


class CNNDetector(Detector):
    name = "cnn"

    def __init__(self, settings):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The neural network needs PyTorch, which isn't installed. "
                "Install it with `pip install torch`, or keep using Signal rules."
            ) from exc
        from ..model import WhistleNet

        path = Path(settings.model.weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Train one with `python -m resona train`, "
                "or keep using Signal rules."
            )
        ckpt = torch.load(path, map_location="cpu")
        self.torch = torch
        self.model = WhistleNet(n_mels=settings.audio.n_mels)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.mean = float(ckpt["mean"])
        self.std = float(ckpt["std"])
        self.window = settings.model.window_frames
        self.every = settings.model.infer_every
        self.reset()

    def reset(self) -> None:
        self.buf: deque[np.ndarray] = deque(maxlen=self.window)
        self.counter = 0
        self.prob = 0.0

    def process(self, feats: FrameFeatures) -> float:
        self.buf.append(feats.logmel)
        self.counter += 1
        if len(self.buf) == self.window and self.counter % self.every == 0:
            x = (np.stack(self.buf) - self.mean) / self.std
            with self.torch.no_grad():
                t = self.torch.from_numpy(x.astype(np.float32))[None, None]
                self.prob = float(self.torch.sigmoid(self.model(t))[0, 0])
        return self.prob
