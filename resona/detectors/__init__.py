"""Whistle detectors. Each returns a whistle probability for every 20 ms frame."""
from .base import Detector
from .dsp import DSPDetector


def build_detector(name: str, settings):
    name = name.lower()
    if name == "dsp":
        return DSPDetector(settings)
    if name == "cnn":
        from .cnn import CNNDetector  # imported lazily so the DSP path never needs PyTorch
        return CNNDetector(settings)
    raise ValueError(f"Unknown detector '{name}'. Use 'dsp' or 'cnn'.")


__all__ = ["Detector", "DSPDetector", "build_detector"]
