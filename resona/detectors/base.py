from __future__ import annotations

from abc import ABC, abstractmethod

from ..features import FrameFeatures


class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def process(self, feats: FrameFeatures) -> float:
        """Return P(whistle) in [0, 1] for the latest frame."""

    def reset(self) -> None:  # pragma: no cover - optional
        pass
