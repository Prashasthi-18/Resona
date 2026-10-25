"""Audio sources: live microphone or a WAV file played back in real time."""
from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_wav(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load any PCM/float WAV as mono float32 in [-1, 1] at the target rate."""
    sr, y = wavfile.read(str(path))
    if y.dtype.kind == "i":
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    elif y.dtype.kind == "u":
        y = (y.astype(np.float32) - 128.0) / 128.0
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != sample_rate:
        g = np.gcd(sr, sample_rate)
        y = resample_poly(y, sample_rate // g, sr // g).astype(np.float32)
    return y


class FileSource:
    """Streams a WAV file in blocks; paced to wall-clock time when realtime=True."""

    def __init__(self, path: str | Path, sample_rate: int, block: int = 1600,
                 realtime: bool = True, loop: bool = False):
        self.y = load_wav(path, sample_rate)
        self.sr, self.block, self.realtime, self.loop = sample_rate, block, realtime, loop
        self.name = Path(path).name
        self.pos = 0

    def blocks(self, running) -> Iterator[np.ndarray]:
        dt = self.block / self.sr
        next_t = time.perf_counter()
        while running():
            if self.pos >= len(self.y):
                if not self.loop:
                    return
                self.pos = 0
            chunk = self.y[self.pos:self.pos + self.block]
            self.pos += self.block
            yield chunk
            if self.realtime:
                next_t += dt
                time.sleep(max(0.0, next_t - time.perf_counter()))

    def close(self) -> None:
        pass


class MicSource:
    """Live microphone via sounddevice (PortAudio)."""

    def __init__(self, sample_rate: int, block: int = 1600, device=None):
        import sounddevice as sd  # imported here so file mode works without PortAudio
        self.sr, self.block = sample_rate, block
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self.name = "microphone"

        def callback(indata, frames, time_info, status):
            try:
                self.q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # drop rather than lag behind real time

        self.stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                     blocksize=block, device=device, callback=callback)
        self.stream.start()

    def blocks(self, running) -> Iterator[np.ndarray]:
        while not self.q.empty():   # discard audio captured while paused
            self.q.get_nowait()
        while running():
            try:
                yield self.q.get(timeout=0.5)
            except queue.Empty:
                continue

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()
