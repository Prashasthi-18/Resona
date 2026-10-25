"""Resona command-line interface.

    python -m resona serve     [--source mic|FILE.wav] [--detector dsp|cnn]
    python -m resona synth     --out data/synthetic --sessions 20
    python -m resona record    --out data/real --name kitchen01 --minutes 15
    python -m resona train     --data data/real
    python -m resona evaluate  --data data/test --detector dsp
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Settings


def cmd_serve(a, s):
    import uvicorn
    from .audio import FileSource, MicSource
    from .server import create_app

    if a.source == "mic":
        factory = lambda: MicSource(s.audio.sample_rate, device=a.device)
    else:
        factory = lambda: FileSource(a.source, s.audio.sample_rate, loop=a.loop)
    app = create_app(s, factory, detector=a.detector, autostart=a.autostart)
    print(f"\n  Resona dashboard: http://{a.host}:{a.port}\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def cmd_synth(a, s):
    from .synth import make_session, write_session
    for i in range(a.sessions):
        y, labels = make_session(seed=a.seed + i, minutes=a.minutes, sr=s.audio.sample_rate)
        p = write_session(a.out, f"synthetic_{i:03d}", y, labels, s.audio.sample_rate)
        print(f"{p}  ({len(labels)} whistles)")


def cmd_record(a, s):
    import numpy as np
    import sounddevice as sd
    from scipy.io import wavfile
    from pathlib import Path

    sr = s.audio.sample_rate
    print(f"Recording {a.minutes} min at {sr} Hz. Press Ctrl+C to stop early.")
    frames = []
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32") as stream:
            for _ in range(int(a.minutes * 60 * sr / 1600)):
                block, _ = stream.read(1600)
                frames.append(block[:, 0].copy())
    except KeyboardInterrupt:
        pass
    y = np.concatenate(frames) if frames else np.zeros(0, np.float32)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.name}.wav"
    wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    print(f"Saved {path} ({len(y) / sr:.1f} s). Label whistles in Audacity and export labels to "
          f"{path.with_suffix('.txt')}")


def cmd_train(a, s):
    from .train import train
    train(a.data, s, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, out=a.out)


def cmd_evaluate(a, s):
    from .evaluate import evaluate_dir, print_report, summarise
    results = evaluate_dir(a.data, s, a.detector)
    summary = summarise(results)
    print_report(results, summary)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=2)


def main(argv=None):
    p = argparse.ArgumentParser(prog="resona", description="Pressure cooker whistle detection")
    p.add_argument("--config", help="JSON settings file (see resona/config.py)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="Run the live dashboard")
    sp.add_argument("--source", default="mic", help="'mic' or path to a WAV file")
    sp.add_argument("--detector", default="dsp", choices=["dsp", "cnn"])
    sp.add_argument("--device", default=None, help="Microphone device index or name")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--loop", action="store_true", help="Loop the WAV file (demo mode)")
    sp.add_argument("--autostart", action="store_true", help="Start listening immediately")
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("synth", help="Generate synthetic test sessions")
    sp.add_argument("--out", default="data/synthetic")
    sp.add_argument("--sessions", type=int, default=20)
    sp.add_argument("--minutes", type=float, default=3.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_synth)

    sp = sub.add_parser("record", help="Record a kitchen session for the dataset")
    sp.add_argument("--out", default="data/real")
    sp.add_argument("--name", required=True)
    sp.add_argument("--minutes", type=float, default=15)
    sp.set_defaults(fn=cmd_record)

    sp = sub.add_parser("train", help="Train the CNN detector")
    sp.add_argument("--data", required=True)
    sp.add_argument("--epochs", type=int, default=25)
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("evaluate", help="Score a detector on labeled sessions")
    sp.add_argument("--data", required=True)
    sp.add_argument("--detector", default="dsp", choices=["dsp", "cnn"])
    sp.add_argument("--json", help="Also write the summary to this file")
    sp.set_defaults(fn=cmd_evaluate)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    a.fn(a, Settings.load(a.config))


if __name__ == "__main__":
    sys.exit(main())
