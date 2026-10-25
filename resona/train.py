"""Train WhistleNet on labeled sessions (WAV + Audacity label file).

Design choices worth defending in a viva:
  * Split by SESSION, not by window. Overlapping windows from the same
    recording in both train and validation would leak and inflate scores.
  * Ambiguous windows (10-50 % whistle) are dropped rather than forced
    into a class, which gives cleaner decision boundaries.
  * Noise mixing is done in the linear power domain (log-sum-exp of dB
    values), which is physically correct for adding two sounds.
  * Class imbalance (whistles are rare) is handled with pos_weight.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .audio import load_wav
from .config import Settings
from .features import logmel_spectrogram
from .model import WhistleNet
from .synth import read_labels


def load_sessions(data_dir: str | Path, settings: Settings):
    sessions = []
    rate = settings.audio.frame_rate
    for wav in sorted(Path(data_dir).glob("*.wav")):
        L = logmel_spectrogram(load_wav(wav, settings.audio.sample_rate), settings.audio)
        frame_label = np.zeros(len(L), dtype=np.float32)
        for s, e in read_labels(wav.with_suffix(".txt")):
            frame_label[int(s * rate):int(e * rate) + 1] = 1.0
        sessions.append((wav.stem, L, frame_label))
    if not sessions:
        raise SystemExit(f"No .wav files found in {data_dir}")
    return sessions


def make_windows(sessions, window: int, hop: int):
    X, y = [], []
    for _, L, lab in sessions:
        for i in range(0, len(L) - window + 1, hop):
            frac = lab[i:i + window].mean()
            if frac >= 0.5:
                X.append(L[i:i + window]); y.append(1.0)
            elif frac <= 0.1:
                X.append(L[i:i + window]); y.append(0.0)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)


class WindowDataset(Dataset):
    def __init__(self, X, y, mean, std, augment: bool):
        self.X, self.y, self.mean, self.std, self.augment = X, y, mean, std, augment
        self.neg_idx = np.where(y == 0)[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            x += np.random.uniform(-6, 6)                         # overall loudness
            if np.random.rand() < 0.5 and len(self.neg_idx):      # mix in kitchen noise
                noise = self.X[np.random.choice(self.neg_idx)] + np.random.uniform(-12, 3)
                x = 10 * np.log10(10 ** (x / 10) + 10 ** (noise / 10))
            for _ in range(2):                                    # SpecAugment masks
                f = np.random.randint(0, 8); f0 = np.random.randint(0, x.shape[1] - f)
                x[:, f0:f0 + f] = x.mean()
                t = np.random.randint(0, 8); t0 = np.random.randint(0, x.shape[0] - t)
                x[t0:t0 + t, :] = x.mean()
        x = (x - self.mean) / self.std
        return torch.from_numpy(x[None].astype(np.float32)), torch.tensor([float(self.y[i])], dtype=torch.float32)


def _f1(logits, targets):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    tp = float((pred * targets).sum()); fp = float((pred * (1 - targets)).sum())
    fn = float(((1 - pred) * targets).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)


def train(data_dir: str, settings: Settings, epochs: int = 25, batch_size: int = 64,
          lr: float = 1e-3, val_fraction: float = 0.2, seed: int = 0, out: str | None = None) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sessions = load_sessions(data_dir, settings)
    random.shuffle(sessions)
    n_val = max(1, int(round(len(sessions) * val_fraction))) if len(sessions) > 1 else 0
    val_s, train_s = sessions[:n_val], sessions[n_val:]
    print(f"Sessions: {len(train_s)} train / {len(val_s)} validation")

    W = settings.model.window_frames
    Xtr, ytr = make_windows(train_s, W, hop=W // 4)
    mean, std = float(Xtr.mean()), float(Xtr.std() + 1e-6)
    train_dl = DataLoader(WindowDataset(Xtr, ytr, mean, std, True), batch_size, shuffle=True)
    val_dl = None
    if val_s:
        Xva, yva = make_windows(val_s, W, hop=W // 2)
        val_dl = DataLoader(WindowDataset(Xva, yva, mean, std, False), batch_size)
    pos = float(ytr.sum())
    print(f"Train windows: {len(ytr)} ({int(pos)} whistle, {len(ytr) - int(pos)} background)")

    model = WhistleNet(settings.audio.n_mels)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(ytr) - pos) / max(pos, 1.0)]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out_path = Path(out or settings.model.weights_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
        sched.step()
        train_loss = total / len(train_dl.dataset)

        val_f1 = None
        if val_dl:
            model.eval()
            with torch.no_grad():
                logits = torch.cat([model(xb) for xb, _ in val_dl])
                targets = torch.cat([yb for _, yb in val_dl])
            val_f1 = _f1(logits, targets)
        score = val_f1 if val_f1 is not None else -train_loss
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_f1": val_f1})
        print(f"epoch {epoch:>3}  loss {train_loss:.4f}  val F1 {val_f1 if val_f1 is None else round(val_f1, 4)}")
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "settings": settings.to_dict(), "epoch": epoch,
                        "val_sessions": [s[0] for s in val_s]}, out_path)

    (out_path.with_suffix(".history.json")).write_text(json.dumps(history, indent=2))
    print(f"Saved best model to {out_path}")
    return {"best": best, "history": history, "path": str(out_path)}
