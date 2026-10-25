# Resona

Real-time acoustic event detection for pressure cooker whistles.

In Indian kitchens, cooking time is measured in whistles, not minutes. Resona
listens through a microphone, detects each whistle, counts them, and alerts you
when the target is reached, so nobody has to stand by the stove counting.

## What's inside

```
resona/
  config.py          every tunable number in one place
  features.py        streaming STFT + log-mel features (shared by training and live use)
  detectors/dsp.py   baseline: band-energy ratio + adaptive noise floor
  detectors/cnn.py   neural detector: WhistleNet on a sliding 1 s window
  model.py           WhistleNet CNN (~95k parameters)
  counter.py         hysteresis state machine that turns probabilities into counted whistles
  engine.py          real-time pipeline on a background thread
  server.py          FastAPI + WebSocket server
  train.py           CNN training (session-level split, noise mixing, SpecAugment)
  evaluate.py        event-level precision / recall / F1, false alarms per hour, latency
  synth.py           synthetic kitchen sessions for pipeline testing
  notify.py          phone alerts via Telegram and ntfy, with retries
web/index.html       animated dashboard (also runs a built-in demo if no server is found)
tests/               unit tests for the counter and the pipeline
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m resona serve             # listen to your microphone
# open http://127.0.0.1:8000 and press "Start listening"
```

No cooker nearby? Generate a synthetic session and play it through the full pipeline:

```bash
python -m resona synth --out data/synthetic --sessions 20
python -m resona serve --source data/synthetic/synthetic_000.wav --loop
```

Opening `web/index.html` directly in a browser (or adding `?demo` to the URL)
runs a simulated kitchen, which is handy for UI work and presentations.

## Building the real dataset

1. Record full cooking sessions, not just whistles:
   `python -m resona record --name kitchen01_hawkins3L --minutes 20`
2. Open the WAV in Audacity, select each whistle, press Ctrl+B to label it
   `whistle`, then File → Export → Export Labels to a `.txt` with the same name.
3. Vary everything: cooker brands and sizes, rooms, mic distance, phone vs. laptop
   mic, and plenty of hard negatives (mixer grinder, exhaust fan, tadka, TV,
   kettle, pressure-release hiss between whistles).
4. Keep some kitchens entirely out of training for the final test set.

## Train and evaluate

```bash
python -m resona train --data data/real --epochs 25
python -m resona evaluate --data data/test --detector dsp --json results_dsp.json
python -m resona evaluate --data data/test --detector cnn --json results_cnn.json
python -m resona serve --detector cnn
```

Synthetic audio only proves the plumbing works. Report results on real recordings
from kitchens the model never saw during training.

## Tuning

Look at spectrograms of your own recordings and set `band_low` / `band_high` in
`resona/config.py` (or a JSON file passed with `--config`) to where whistle
energy actually sits. The counter's `min_duration` and `merge_gap` control how
short sounds are rejected and how sputtering whistles are merged.

## Tests

```bash
pytest -q
```

`test_dsp_whistle_under_mixer` is an expected failure: the rule-based baseline
misses a whistle buried under a loud mixer grinder. That case is the motivation
for the neural detector.

## Phone alerts (Telegram and the ntfy app)

Resona can message your phone so you can leave the kitchen. Set it up from the
dashboard: click **Set up phone alerts**. You can use either channel or both.

**Telegram**
1. In Telegram, open @BotFather, send `/newbot`, and copy the token it gives you.
2. Paste the token into Resona and click **Connect bot**.
3. Open your new bot (link and QR code shown) and press **Start**. Resona detects
   your chat automatically and sends a confirmation.

**ntfy app** (free, Android and iPhone, no account)
1. Install ntfy from the App Store or Google Play.
2. Click **Create my topic** in Resona, then subscribe to that topic in the app
   (tap +, or scan the QR code).

What you'll receive:

| When | Message | Priority |
|---|---|---|
| Each whistle before the target (can be turned off) | "Whistle 1 of 3" | normal |
| Target reached | "3 whistles reached. Time to turn off the stove." | urgent |
| Whistles continue past the target | "Whistle 4, past your target" | urgent |
| Audio engine stops unexpectedly | "Resona needs attention" | urgent |

Setup is saved to `notify.json` (git-ignored, because it holds your bot token).
For headless setups you can use environment variables instead:
`RESONA_TELEGRAM_TOKEN`, `RESONA_TELEGRAM_CHAT`, `RESONA_NTFY_TOPIC`,
`RESONA_NTFY_SERVER` (for a self-hosted ntfy server).

Alerts are sent on a background thread with automatic retries, so a slow
network never delays whistle detection. The computer running Resona needs an
internet connection; your phone can be anywhere.
