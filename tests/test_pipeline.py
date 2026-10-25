import numpy as np
import pytest

from resona.config import Settings
from resona.evaluate import match_events, run_detector
from resona.synth import make_session


def _score(seed):
    s = Settings()
    y, labels = make_session(seed=seed, minutes=1.5, n_whistles=3)
    _, preds = run_detector(y, s, "dsp")
    return match_events([(a, b) for a, b, _ in labels], preds)


@pytest.mark.parametrize("seed", [1, 2, 3, 7])
def test_dsp_counts_synthetic_session(seed):
    tp, fp, fn, _ = _score(seed)
    assert (tp, fp, fn) == (3, 0, 0)


@pytest.mark.xfail(reason="Known baseline limitation: whistle masked by a loud mixer grinder. "
                          "The CNN detector is expected to fix this case.", strict=True)
def test_dsp_whistle_under_mixer():
    tp, fp, fn, _ = _score(42)
    assert fn == 0


def test_silence_produces_no_whistles():
    s = Settings()
    _, preds = run_detector(np.zeros(16000 * 10, dtype=np.float32), s, "dsp")
    assert preds == []
