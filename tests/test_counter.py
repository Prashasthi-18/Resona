from resona.counter import WhistleCounter


def feed(c, values, rate=50):
    out = []
    for i, p in enumerate(values):
        out += c.update(p, i / rate)
    out += c.flush(len(values) / rate)
    return out


def test_counts_one_clean_whistle():
    c = WhistleCounter()
    feed(c, [0.0] * 50 + [0.9] * 150 + [0.0] * 100)
    assert c.count == 1
    assert abs(c.events[0].duration - 3.0) < 0.05


def test_rejects_short_blip():
    c = WhistleCounter(min_duration=0.6)
    feed(c, [0.0] * 50 + [0.95] * 15 + [0.0] * 50)   # 0.3 s clang
    assert c.count == 0


def test_merges_brief_dropout():
    c = WhistleCounter(merge_gap=0.8)
    feed(c, [0.9] * 100 + [0.1] * 20 + [0.9] * 100 + [0.0] * 100)  # 0.4 s dip
    assert c.count == 1


def test_splits_separate_whistles():
    c = WhistleCounter(merge_gap=0.8)
    feed(c, [0.9] * 100 + [0.0] * 150 + [0.9] * 100 + [0.0] * 100)  # 3 s apart
    assert c.count == 2


def test_hysteresis_prevents_flicker():
    c = WhistleCounter(on_threshold=0.6, off_threshold=0.4)
    wobble = [0.9, 0.5] * 100   # hovers between thresholds after starting
    feed(c, wobble + [0.0] * 100)
    assert c.count == 1
