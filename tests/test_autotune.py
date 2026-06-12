"""End-to-end DSP test: a detuned tone should be pulled toward the scale."""

import numpy as np

from gesture_autotune.autotune import AutoTuner

SR = 44100


def dominant_freq(x, sr=SR):
    x = x - np.mean(x)
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, 1 / sr)
    return freqs[np.argmax(spec)]


def make_sine(freq, dur=2.5, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


def stream(tuner, signal, block=1024):
    out = []
    for i in range(0, len(signal) - block, block):
        out.append(tuner.process(signal[i:i + block]))
    return np.concatenate(out)


def test_autotune_pulls_sharp_note_to_scale():
    # 452 Hz is a sharp A4. With full correction in C major it should move
    # measurably toward A4 (440), i.e. closer to 440 than it started.
    tuner = AutoTuner(sr=SR, key="C", mode="major")
    tuner.set_amount(1.0)
    tuner.set_correction_strength(1.0)

    out = stream(tuner, make_sine(452.0))
    f_out = dominant_freq(out[-SR:])

    start_err = abs(452.0 - 440.0)
    end_err = abs(f_out - 440.0)
    assert end_err < start_err  # pitch was corrected toward the scale tone


def test_autotune_amount_zero_is_passthrough():
    # amount == 0 -> dry voice only -> frequency unchanged.
    tuner = AutoTuner(sr=SR, key="C", mode="major")
    tuner.set_amount(0.0)
    out = stream(tuner, make_sine(452.0))
    f_out = dominant_freq(out[-SR:])
    assert abs(f_out - 452.0) / 452.0 < 0.02


def test_autotune_handles_silence_without_error():
    tuner = AutoTuner(sr=SR)
    out = stream(tuner, np.zeros(SR))
    assert np.all(np.isfinite(out))
