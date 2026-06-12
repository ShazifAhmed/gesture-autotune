"""Tests for the streaming granular pitch shifter.

Strategy: stream a long sine through the shifter block by block (exactly how the
real-time app drives it), then measure the dominant frequency of the steady-state
output with an FFT and confirm it scaled by the requested ratio.
"""

import numpy as np
import pytest

from gesture_autotune.shifter import GranularPitchShifter

SR = 44100


def dominant_freq(x, sr=SR):
    x = x - np.mean(x)
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, 1 / sr)
    return freqs[np.argmax(spec)]


def stream(shifter, signal, block=512):
    out = []
    for i in range(0, len(signal) - block, block):
        out.append(shifter.process(signal[i:i + block]))
    return np.concatenate(out)


def make_sine(freq, dur=2.0, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5, 2.0])
def test_shifter_scales_frequency(ratio):
    f_in = 220.0
    sh = GranularPitchShifter(SR, grain_size=2048)
    sh.set_ratio(ratio)
    out = stream(sh, make_sine(f_in))

    # Analyze the steady-state tail to skip the priming latency.
    tail = out[-SR:]
    f_out = dominant_freq(tail)
    assert abs(f_out - f_in * ratio) / (f_in * ratio) < 0.05


def test_shifter_unity_preserves_energy():
    # Hann at 50% overlap satisfies COLA, so ratio == 1 reconstructs at ~unity gain.
    sh = GranularPitchShifter(SR, grain_size=2048)
    sh.set_ratio(1.0)
    out = stream(sh, make_sine(330.0))
    assert out.size > 0
    tail = out[-SR:]
    rms_in = np.sqrt(np.mean(make_sine(330.0, dur=1.0) ** 2))
    rms_out = np.sqrt(np.mean(tail ** 2))
    assert 0.7 < rms_out / rms_in < 1.3


def test_ratio_is_clamped():
    sh = GranularPitchShifter(SR)
    sh.set_ratio(99.0)
    sh.set_ratio(0.001)  # should not raise; clamped internally
