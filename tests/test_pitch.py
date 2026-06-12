"""Tests for pitch detection and scale snapping (no audio hardware needed)."""

import numpy as np
import pytest

from gesture_autotune.pitch import (
    build_scale,
    detect_f0,
    hz_to_midi,
    midi_to_hz,
    nearest_scale_freq,
)

SR = 44100


def sine(freq, dur=0.05, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


@pytest.mark.parametrize("freq", [110.0, 146.83, 220.0, 440.0, 660.0])
def test_detect_f0_recovers_pure_tone(freq):
    est = detect_f0(sine(freq), SR)
    # Within 1% of the true fundamental.
    assert abs(est - freq) / freq < 0.01


def test_detect_f0_returns_zero_on_silence():
    assert detect_f0(np.zeros(2048), SR) == 0.0


def test_detect_f0_returns_zero_on_noise():
    rng = np.random.default_rng(0)
    # White noise has no clear period -> should read as unvoiced.
    assert detect_f0(rng.standard_normal(2048) * 0.01, SR) == 0.0


def test_midi_hz_roundtrip():
    for midi in range(40, 80):
        assert abs(hz_to_midi(midi_to_hz(midi)) - midi) < 1e-9


def test_build_scale_c_major():
    # C major has no sharps/flats: C D E F G A B == pitch classes 0,2,4,5,7,9,11
    assert build_scale("C", "major") == {0, 2, 4, 5, 7, 9, 11}


def test_nearest_scale_freq_snaps_to_a440():
    # 445 Hz is a sharp A4; in C major it should snap to A4 == 440.
    snapped = nearest_scale_freq(445.0, "C", "major")
    assert abs(snapped - 440.0) < 0.5


def test_nearest_scale_freq_skips_out_of_scale_note():
    # ~466 Hz (A#4) is not in C major; nearest scale tones are A4 (440) and
    # B4 (~493.9). It must land on one of them, never on A#4 itself.
    snapped = nearest_scale_freq(466.0, "C", "major")
    assert abs(snapped - 440.0) < 1.0 or abs(snapped - 493.88) < 1.0


def test_nearest_scale_freq_zero_passthrough():
    assert nearest_scale_freq(0.0) == 0.0
