"""Streaming pitch shifter -- two-tap crossfading delay line (Doppler method).

For a real-time, gesture-driven effect we need a shifter that:
  * processes audio in small blocks with bounded, constant latency,
  * accepts a pitch ratio that changes continuously (the hand is always moving),
  * stays cheap enough to run inside the audio callback,
  * is *exact* in the frequency domain so an autotune correction lands on pitch.

The classic technique that satisfies all of these is a modulated delay line.
If we read a signal through a delay ``d`` that changes linearly in time, the
output is Doppler-shifted: ``y(t) = x(t - d(t))`` has instantaneous frequency
``f * (1 - d'(t))``. Choosing ``d'(t) = 1 - ratio`` gives an output frequency of
exactly ``f * ratio`` with the duration preserved.

A single ramped delay would hit the buffer edge and click on reset, so we run
*two* delay taps a half-cycle out of phase and crossfade them with a raised-cosine
window whose two copies sum to 1. Each tap is faded to zero exactly when it wraps,
so the reset is inaudible. At ``ratio == 1`` the modulation stops and the output
is a clean, unity-gain delayed copy of the input.

Pure NumPy and hardware-free, so the unit tests drive it directly.
"""

from __future__ import annotations

import numpy as np


class GranularPitchShifter:
    """Real-time pitch shifter operating on a continuous mono stream.

    ``process(block)`` returns the same number of samples it is given, with a
    constant internal latency of about ``grain_size / 2`` samples.

    Parameters
    ----------
    sr:
        Sample rate in Hz.
    grain_size:
        Delay-sweep window in samples; also the maximum delay. ~46 ms
        (2048 @ 44.1k) is a good default for voice.
    overlap:
        Accepted for API compatibility; the two-tap crossfade is fixed at 50%.
    """

    def __init__(self, sr: int, grain_size: int = 2048, overlap: float = 0.5) -> None:
        self.sr = sr
        self.grain_size = int(grain_size)
        self.window = self.grain_size  # max delay / sweep length, in samples

        # Power-of-two delay line large enough to hold the full sweep plus a block.
        self._buf_len = 1
        while self._buf_len < self.grain_size * 4:
            self._buf_len <<= 1
        self._dline = np.zeros(self._buf_len, dtype=np.float64)
        self._w = 0           # absolute write index
        self._phase = 0.0     # delay-sweep phase in [0, 1)
        self._ratio = 1.0

    def set_ratio(self, ratio: float) -> None:
        """Set the pitch ratio (2.0 == one octave up, 0.5 == one octave down)."""
        self._ratio = float(np.clip(ratio, 0.25, 4.0))

    def _read_frac(self, positions: np.ndarray) -> np.ndarray:
        """Linearly interpolate the delay line at fractional absolute ``positions``."""
        i0 = np.floor(positions).astype(np.int64)
        frac = positions - i0
        a = self._dline[i0 % self._buf_len]
        b = self._dline[(i0 + 1) % self._buf_len]
        return a * (1.0 - frac) + b * frac

    def process(self, block: np.ndarray) -> np.ndarray:
        """Push one input ``block``; return the same number of pitch-shifted samples."""
        block = np.asarray(block, dtype=np.float64).reshape(-1)
        n = block.size
        if n == 0:
            return block

        # 1) Write the block into the delay line (vectorized, wrap-aware).
        start = self._w % self._buf_len
        end = start + n
        if end <= self._buf_len:
            self._dline[start:end] = block
        else:
            first = self._buf_len - start
            self._dline[start:] = block[:first]
            self._dline[: n - first] = block[first:]
        write_idx = self._w + np.arange(n)  # absolute index of each sample
        self._w += n

        # 2) Sweep phase advances linearly within the block (ratio constant here).
        #    d'(n) = (1 - ratio) samples per sample, swept over a window of length W,
        #    so phase advances by (1 - ratio) / W each sample, wrapped to [0, 1).
        slope = (1.0 - self._ratio) / self.window
        phase = (self._phase + np.arange(n) * slope) % 1.0
        self._phase = float((self._phase + n * slope) % 1.0)

        # 3) Two delay taps a half-cycle apart, crossfaded so their gains sum to 1.
        phase_b = (phase + 0.5) % 1.0
        delay_a = phase * self.window
        delay_b = phase_b * self.window
        gain_a = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
        gain_b = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase_b)

        tap_a = self._read_frac(write_idx - delay_a)
        tap_b = self._read_frac(write_idx - delay_b)
        return gain_a * tap_a + gain_b * tap_b
