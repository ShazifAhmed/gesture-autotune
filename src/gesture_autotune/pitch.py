"""Pitch detection and musical-scale snapping.

Pure NumPy so it can be unit-tested without any audio hardware. The two public
entry points are:

    detect_f0(frame, sr)        -> fundamental frequency in Hz (or 0.0 if unvoiced)
    nearest_scale_freq(f0, key) -> the in-scale frequency to retune toward

Pitch detection uses normalized autocorrelation with parabolic interpolation of
the peak. It is intentionally lightweight: an autotune effect only needs a good
estimate of the fundamental on roughly every audio block, and autocorrelation is
cheap enough to run in the real-time callback.
"""

from __future__ import annotations

import numpy as np

# Semitone names within an octave, indexed by pitch class (0 == C).
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale formulas as semitone offsets from the tonic.
_SCALE_FORMULAS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],          # natural minor
    "chromatic": list(range(12)),
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
}

# A4 = 440 Hz is MIDI note 69; used to convert between Hz and MIDI numbers.
_A4_HZ = 440.0
_A4_MIDI = 69


def hz_to_midi(freq: float) -> float:
    """Convert a frequency in Hz to a (fractional) MIDI note number."""
    return _A4_MIDI + 12.0 * np.log2(freq / _A4_HZ)


def midi_to_hz(midi: float) -> float:
    """Convert a (fractional) MIDI note number to a frequency in Hz."""
    return _A4_HZ * (2.0 ** ((midi - _A4_MIDI) / 12.0))


def build_scale(key: str = "C", mode: str = "major") -> set[int]:
    """Return the set of pitch classes (0-11) belonging to a key/mode.

    Example: build_scale("A", "minor") -> {9, 11, 0, 2, 4, 5, 7}
    """
    mode = mode.lower()
    if mode not in _SCALE_FORMULAS:
        raise ValueError(f"Unknown mode {mode!r}; options: {list(_SCALE_FORMULAS)}")
    try:
        tonic = _NOTE_NAMES.index(key.upper())
    except ValueError as exc:
        raise ValueError(f"Unknown key {key!r}; options: {_NOTE_NAMES}") from exc
    return {(tonic + step) % 12 for step in _SCALE_FORMULAS[mode]}


def detect_f0(
    frame: np.ndarray,
    sr: int,
    fmin: float = 70.0,
    fmax: float = 1000.0,
    threshold: float = 0.30,
) -> float:
    """Estimate the fundamental frequency of a mono audio ``frame``.

    Returns 0.0 when the frame is too quiet or no clear periodicity is found
    (i.e. unvoiced), which the caller treats as "pass audio through untouched".

    The method is normalized autocorrelation:
      1. Remove DC and bail out on near-silence.
      2. Compute autocorrelation and normalize so lag 0 == 1.0.
      3. Search the lag range implied by [fmin, fmax] for the highest peak.
      4. Refine the peak location with parabolic interpolation for sub-sample
         (sub-Hz) accuracy.
    """
    x = np.asarray(frame, dtype=np.float64)
    if x.ndim != 1:
        x = x.reshape(-1)

    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x ** 2)) if x.size else 0.0
    if rms < 1e-4:  # effectively silence
        return 0.0

    # Full autocorrelation via FFT (only the non-negative lags are needed).
    n = x.size
    fft_size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, fft_size)
    acf = np.fft.irfft(spec * np.conj(spec), fft_size)[:n]

    if acf[0] <= 0:
        return 0.0
    acf = acf / acf[0]  # normalize: acf[0] == 1.0

    min_lag = max(1, int(sr / fmax))
    max_lag = min(n - 1, int(sr / fmin))
    if max_lag <= min_lag:
        return 0.0

    search = acf[min_lag:max_lag]
    peak = int(np.argmax(search)) + min_lag
    if acf[peak] < threshold:  # not periodic enough -> unvoiced
        return 0.0

    # Parabolic interpolation around the integer peak for a finer lag estimate.
    if 0 < peak < n - 1:
        a, b, c = acf[peak - 1], acf[peak], acf[peak + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        peak = peak + shift

    return float(sr / peak)


def nearest_scale_freq(f0: float, key: str = "C", mode: str = "major") -> float:
    """Snap ``f0`` to the nearest frequency allowed by ``key``/``mode``.

    Works across all octaves: the input is converted to a MIDI number, rounded to
    the nearest semitone that belongs to the scale, then converted back to Hz.
    Returns 0.0 for non-positive input so unvoiced frames stay untouched.
    """
    if f0 <= 0:
        return 0.0

    scale = build_scale(key, mode)
    midi = hz_to_midi(f0)
    nearest = round(midi)

    # Walk outward from the nearest semitone until we land on a scale tone.
    for offset in range(0, 7):
        for candidate in {nearest - offset, nearest + offset}:
            if candidate % 12 in scale:
                return midi_to_hz(candidate)
    return midi_to_hz(nearest)  # fallback (chromatic always hits offset 0)
