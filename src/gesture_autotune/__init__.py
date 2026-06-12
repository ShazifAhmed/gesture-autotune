"""gesture_autotune -- control a from-scratch autotune effect with your hands.

Public API:
    AutoTuner            -- block-based autotune processor (pure NumPy DSP)
    GranularPitchShifter -- streaming pitch shifter used by AutoTuner
    detect_f0            -- autocorrelation pitch detector
    nearest_scale_freq   -- snap a frequency to a musical scale
"""

from .autotune import AutoTuner
from .pitch import build_scale, detect_f0, hz_to_midi, midi_to_hz, nearest_scale_freq
from .shifter import GranularPitchShifter

__all__ = [
    "AutoTuner",
    "GranularPitchShifter",
    "detect_f0",
    "nearest_scale_freq",
    "build_scale",
    "hz_to_midi",
    "midi_to_hz",
]

__version__ = "0.1.0"
