"""AutoTuner -- ties pitch detection, scale snapping and the pitch shifter
together into a single block processor, with a gesture-controlled amount.

The signal chain per block is:

    detect F0  ->  snap to nearest scale note  ->  target/F0 = correction ratio
                                                       |
    smooth ratio (avoid zipper noise)  <---------------+
                                                       |
    apply correction_strength (0..1)  <----------------+   (how hard it snaps)
                                                       v
    GranularPitchShifter.process()  ->  wet signal
                                                       |
    wet/dry mix by ``amount`` (0..1)  <----------------+   (hand openness)

Two knobs are exposed to the gesture layer:

* ``amount``             -- wet/dry mix. 0 = original voice, 1 = full effect.
* ``correction_strength``-- 0 = leave pitch alone, 1 = hard snap to scale.

In the app both are driven by how open the hand is, which is exactly the
"control the vocoder with my hands" behavior from the reference video.
"""

from __future__ import annotations

import numpy as np

from .pitch import detect_f0, nearest_scale_freq
from .shifter import GranularPitchShifter


class AutoTuner:
    def __init__(
        self,
        sr: int = 44100,
        key: str = "C",
        mode: str = "major",
        grain_size: int = 2048,
        ratio_smoothing: float = 0.25,
    ) -> None:
        self.sr = sr
        self.key = key
        self.mode = mode
        self.shifter = GranularPitchShifter(sr, grain_size=grain_size)

        # One-pole smoothing factor for the correction ratio. Lower == smoother
        # (more glide), higher == snappier. Keeps the pitch from "zippering" as
        # F0 estimates jitter block to block.
        self.ratio_smoothing = float(np.clip(ratio_smoothing, 0.01, 1.0))

        self._smoothed_ratio = 1.0
        self.amount = 1.0               # wet/dry mix (gesture-controlled)
        self.correction_strength = 1.0  # snap hardness (gesture-controlled)
        self.last_f0 = 0.0              # exposed for UI/telemetry

    def set_amount(self, amount: float) -> None:
        self.amount = float(np.clip(amount, 0.0, 1.0))

    def set_correction_strength(self, strength: float) -> None:
        self.correction_strength = float(np.clip(strength, 0.0, 1.0))

    def process(self, block: np.ndarray) -> np.ndarray:
        """Autotune one mono audio ``block`` and return the processed block."""
        dry = np.asarray(block, dtype=np.float64).reshape(-1)

        f0 = detect_f0(dry, self.sr)
        self.last_f0 = f0

        if f0 > 0:
            target = nearest_scale_freq(f0, self.key, self.mode)
            raw_ratio = target / f0 if target > 0 else 1.0
        else:
            raw_ratio = 1.0  # unvoiced -> no shift

        # Blend toward the corrected ratio by the correction strength: 0 leaves
        # pitch untouched, 1 fully snaps to the scale tone.
        target_ratio = 1.0 + self.correction_strength * (raw_ratio - 1.0)

        # One-pole smoothing to avoid audible jumps as F0 jitters.
        self._smoothed_ratio += self.ratio_smoothing * (target_ratio - self._smoothed_ratio)
        self.shifter.set_ratio(self._smoothed_ratio)

        wet = self.shifter.process(dry)

        # Equal-length wet/dry mix. The shifter delays the wet path by its
        # constant latency; for a real-time effect that small offset is
        # inaudible, and the dry path is only fully heard at amount == 0.
        return (1.0 - self.amount) * dry + self.amount * wet
