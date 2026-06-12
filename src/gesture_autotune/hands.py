"""Hand-openness tracking with MediaPipe, running on its own thread.

The webcam loop and the audio loop must not block each other, so camera capture
and hand inference run in a background thread and publish a single smoothed float
-- ``openness`` in [0, 1] -- that the audio callback samples whenever it wants.
Reading/writing a Python float is atomic under CPython, so no lock is needed on
the hot path.

"Openness" is computed geometrically and scale-invariantly: the mean distance
from the four finger tips to the wrist, divided by the palm length (wrist -> middle
-finger knuckle). A closed fist gives a small ratio; a spread hand a large one.
The raw ratio is mapped through a calibration range to [0, 1] and lightly smoothed.

This module imports cv2 and mediapipe lazily so the rest of the package (the DSP)
stays importable -- and unit-testable -- on machines without a camera.
"""

from __future__ import annotations

import threading
import time

import numpy as np

# Landmark indices (MediaPipe Hands).
_WRIST = 0
_MIDDLE_MCP = 9
_TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky finger tips


class HandTracker:
    """Background webcam hand tracker exposing a smoothed ``openness`` in [0, 1].

    Parameters
    ----------
    camera_index:
        OpenCV camera index (0 is usually the built-in webcam).
    closed_ratio, open_ratio:
        Calibration endpoints for the tip/palm distance ratio. Defaults work for
        most people; tune them with ``--calibrate`` in the app if needed.
    smoothing:
        One-pole smoothing factor (0..1); lower is smoother/slower.
    draw:
        If True, keep the annotated BGR frame in ``self.frame`` for display.
    """

    def __init__(
        self,
        camera_index: int = 0,
        closed_ratio: float = 1.2,
        open_ratio: float = 2.3,
        smoothing: float = 0.4,
        draw: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.closed_ratio = closed_ratio
        self.open_ratio = open_ratio
        self.smoothing = float(np.clip(smoothing, 0.01, 1.0))
        self.draw = draw

        self.openness = 0.0          # published value, read by the audio thread
        self.hand_present = False
        self.frame = None            # latest annotated frame (if draw=True)

        self._raw = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "HandTracker":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # -- geometry ----------------------------------------------------------
    @staticmethod
    def _openness_ratio(landmarks: np.ndarray) -> float:
        """tip->wrist mean distance divided by palm length (scale invariant)."""
        wrist = landmarks[_WRIST]
        palm = np.linalg.norm(landmarks[_MIDDLE_MCP] - wrist) + 1e-9
        tip_dist = np.mean([np.linalg.norm(landmarks[t] - wrist) for t in _TIPS])
        return float(tip_dist / palm)

    def _publish(self, ratio: float | None) -> None:
        if ratio is None:
            self.hand_present = False
            # Relax toward closed when the hand disappears.
            self._raw += self.smoothing * (0.0 - self._raw)
        else:
            self.hand_present = True
            norm = (ratio - self.closed_ratio) / (self.open_ratio - self.closed_ratio)
            norm = float(np.clip(norm, 0.0, 1.0))
            self._raw += self.smoothing * (norm - self._raw)
        self.openness = self._raw

    # -- main loop ---------------------------------------------------------
    def _loop(self) -> None:
        import cv2
        import mediapipe as mp

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.camera_index}. "
                "Grant camera permission and check the device index."
            )

        with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,           # fastest; plenty for openness
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                frame = cv2.flip(frame, 1)  # mirror for natural control
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    lm = result.multi_hand_landmarks[0]
                    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
                    self._publish(self._openness_ratio(pts))
                    if self.draw:
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                else:
                    self._publish(None)

                if self.draw:
                    self.frame = frame

        cap.release()
