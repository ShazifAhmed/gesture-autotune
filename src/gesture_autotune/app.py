"""Real-time app: control a from-scratch autotune with your hands.

Architecture (two loops, decoupled):

    [webcam thread]  HandTracker  --openness(float)-->  shared state
                                                              |
    [audio callback] mic block --> AutoTuner.process --> speakers
                       (reads openness, maps it to wet/dry + snap strength)

The audio callback must stay light and never block, so all it does is read one
float, set two parameters, and run the NumPy DSP. Hand inference (the expensive
part) lives on its own thread.

Gesture mapping (open hand == more effect, just like the reference video):
    amount              = openness          (wet/dry mix)
    correction_strength = 0.3 + 0.7*openness (snap gets harder as you open up)

Run:
    python -m gesture_autotune.app --key C --mode major
    python -m gesture_autotune.app --list-devices
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .autotune import AutoTuner


def list_devices() -> None:
    import sounddevice as sd
    print(sd.query_devices())


def run(args: argparse.Namespace) -> None:
    import sounddevice as sd
    from .hands import HandTracker

    sr = args.samplerate
    tuner = AutoTuner(sr=sr, key=args.key, mode=args.mode, grain_size=args.grain)

    # macOS shows the camera-permission dialog only from the MAIN thread, and
    # OpenCV can't request it from the tracker's background thread. So open the
    # camera here once to trigger/confirm permission, then let the thread skip
    # the auth path (it's already authorized at that point).
    import os, cv2, time
    probe = cv2.VideoCapture(args.camera)
    ok = False
    for _ in range(40):
        ok, _ = probe.read()
        if ok:
            break
        time.sleep(0.1)
    probe.release()
    if not ok:
        print("Camera not ready. If macOS just asked for permission, click Allow, "
              "then run the command again.")
        return
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

    tracker = HandTracker(
        camera_index=args.camera,
        closed_ratio=args.closed_ratio,
        open_ratio=args.open_ratio,
        draw=not args.no_window,
    ).start()

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0]

        openness = tracker.openness
        tuner.set_amount(openness)
        tuner.set_correction_strength(0.3 + 0.7 * openness)

        wet = tuner.process(mono)
        outdata[:, 0] = np.clip(wet, -1.0, 1.0)

    print("Opening audio stream… open your hand to dial in the autotune. Ctrl-C to stop.")
    stream = sd.Stream(
        samplerate=sr,
        blocksize=args.block,
        channels=1,
        dtype="float32",
        latency="low",
        callback=callback,
        device=(args.input_device, args.output_device),
    )

    try:
        with stream:
            _ui_loop(tracker, tuner, args)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        print("\nStopped.")


def _ui_loop(tracker, tuner, args) -> None:
    """Either show the annotated webcam window, or print a small text meter."""
    if args.no_window:
        while True:
            bar = "#" * int(tracker.openness * 20)
            note = f"{tuner.last_f0:6.1f} Hz" if tuner.last_f0 > 0 else "  --  "
            print(f"\ropenness [{bar:<20}] {tracker.openness:4.2f}  pitch {note}", end="")
            time.sleep(0.05)
        return

    import cv2
    while True:
        frame = tracker.frame
        if frame is not None:
            h = frame.shape[0]
            w_bar = int(tracker.openness * (frame.shape[1] - 40))
            cv2.rectangle(frame, (20, h - 40), (20 + w_bar, h - 20), (0, 220, 0), -1)
            cv2.putText(
                frame, f"effect {tracker.openness:4.2f}", (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.imshow("gesture-autotune  (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(0.005)
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hand-controlled autotune (real-time).")
    p.add_argument("--key", default="C", help="Musical key, e.g. C, A, F#")
    p.add_argument("--mode", default="major",
                   help="major | minor | chromatic | pentatonic_major | pentatonic_minor")
    p.add_argument("--samplerate", type=int, default=44100)
    p.add_argument("--block", type=int, default=512, help="Audio block size (latency vs. stability)")
    p.add_argument("--grain", type=int, default=2048, help="Pitch-shifter window size")
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--input-device", default=None, help="Input audio device (name or index)")
    p.add_argument("--output-device", default=None, help="Output audio device (name or index)")
    p.add_argument("--closed-ratio", type=float, default=1.2, help="Calibration: closed fist")
    p.add_argument("--open-ratio", type=float, default=2.3, help="Calibration: open hand")
    p.add_argument("--no-window", action="store_true", help="Headless: text meter instead of webcam window")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    # argparse turns --input-device into args.input_device etc.
    if getattr(args, "list_devices", False):
        list_devices()
        return
    # Allow numeric device indices passed as strings.
    for attr in ("input_device", "output_device"):
        val = getattr(args, attr)
        if val is not None and str(val).isdigit():
            setattr(args, attr, int(val))
    run(args)


if __name__ == "__main__":
    main()
