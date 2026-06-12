"""Offline demo: hear the autotune without a microphone.

Synthesizes a deliberately out-of-tune "vocal" (a few harmonics with slow pitch
drift and vibrato), runs it through the AutoTuner, and writes before/after WAVs.
It also sweeps the gesture ``amount`` from 0 to 1 so you can hear the effect fade
in exactly like opening a hand.

    python scripts/demo_offline.py --out demo_out

Produces:
    demo_out/dry.wav     -- the raw, detuned vocal
    demo_out/tuned.wav   -- fully autotuned (amount = 1)
    demo_out/sweep.wav   -- amount swept 0 -> 1 over the clip (the "hand opening")
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gesture_autotune import AutoTuner  # noqa: E402

SR = 44100


def synth_detuned_vocal(dur=4.0, sr=SR):
    """A glide between a few notes, each pushed sharp/flat, with light vibrato."""
    t = np.arange(int(dur * sr)) / sr
    # Target a little off the C-major scale so there's something to correct.
    base_notes = [262 * 1.03, 294 * 0.97, 330 * 1.04, 392 * 0.96]  # C D E G, detuned
    seg = len(t) // len(base_notes)
    f0 = np.zeros_like(t)
    for i, f in enumerate(base_notes):
        f0[i * seg:(i + 1) * seg] = f
    f0[len(base_notes) * seg:] = base_notes[-1]
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t)  # 5 Hz, ~1%
    f0 = f0 * vibrato

    phase = 2 * np.pi * np.cumsum(f0) / sr
    # A few harmonics so it reads as "voice", not a sine.
    sig = (np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase))
    env = np.minimum(1.0, np.minimum(t / 0.05, (dur - t) / 0.05))  # fade in/out
    return (sig * env * 0.3).astype(np.float64)


def run(tuner_factory, signal, amount_fn, block=1024):
    tuner = tuner_factory()
    out = []
    n = len(signal)
    for i in range(0, n - block, block):
        tuner.set_amount(amount_fn(i / n))
        out.append(tuner.process(signal[i:i + block]))
    return np.concatenate(out)


def to_wav(path, x):
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(path, SR, (x * 32767).astype(np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--key", default="C")
    ap.add_argument("--mode", default="major")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dry = synth_detuned_vocal()

    def factory():
        return AutoTuner(sr=SR, key=args.key, mode=args.mode)

    tuned = run(factory, dry, amount_fn=lambda p: 1.0)
    sweep = run(factory, dry, amount_fn=lambda p: p)  # hand opening over time

    to_wav(os.path.join(args.out, "dry.wav"), dry)
    to_wav(os.path.join(args.out, "tuned.wav"), tuned)
    to_wav(os.path.join(args.out, "sweep.wav"), sweep)
    print(f"Wrote dry.wav, tuned.wav, sweep.wav to {args.out}/")


if __name__ == "__main__":
    main()
