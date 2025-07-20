#!/usr/bin/env python3
"""
split_on_silence.py
Splits input.wav at silent intervals into separate chunk_000.wav, chunk_001.wav, …
Usage:
    python split_on_silence.py input.wav out_dir/
"""

import os
import sys
from pydub import AudioSegment, silence


def main(input_wav, out_dir, min_silence_ms=500, silence_thresh=-40, keep_silence=200):
    os.makedirs(out_dir, exist_ok=True)
    audio = AudioSegment.from_wav(input_wav)

    # Split on silence
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
    )

    # Export chunks
    for i, chunk in enumerate(chunks):
        out_path = os.path.join(out_dir, f"chunk_{i:03d}.wav")
        chunk.export(out_path, format="wav")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_on_silence.py input.wav out_dir/")
        sys.exit(1)
    _, inp, outp = sys.argv
    main(inp, outp)
