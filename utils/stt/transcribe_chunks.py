#!/usr/bin/env python3
"""
transcribe_chunks.py

Iterate over chunk*.wav files in a directory, run Polish ASR locally
using OpenAI Whisper, and write out a CSV for downstream TTS fine-tuning.

Usage:
    python transcribe_chunks.py \
      --input_dir data/chunks \
      --output_csv data/metadata.csv \
      --model_size small \
      --device cuda
"""

import os
import argparse
import glob

import pandas as pd
import soundfile as sf
import whisper
from tqdm import tqdm


def transcribe_folder(input_dir: str,
                      output_csv: str,
                      model_size: str = "small",
                      device: str = "cuda"):
    # 1) Load Whisper model once
    print(f"Loading Whisper model '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)

    # 2) Gather chunk files
    pattern = os.path.join(input_dir, "chunk*.wav")
    wav_paths = sorted(glob.glob(pattern))
    if not wav_paths:
        raise RuntimeError(f"No files match '{pattern}'")

    # 3) Transcribe each file
    records = []
    for wav_path in tqdm(wav_paths, desc="Transcribing"):
        fname = os.path.basename(wav_path)

        # read duration
        info = sf.info(wav_path)
        duration = info.frames / info.samplerate

        # ASR call (Polish)
        result = model.transcribe(wav_path, language="pl")
        text = result["text"].strip()

        records.append({
            "wav_filename": fname,
            "transcript": text,
            "duration": round(duration, 3)
        })

    # 4) Build DataFrame and save CSV
    df = pd.DataFrame.from_records(records,
                                   columns=["wav_filename", "transcript", "duration"])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n✓ Saved metadata for {len(df)} files to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-transcribe chunk*.wav files (Polish) → metadata.csv")
    parser.add_argument("--input_dir",
                        required=True,
                        help="Folder containing chunk_*.wav files")
    parser.add_argument("--output_csv",
                        required=True,
                        help="Path to write metadata CSV")
    parser.add_argument("--model_size",
                        default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--device",
                        default="cuda",
                        help="torch device: cuda or cpu")
    args = parser.parse_args()

    transcribe_folder(args.input_dir, args.output_csv,
                      model_size=args.model_size,
                      device=args.device)
