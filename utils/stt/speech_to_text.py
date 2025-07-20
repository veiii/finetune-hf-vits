#!/usr/bin/env python3
"""
async_whisper_transcribe.py

Batch‐transcribe WAV chunks with Whisper on GPU using asyncio.
– Appends each transcript immediately to CSV so you can kill/restart without loss.
– Skips already‐transcribed files.
– Guards Whisper calls with a lock to avoid internal kv‐cache shape errors.
"""

import os
import glob
import csv
import argparse
import asyncio

import pandas as pd
import soundfile as sf
import whisper
from tqdm import tqdm


async def write_record(output_csv: str, record: dict, lock: asyncio.Lock):
    """
    Append a single record to CSV under an async lock.
    Creates file + header if needed.
    """
    header_needed = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    async with lock:
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["wav_filename", "transcript", "duration"])
            if header_needed:
                writer.writeheader()
            writer.writerow(record)


def _do_transcribe(model, path, language):
    """
    Blocking whisper.transcribe call.
    """
    return model.transcribe(path, language=language)["text"].strip()


async def transcribe_file(path: str,
                          model,
                          sem: asyncio.Semaphore,
                          model_lock: asyncio.Lock,
                          language: str) -> dict:
    """
    Acquire `sem` to limit concurrency, then acquire `model_lock` to serialize
    actual whisper calls. Returns a record dict or raises.
    """
    async with sem:
        loop = asyncio.get_event_loop()
        # ensure only one thread invokes whisper at a time:
        async with model_lock:
            text = await loop.run_in_executor(None, _do_transcribe, model, path, language)

        info = sf.info(path)
        duration = round(info.frames / info.samplerate, 3)
        return {
            "wav_filename": os.path.basename(path),
            "transcript": text,
            "duration": duration
        }


async def main(args):
    # 1) load existing CSV to skip duplicates
    processed = set()
    if os.path.exists(args.output_csv):
        try:
            df_prev = pd.read_csv(args.output_csv, dtype=str)
            processed = set(df_prev["wav_filename"].tolist())
        except Exception:
            processed = set()

    # 2) find chunk files
    pattern = os.path.join(args.input_dir, "chunk*.wav")
    all_files = sorted(glob.glob(pattern))
    to_process = [f for f in all_files if os.path.basename(f) not in processed]

    if not to_process:
        print(f"Nothing to do: all {len(all_files)} files already done.")
        return

    # 3) load model
    print(f"Loading Whisper model '{args.model_size}' on {args.device}...")
    model = whisper.load_model(args.model_size, device=args.device)

    # 4) prepare concurrency controls
    sem = asyncio.Semaphore(args.concurrency)
    model_lock = asyncio.Lock()   # serialize model calls
    csv_lock = asyncio.Lock()     # serialize CSV writes

    # 5) schedule transcription tasks
    tasks = [
        asyncio.create_task(
            transcribe_file(path, model, sem, model_lock, args.language)
        )
        for path in to_process
    ]

    # 6) consume as they complete, write each immediately
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Transcribing")
    for fut in pbar:
        try:
            rec = await fut
        except Exception as e:
            fname = getattr(e, 'filename', 'unknown')
            print(f"\n⚠️  Error on {fname}: {e}")
            continue

        await write_record(args.output_csv, rec, csv_lock)
        pbar.set_postfix(file=rec["wav_filename"])

    print(f"\n✓ Appended {len(to_process)} records to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async Whisper Transcription")
    parser.add_argument("--input_dir",   required=True, help="folder with chunk*.wav")
    parser.add_argument("--output_csv",  required=True, help="path to metadata.csv")
    parser.add_argument("--model_size",  default="small",
                        choices=["tiny","base","small","medium","large"],
                        help="Whisper model size")
    parser.add_argument("--device",      default="cuda", help="cuda or cpu")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="max concurrent whisper calls")
    parser.add_argument("--language",    default="pl", help="language code for Whisper")
    args = parser.parse_args()

    asyncio.run(main(args))
