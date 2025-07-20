#!/usr/bin/env python3
"""
transcribe_chunks.py

Iterate over chunk*.wav files in a directory, run Polish ASR locally
using OpenAI Whisper, and write out a CSV for downstream TTS fine-tuning.

Usage:
    python transcribe_chunks.py \
      --input-dir data/chunks \
      --output-csv data/metadata.csv \
      --model-size small \
      --device cuda \
      --max-workers 4 \
      --batch-size 10
"""

import os
import argparse
import glob
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

import pandas as pd
import soundfile as sf
import whisper
from tqdm import tqdm


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('transcribe_chunks.log')
        ]
    )
    return logging.getLogger(__name__)


def validate_audio_file(wav_path: str, logger: logging.Logger) -> Optional[Tuple[str, float]]:
    """Validate audio file and return filename and duration."""
    try:
        info = sf.info(wav_path)
        duration = info.frames / info.samplerate
        if duration <= 0:
            logger.warning(f"Invalid duration for {wav_path}: {duration}s")
            return None
        return os.path.basename(wav_path), duration
    except Exception as e:
        logger.error(f"Failed to read audio file {wav_path}: {e}")
        return None


def transcribe_single_file(args: Tuple[str, whisper.Whisper, logging.Logger]) -> Optional[Dict]:
    """Transcribe a single audio file."""
    wav_path, model, logger = args

    try:
        # Validate file
        validation_result = validate_audio_file(wav_path, logger)
        if validation_result is None:
            return None

        fname, duration = validation_result

        # Transcribe
        logger.debug(f"Transcribing {fname}")
        start_time = time.time()
        result = model.transcribe(audio=wav_path, language="pl")
        transcribe_time = time.time() - start_time

        text = result["text"].strip()
        if not text:
            logger.warning(f"Empty transcription for {fname}")
            return None

        logger.debug(f"Transcribed {fname} in {transcribe_time:.2f}s")

        return {
            "audio": fname,
            "text": text,
            "duration": round(duration, 3),
            "speaker_id": "1",
            "transcribe_time": round(transcribe_time, 3)
        }

    except Exception as e:
        logger.error(f"Failed to transcribe {wav_path}: {e}")
        return None


def process_batch(wav_paths: List[str],
                  model: whisper.Whisper,
                  logger: logging.Logger,
                  max_workers: int) -> List[Dict]:
    """Process a batch of files with concurrent transcription."""
    records = []
    failed_files = []

    # Prepare arguments for worker threads
    args_list = [(path, model, logger) for path in wav_paths]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(transcribe_single_file, args): args[0]
            for args in args_list
        }

        # Process results as they complete
        with tqdm(as_completed(future_to_path),
                  total=len(wav_paths),
                  desc="Transcribing batch") as pbar:
            for future in pbar:
                wav_path = future_to_path[future]
                try:
                    result = future.result()
                    if result is not None:
                        records.append(result)
                        pbar.set_postfix({"Success": len(records), "Failed": len(failed_files)})
                    else:
                        failed_files.append(wav_path)
                except Exception as e:
                    logger.error(f"Exception processing {wav_path}: {e}")
                    failed_files.append(wav_path)

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files: {failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")

    return records


def transcribe_folder(input_dir: str,
                      output_csv: str,
                      model_size: str = "small",
                      device: str = "cuda",
                      max_workers: int = 4,
                      batch_size: int = 10,
                      log_level: str = "INFO"):
    """Main transcription function with improved performance and logging."""

    logger = setup_logging(log_level)
    logger.info(f"Starting transcription with model='{model_size}', device='{device}', workers={max_workers}")

    start_time = time.time()

    # 1) Load Whisper model once
    logger.info(f"Loading Whisper model '{model_size}' on {device}...")
    try:
        model = whisper.load_model(model_size, device=device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # 2) Gather and validate chunk files
    pattern = os.path.join(input_dir, "chunk*.wav")
    wav_paths = sorted(glob.glob(pattern))

    if not wav_paths:
        error_msg = f"No files match '{pattern}'"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Found {len(wav_paths)} audio files to process")

    # Quick validation pass
    logger.info("Validating audio files...")
    valid_paths = []
    for wav_path in tqdm(wav_paths, desc="Validating"):
        if validate_audio_file(wav_path, logger) is not None:
            valid_paths.append(wav_path)

    if not valid_paths:
        error_msg = "No valid audio files found"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Found {len(valid_paths)} valid audio files ({len(wav_paths) - len(valid_paths)} invalid)")

    # 3) Process files in batches
    all_records = []
    total_batches = (len(valid_paths) + batch_size - 1) // batch_size

    for i in range(0, len(valid_paths), batch_size):
        batch_num = i // batch_size + 1
        batch_paths = valid_paths[i:i + batch_size]

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} files)")

        batch_records = process_batch(batch_paths, model, logger, max_workers)
        all_records.extend(batch_records)

        logger.info(f"Batch {batch_num} completed: {len(batch_records)}/{len(batch_paths)} files successful")

    if not all_records:
        error_msg = "No files were successfully transcribed"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 4) Build DataFrame and save CSV
    logger.info("Creating DataFrame and saving results...")

    # Remove transcribe_time column before saving (internal metric)
    output_records = [{k: v for k, v in record.items() if k != 'transcribe_time'}
                      for record in all_records]

    df = pd.DataFrame.from_records(output_records,
                                   columns=["audio", "text", "duration", "speaker_id"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df.to_csv(output_csv, index=False, encoding="utf-8")

    # 5) Summary statistics
    total_time = time.time() - start_time
    total_audio_duration = df['duration'].sum()
    avg_transcribe_time = sum(r.get('transcribe_time', 0) for r in all_records) / len(all_records)

    logger.info("=" * 50)
    logger.info("TRANSCRIPTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"✓ Total files processed: {len(all_records)}/{len(valid_paths)}")
    logger.info(f"✓ Success rate: {len(all_records)/len(valid_paths)*100:.1f}%")
    logger.info(f"✓ Total audio duration: {total_audio_duration:.1f}s ({total_audio_duration/60:.1f}m)")
    logger.info(f"✓ Total processing time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"✓ Average transcription time per file: {avg_transcribe_time:.2f}s")
    logger.info(f"✓ Processing speed: {total_audio_duration/total_time:.1f}x realtime")
    logger.info(f"✓ Output saved to: {output_csv}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-transcribe chunk*.wav files (Polish) → metadata.csv with improved performance")

    parser.add_argument("--input-dir",
                        required=True,
                        help="Folder containing chunk_*.wav files")
    parser.add_argument("--output-csv",
                        required=True,
                        help="Path to write metadata CSV")
    parser.add_argument("--model-size",
                        default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--device",
                        default="cuda",
                        help="torch device: cuda or cpu")
    parser.add_argument("--max-workers",
                        type=int,
                        default=4,
                        help="Maximum number of concurrent workers for I/O operations")
    parser.add_argument("--batch-size",
                        type=int,
                        default=10,
                        help="Number of files to process in each batch")
    parser.add_argument("--log-level",
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    transcribe_folder(
        args.input_dir,
        args.output_csv,
        model_size=args.model_size,
        device=args.device,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        log_level=args.log_level
    )
