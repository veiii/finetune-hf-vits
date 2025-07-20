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
import numpy as np


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


def validate_and_preprocess_audio(wav_path: str, logger: logging.Logger) -> Optional[Tuple[str, float, np.ndarray]]:
    """Validate and preprocess audio file, return filename, duration, and audio data."""
    try:
        # Read audio file
        audio_data, sample_rate = sf.read(wav_path, dtype=np.float32)

        # Validate basic properties
        if len(audio_data) == 0:
            logger.warning(f"Empty audio file: {wav_path}")
            return None

        duration = len(audio_data) / sample_rate
        if duration <= 0.1:  # Too short
            logger.warning(f"Audio too short ({duration:.2f}s): {wav_path}")
            return None

        if duration > 30.0:  # Too long for Whisper
            logger.warning(f"Audio too long ({duration:.2f}s), truncating to 30s: {wav_path}")
            audio_data = audio_data[:int(30.0 * sample_rate)]
            duration = 30.0

        # Handle stereo audio (convert to mono)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            logger.debug(f"Converted stereo to mono: {wav_path}")

        # Normalize audio to prevent clipping
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            logger.debug(f"Normalized audio levels: {wav_path}")

        # Check for silence or very low volume
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 0.001:  # Very quiet audio
            logger.warning(f"Very low audio level (RMS: {rms:.6f}): {wav_path}")
            return None

        # # Resample to 16kHz if needed (Whisper's expected sample rate)
        # if sample_rate != 16000:
        #     logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz: {wav_path}")
        #     # Simple resampling using linear interpolation
        #     target_length = int(len(audio_data) * 16000 / sample_rate)
        #     audio_data = np.interp(
        #         np.linspace(0, len(audio_data), target_length),
        #         np.arange(len(audio_data)),
        #         audio_data
        #     )
        #     duration = len(audio_data) / 16000

        return os.path.basename(wav_path), duration, audio_data

    except Exception as e:
        logger.error(f"Failed to process audio file {wav_path}: {e}")
        return None


def transcribe_single_file(args: Tuple[str, whisper.Whisper, logging.Logger]) -> Optional[Dict]:
    """Transcribe a single audio file with enhanced error handling."""
    wav_path, model, logger = args

    try:
        # Validate and preprocess file
        validation_result = validate_and_preprocess_audio(wav_path, logger)
        if validation_result is None:
            return None

        fname, duration, audio_data = validation_result

        # Transcribe using preprocessed audio data
        logger.debug(f"Transcribing {fname} (duration: {duration:.2f}s)")
        start_time = time.time()

        # Use the preprocessed audio data directly
        result = model.transcribe(
            audio=audio_data,
            language="pl",
            fp16=False,  # Use fp32 for stability
            verbose=False  # Reduce noise in logs
        )

        transcribe_time = time.time() - start_time

        text = result["text"].strip()
        if not text:
            logger.warning(f"Empty transcription for {fname}")
            return None

        # Additional validation of transcription quality
        if len(text) < 3:  # Very short transcription might be noise
            logger.warning(f"Very short transcription ({len(text)} chars) for {fname}: '{text}'")
            return None

        logger.debug(f"Transcribed {fname} in {transcribe_time:.2f}s: '{text[:50]}{'...' if len(text) > 50 else ''}'")

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
        if validate_and_preprocess_audio(wav_path, logger) is not None:
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
