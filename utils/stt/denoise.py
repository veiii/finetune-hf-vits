#!/usr/bin/env python3
"""
async_denoise.py

A GPU‐accelerated, chunked, async & threaded denoiser with progress bar.

Usage:
    python async_denoise.py input.wav output.wav
"""

import argparse
import asyncio
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
from denoiser import pretrained
from tqdm.auto import tqdm


class AsyncDenoiser:
    def __init__(
        self,
        model_name: str = "dns48",
        device: str = "cuda",
        mem_fraction: float = 0.5,
        max_workers: int = 2,
    ):
        """
        model_name: which denoiser (dns48 or dns64)
        device: "cuda" or "cpu"
        mem_fraction: fraction of GPU RAM to reserve for this process
        max_workers: threads to preprocess/postprocess chunks
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # throttle GPU memory if supported
        if self.device.type == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(mem_fraction, self.device)
            except Exception:
                pass

        # load model
        model = pretrained.dns48() if model_name == "dns48" else pretrained.dns64()
        self.model = model.to(self.device).eval()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _denoise_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Blocking call: denoise a single chunk (Frame shape: [T])"""
        # to tensor: [1, T]
        wav_t = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
        with torch.no_grad():
            clean_t = self.model(wav_t)
        clean = clean_t.squeeze().cpu().numpy()
        return clean

    async def denoise(
        self,
        audio: np.ndarray,
        sample_rate: int,
        chunk_duration: float = 2.0,
    ) -> np.ndarray:
        """
        Split `audio` into ~chunk_duration-second chunks,
        denoise them concurrently in threads (GPU work queued),
        and stitch back into one array.
        """
        frame_len = int(chunk_duration * sample_rate)
        n_frames = math.ceil(len(audio) / frame_len)

        # pad audio so it's a multiple of frame_len
        pad_len = frame_len * n_frames - len(audio)
        audio_padded = np.concatenate([audio, np.zeros(pad_len, dtype=audio.dtype)])

        results = [None] * n_frames
        loop = asyncio.get_event_loop()

        async def schedule(i: int):
            start = i * frame_len
            end = start + frame_len
            chunk = audio_padded[start:end]
            # run blocking _denoise_chunk in thread
            clean = await loop.run_in_executor(
                self.executor, self._denoise_chunk, chunk
            )
            results[i] = clean

        # schedule all denoising jobs with progress bar
        tasks = []
        for i in range(n_frames):
            tasks.append(schedule(i))

        for f in tqdm(asyncio.as_completed(tasks), total=n_frames, desc="Denoising"):
            await f

        # stitch and trim padding
        denoised = np.concatenate(results)[: len(audio)]
        return denoised


async def main(args):
    # 1) load file
    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 2) initialize denoiser
    denoiser = AsyncDenoiser(
        model_name=args.model,
        device=args.device,
        mem_fraction=args.mem_fraction,
        max_workers=args.threads,
    )

    # 3) denoise
    clean = await denoiser.denoise(audio, sr, chunk_duration=args.chunk_duration)

    # 4) normalize & write out
    clean = clean / (np.max(np.abs(clean)) + 1e-9) * 0.99
    sf.write(args.output, clean, sr, subtype="PCM_16")
    print(f"✔ Denoised file written to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async GPU Denoiser")
    parser.add_argument("input", help="Path to input WAV")
    parser.add_argument("output", help="Path to output denoised WAV")
    parser.add_argument("--model", choices=["dns48", "dns64"], default="dns48")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--mem_fraction", type=float, default=0.5, help="GPU RAM fraction"
    )
    parser.add_argument(
        "--chunk_duration", type=float, default=2.0, help="seconds per chunk"
    )
    parser.add_argument(
        "--threads", type=int, default=2, help="max preprocessing threads"
    )
    args = parser.parse_args()

    asyncio.run(main(args))
