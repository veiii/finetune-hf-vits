import logging
import os
import tempfile
import numpy as np
import soundfile as sf
import torch
import whisper
import multiprocessing


def estimate_max_concurrency(
    model_name: str = "small",
    chunk_length: float = 1.0,
    sample_rate: int = 16000,
    device: str = "cuda",
    safety: float = 0.9,
    fallback: int = None,
) -> int:
    """
    Estimate the max number of parallel Whisper inferences you can run
    on `device` without OOM, based on VRAM.

    Args:
      model_name: whisper model (tiny|base|small|medium|large)
      chunk_length: length (s) of dummy WAV to transcribe
      sample_rate: sampling rate for dummy WAV
      device: 'cuda' or 'cpu'
      safety: fraction of VRAM to reserve for OS/overhead
      fallback: if measurement fails, return this or cpu_count()

    Returns:
      max_concurrency >= 1
    """
    if device == "cpu" or not torch.cuda.is_available():
        return fallback or 1

    # helper to sync & get reserved
    def peak_reserved():
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_reserved(device)

    # 1) Load model & measure base reserved VRAM
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = whisper.load_model(model_name, device=device)
    base_reserved = peak_reserved()
    logging.info(f"Peak reserved for {model_name}: {base_reserved}")

    # 2) Create temp silent WAV of length chunk_length
    n = int(chunk_length * sample_rate)
    silent = np.zeros(n, dtype=np.float32)
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tf.name, silent, sample_rate)
    tmp_path = tf.name
    tf.close()

    # 3) Measure reserved VRAM during one inference
    torch.cuda.reset_peak_memory_stats(device)
    _ = model.transcribe(tmp_path, language="pl")
    per_reserved = peak_reserved()
    logging.info(f"Peak reserved for {model_name}: {per_reserved}")
    # cleanup
    os.remove(tmp_path)

    added = per_reserved - base_reserved
    total = torch.cuda.get_device_properties(device).total_memory
    usable = total * safety - base_reserved
    logging.info(f"Total memory for {total}: {usable}")
    logging.info(f"Usable memory for {model_name}: {usable}")

    if added <= 0 or usable <= 0:
        # couldn’t measure extra usage—fallback to CPU count or provided fallback
        return fallback or multiprocessing.cpu_count()

    max_conc = max(1, int(usable // added))
    return max_conc


if __name__ == "__main__":
    conc = estimate_max_concurrency(
        model_name="small",
        chunk_length=2.5,
        sample_rate=16000,
        device="cuda",
        safety=0.9,
    )
    print(f"Estimated max concurrency: {conc}")
