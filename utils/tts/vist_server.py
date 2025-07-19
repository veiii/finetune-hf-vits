import argparse
import gc
import asyncio
import logging
import os
import time

import numpy as np
from pydub import AudioSegment
from transformers import VitsModel, AutoTokenizer, set_seed
from tqdm.asyncio import tqdm
import signal
import torch
import sys

PAUSE_DURATION_MS = 800
PAUSE_DURATION_MS_END_OF_SEQUENCES = 500
GPU_MEMORY_MAX_USE_GB = torch.cuda.get_device_properties(0).total_memory * 0.8  # Use 80% of total GPU memory
set_seed(456)  # make model determistic
LOGING_LEVEL = logging.INFO
FORCE_CLEAR_INPUTS_AFTER_INTERFERENCE = True
FORCE_CLEAR_CACHE_AFTER_INTERFERENCE = True

logging.basicConfig(level=LOGING_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


# force clean cache after program interrupt
def handle_interrupt(signum, frame):
    print("🔌 Interrupt received. Clearing GPU cache...")
    torch.cuda.empty_cache()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_interrupt)


class VISTModelWrapper:
    def __init__(self, model):
        self.model = model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-pol")
        self.semaphore = asyncio.Semaphore(self.estimate_capacity())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(
            f"Using device: {self.device}, of capacity: {self.estimate_capacity()}"
        )

    @staticmethod
    def estimate_capacity():
        def _bytes_to(byte, to, bsize=1024):
            a = {"k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6}
            return byte / (bsize ** a[to])

        # Simulate available GPU capacity logic
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = min(total_memory, GPU_MEMORY_MAX_USE_GB * 1024**3)
        mem_per_request = 500 * 1024**2  # 500MB per request (estimate)
        logging.info(
            f"Total memory: {_bytes_to(total_memory, to='m')}, available memory: {_bytes_to(available_memory, to='m')}, mem_per_request: {_bytes_to(mem_per_request, to='m')}"
        )
        return max(1, available_memory // mem_per_request)

    async def process_request(self, input_data):
        async with self.semaphore:
            return await asyncio.to_thread(self.run_on_gpu, input_data)

    def run_on_gpu(self, input_text) -> AudioSegment:
        if input_text == "[pause]":
            return AudioSegment.silent(duration=PAUSE_DURATION_MS)
        torch.cuda.reset_peak_memory_stats()  # Reset before synthesis
        inputs = self.tokenizer(
            input_text, add_special_tokens=True, return_tensors="pt", padding=True
        )
        logging.debug(f"Tokenized inputs: {inputs}")
        if inputs["input_ids"].size(1) == 0:
            logging.warning(f"Skipping track due to invalid input: {input_text}")
            return AudioSegment.empty()

        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs).waveform
        speech = output.squeeze().cpu().numpy()
        # Log GPU memory usage
        current_mem = torch.cuda.memory_allocated(self.device) / (1024**2)
        peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        # force clear cache
        if FORCE_CLEAR_INPUTS_AFTER_INTERFERENCE:
            del inputs
        if FORCE_CLEAR_CACHE_AFTER_INTERFERENCE:
            gc.collect()
            torch.cuda.empty_cache()
        tqdm.write(
            f"Memory used: {current_mem:.2f} MB, Peak during synthesis: {peak_mem:.2f} MB"
        )
        # convert to AudioSegment
        scaled_output = np.int16(speech * 32767)
        audio = AudioSegment(
            scaled_output.tobytes(),
            frame_rate=self.model.config.sampling_rate,
            sample_width=2,
            channels=1,
        )
        return audio


def load_text_from_file(txt_file_path):
    with open(txt_file_path, "r", encoding="utf-8") as f:
        return f.read()


def slow_down(audio: AudioSegment, speed: float = 0.9) -> AudioSegment:
    # Slows down by resampling
    if speed == 1.0:
        return audio
    new_frame_rate = int(audio.frame_rate * speed)
    slowed = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
    return slowed.set_frame_rate(audio.frame_rate)


def preprocess_audio_segement(audio: AudioSegment):
    processed_audio = slow_down(audio, speed=1.0)
    if PAUSE_DURATION_MS_END_OF_SEQUENCES > 0:
        processed_audio += AudioSegment.silent(
            duration=PAUSE_DURATION_MS_END_OF_SEQUENCES
        )
    return processed_audio


def _log_track_length(t):
    return f"{int(t / 3600)}H {int((t / 60) % 60) if t / 3600 > 0 else int(t / 60)}M {int(t % 60)}S"


async def main(inputs, mp3_output_path, model_name="facebook/mms-tts-pol"):
    logging.debug(f"Running:{model_name}, Output mp3 path: {mp3_output_path}")
    # check if model name is a local file path
    if model_name == "facebook/mms-tts-pol":
        model = VitsModel.from_pretrained(model_name)
    elif model_name.startswith("/") or model_name.startswith("./"):
        model = VitsModel.from_pretrained(
            model_name,
            local_files_only=True,  # Don't try to download from HF Hub
            trust_remote_code=True,
        )
    else:
        Exception("Wrong model name")

    vist = VISTModelWrapper(model)
    logging.info(f"Processing... ")
    tasks = [vist.process_request(i) for i in inputs]
    results = await tqdm.gather(*tasks)
    track = AudioSegment.empty()
    for idx, res in tqdm(enumerate(results)):
        logging.debug(f"Processing result: {res}")
        # slow down audio and make silence at end od sequense
        processed_audio = preprocess_audio_segement(res)
        track += processed_audio
    logging.info(f"Track length: {_log_track_length(track.duration_seconds)}")
    track.export(mp3_output_path, format="mp3")
    logging.info(f"Exported MP3: {mp3_output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Text-to-Speech Synthesizer with VITS")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input text file"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path to output MP3 file"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="facebook/mms-tts-pol",
        help="Name or path of VITS model to use",
    )
    parser.add_argument(
        "--max-lines",
        "-n",
        type=int,
        default=20,
        help="Number of lines to process from the input text, IF: 0 process all file lines.",
    )
    input_args = parser.parse_args()
    if not os.path.exists(input_args.input):
        logging.error(f"Input file not found: {input_args.input}")
        exit(1)
    return input_args


if __name__ == "__main__":
    s = time.perf_counter()  # Elapsed time counter
    args = parse_arguments()
    logging.info(f"Loading text from: {args.input}")
    text = load_text_from_file(args.input)
    text_list = text.split("\n")
    playbook = [line.strip() if line.strip() else "[pause]" for line in text_list]
    max_lines = args.max_lines if args.max_lines != 0 else len(text_list)
    logging.info(f"Playbook length: {max_lines}")
    playbook = playbook[:max_lines]
    logging.debug(f"Playbook: {playbook}")
    logging.info(f"Synthesizing {len(playbook)} lines with model: {args.model}")
    asyncio.run(main(playbook, args.output, args.model))
    elapsed = time.perf_counter() - s
    logging.info(f"{__file__} executed in {elapsed:0.2f} seconds.")
