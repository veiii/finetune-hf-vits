import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def extract_vocals_with_demucs(input_file, temp_output_dir):
    """
    Extract vocals from audio file using demucs

    Args:
        input_file: Path to input audio file
        temp_output_dir: Temporary directory for demucs output

    Returns:
        Path to extracted vocals file or None if failed
    """
    try:
        # Run demucs command with vocals separation only
        result = subprocess.run([
            "demucs",
            "--two-stems", "vocals",
            "--out", temp_output_dir,
            str(input_file)
        ], capture_output=True, text=True, check=True)

        # Find the vocals file in the demucs output structure
        # Demucs creates: temp_output_dir/htdemucs_ft/{filename_without_ext}/vocals.wav
        input_stem = Path(input_file).stem
        vocals_file = Path(temp_output_dir) / "htdemucs_ft" / input_stem / "vocals.wav"

        if vocals_file.exists():
            return vocals_file
        else:
            # Try alternative model names that demucs might use
            for model_name in ["htdemucs", "mdx_extra", "mdx"]:
                alt_vocals_file = Path(temp_output_dir) / model_name / input_stem / "vocals.wav"
                if alt_vocals_file.exists():
                    return alt_vocals_file

            print(f"Warning: Vocals file not found at expected location: {vocals_file}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_file}: {e}")
        print(f"Stderr: {e.stderr}")
        return None


def process_audio_directory(input_dir, output_dir, temp_dir=None, max_workers=None):
    """
    Process all WAV files in input directory to extract vocals

    Args:
        input_dir: Directory containing input WAV files
        output_dir: Directory to save extracted vocals
        temp_dir: Temporary directory for demucs processing (optional)
        max_workers: Maximum number of worker threads (optional)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup temporary directory
    if temp_dir is None:
        temp_path = output_path / "temp_demucs"
    else:
        temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    # Find all WAV files
    wav_files = list(input_path.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return

    # Determine number of workers
    if max_workers is None:
        max_workers = min(os.cpu_count(), len(wav_files))

    print(f"Found {len(wav_files)} WAV files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Temporary directory: {temp_path}")
    print(f"Using {max_workers} worker threads")
    print()

    # Thread-safe counters
    successful_count = 0
    failed_count = 0
    count_lock = threading.Lock()

    def process_single_file(wav_file):
        """Process a single WAV file"""
        nonlocal successful_count, failed_count

        try:
            # Create unique temp directory for this thread to avoid conflicts
            thread_temp_dir = temp_path / f"thread_{threading.get_ident()}"
            thread_temp_dir.mkdir(exist_ok=True)

            # Extract vocals using demucs
            vocals_file = extract_vocals_with_demucs(wav_file, thread_temp_dir)

            if vocals_file and vocals_file.exists():
                # Copy vocals file to output directory with original filename
                output_file = output_path / f"{wav_file.stem}_vocals.wav"
                shutil.copy2(vocals_file, output_file)

                with count_lock:
                    successful_count += 1

                return f"✓ Processed: {wav_file.name} -> {output_file.name}"
            else:
                with count_lock:
                    failed_count += 1
                return f"✗ Failed: {wav_file.name}"

        except Exception as e:
            with count_lock:
                failed_count += 1
            return f"✗ Error processing {wav_file.name}: {str(e)}"

    # Process files with multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, wav_file): wav_file
                         for wav_file in wav_files}

        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_file),
                          total=len(future_to_file),
                          desc="Extracting vocals"):
            try:
                result_message = future.result()
                tqdm.write(result_message)
            except Exception as e:
                wav_file = future_to_file[future]
                tqdm.write(f"✗ Unexpected error with {wav_file.name}: {str(e)}")

    # Cleanup temporary directory
    try:
        shutil.rmtree(temp_path)
        print(f"\nCleaned up temporary directory: {temp_path}")
    except Exception as e:
        print(f"\nWarning: Could not clean up temporary directory {temp_path}: {e}")

    # Summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Total: {len(wav_files)} files")


def main():
    parser = argparse.ArgumentParser(description="Extract vocals from WAV files using demucs")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing WAV files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for vocals files")
    parser.add_argument("-t", "--temp", help="Temporary directory for demucs processing (optional)")
    parser.add_argument("-w", "--max-workers", type=int, default=None,
                        help="Maximum number of worker threads (default: number of CPU cores)")
    parser.add_argument("--check-demucs", action="store_true", help="Check if demucs is installed")

    args = parser.parse_args()

    # Check if demucs is installed
    if args.check_demucs:
        try:
            result = subprocess.run(["demucs", "--help"], capture_output=True, text=True)
            print("✓ demucs is installed and accessible")
            return
        except FileNotFoundError:
            print("✗ demucs is not installed or not in PATH")
            print("Install with: pip install demucs")
            return

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return

    # Check if demucs is available
    try:
        subprocess.run(["demucs", "--help"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("Error: demucs is not installed or not in PATH")
        print("Install with: pip install demucs")
        return
    except subprocess.CalledProcessError:
        print("Error: demucs command failed")
        return

    # Process the directory
    process_audio_directory(args.input, args.output, args.temp, args.max_workers)


if __name__ == "__main__":
    main()