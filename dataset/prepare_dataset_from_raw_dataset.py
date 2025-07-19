import pandas as pd
import os
import numpy as np
import soundfile as sf
from datasets import Dataset, DatasetDict, Audio, load_from_disk
import shutil

def create_output_directory(output_path):
    """Create output directory if it does not exist"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        # Clear the directory if it already exists
        shutil.rmtree(output_path)
        os.makedirs(output_path)

def load_audio(audio_path):
    """Load and process audio file"""
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_array, sampling_rate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        return {
            "path": audio_path,
            "array": audio_array.astype(np.float32),
            "sampling_rate": sampling_rate
        }
    except Exception as e:
        print(f"Error loading {audio_path}: {str(e)}")
        return None

def create_dataset_from_csv(csv_path, audio_dir, output_path):
    """
    Creates a dataset from a CSV file or list of CSV files containing audio metadata.
    Copies audio files to a standardized location with unique names.

    Args:
        csv_path (str or list): Path to the CSV file or list of CSV file paths containing audio metadata.
        audio_dir (str or list): Directory containing the audio files (used for relative paths).
                                If list, should match the length of csv_path list.
        output_path (str): Path where the processed dataset will be saved.
    """
    # Handle both single CSV path and list of CSV paths
    if isinstance(csv_path, str):
        csv_paths = [csv_path]
    elif isinstance(csv_path, list):
        csv_paths = csv_path
    else:
        raise ValueError("csv_path must be a string or a list of strings")

    # Handle both single audio dir and list of audio dirs
    if isinstance(audio_dir, str):
        audio_dirs = [audio_dir] * len(csv_paths)  # Use same audio_dir for all CSVs
    elif isinstance(audio_dir, list):
        audio_dirs = audio_dir
        if len(audio_dirs) != len(csv_paths):
            raise ValueError(f"Number of audio directories ({len(audio_dirs)}) must match number of CSV files ({len(csv_paths)})")
    else:
        raise ValueError("audio_dir must be a string or a list of strings")

    # Load and concatenate all CSV files
    dataframes = []
    for i, (csv_file_path, audio_directory) in enumerate(zip(csv_paths, audio_dirs)):
        if not os.path.exists(csv_file_path):
            print(f"Warning: CSV file not found: {csv_file_path}")
            continue

        print(f"Loading CSV file {i+1}/{len(csv_paths)}: {csv_file_path}")
        print(f"  Using audio directory: {audio_directory}")
        df = pd.read_csv(csv_file_path, header=None, names=['audio', 'transcript', 'duration'])

        # Add source info to track which CSV and audio_dir each row came from
        df['source_csv'] = csv_file_path
        df['source_audio_dir'] = audio_directory

        print(f"  - Loaded {len(df)} rows from {csv_file_path}")
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No valid CSV files found")

    # Concatenate all dataframes
    df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows after concatenation: {len(df)}")

    # Filter out rows with empty transcripts
    df = df[df['transcript'].notna() & (df['transcript'].str.strip() != '')]
    print(f"Rows after filtering empty transcripts: {len(df)}")

    # Create dataset directory structure
    dataset_audio_dir = os.path.join(output_path, "wavs")
    create_output_directory(output_path)
    os.makedirs(dataset_audio_dir, exist_ok=True)

    audio_data = []
    transcripts = []
    speaker_ids = []
    processed_rows = []

    print("Processing audio files and copying to dataset directory...")
    for idx, row in df.iterrows():
        # Skip rows with empty transcripts
        if pd.isna(row['transcript']) or str(row['transcript']).strip() == '':
            continue

        # Get original audio path using the corresponding audio directory for this row
        original_audio_path = row['audio']
        row_audio_dir = row['source_audio_dir']

        if not os.path.isabs(original_audio_path):
            original_audio_path = os.path.join(row_audio_dir, original_audio_path)

        if not os.path.exists(original_audio_path):
            print(f"Warning: Audio file not found: {original_audio_path}")
            continue

        # Create unique filename
        file_extension = os.path.splitext(original_audio_path)[1]
        unique_filename = f"audio_{idx:06d}{file_extension}"
        new_audio_path = os.path.join(dataset_audio_dir, unique_filename)

        # Copy audio file to dataset directory
        try:
            shutil.copy2(original_audio_path, new_audio_path)
        except Exception as e:
            print(f"Error copying file {original_audio_path}: {str(e)}")
            continue

        # Load audio data
        audio = load_audio(new_audio_path)
        if audio is not None:
            audio_data.append(audio)
            transcripts.append(str(row['transcript']).strip())  # Use 'transcript' from CSV
            speaker_ids.append("1")

            # Store processed row info for parquet file
            processed_rows.append({
                'audio': new_audio_path,  # Use absolute path instead of relative
                'transcript': str(row['transcript']).strip(),  # Training script expects 'text' column
                'duration': row.get('duration', 0),
                'speaker_id': "1"
            })

    print(f"Successfully processed {len(audio_data)} audio files")

    # Create and save parquet file with relative paths
    df_processed = pd.DataFrame(processed_rows)
    parquet_path = os.path.join(output_path, "dataset.parquet")
    df_processed.to_parquet(parquet_path, index=False)
    print(df_processed)

    print(f"Dataset created successfully with {len(audio_data)} samples")
    print(f"Dataset saved to: {output_path}")
    print(f"Audio files copied to: {dataset_audio_dir}")
    print(f"Parquet file saved to: {parquet_path}")

    # Print summary
    print("\nDataset Summary:")
    print(f"  - Total audio files: {len(audio_data)}")
    print(f"  - Audio directory: data/")
    print(f"  - Parquet file: dataset.parquet")
    print(f"  - HuggingFace dataset: saved to disk")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create dataset from CSV files containing audio metadata"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file or comma-separated list of CSV files"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing audio files or comma-separated list of audio directories (one per CSV file)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory where the dataset will be saved"
    )

    args = parser.parse_args()

    # Handle comma-separated list of CSV files
    if ',' in args.csv_path:
        csv_paths = [path.strip() for path in args.csv_path.split(',')]
    else:
        csv_paths = args.csv_path

    # Handle comma-separated list of audio directories
    if ',' in args.audio_dir:
        audio_dirs = [path.strip() for path in args.audio_dir.split(',')]
    else:
        audio_dirs = args.audio_dir

    print(f"Creating dataset from CSV: {csv_paths}")
    print(f"Audio directories: {audio_dirs}")
    print(f"Output path: {args.output_path}")
    print("-" * 50)

    try:
        create_dataset_from_csv(
            csv_path=csv_paths,
            audio_dir=audio_dirs,
            output_path=args.output_path
        )
        print("-" * 50)
        print("✅ Dataset creation completed successfully!")
    except Exception as e:
        print(f"❌ Error creating dataset: {str(e)}")
        exit(1)

if __name__ == "__main__":
    import logging
    import tqdm

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("tqdm").setLevel(logging.INFO)

    # Call the main function to run the script
    main()