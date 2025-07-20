#!/usr/bin/env bash
# This script converts audio files in a directory to WAV format using ffmpeg
# Usage: ./convert.sh "directory_path"
set -e
# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found. Please install it first."
    exit 1
fi
# Check if a directory was provided
if [ $# -eq 0 ]; then
    echo "Error: No directory provided"
    echo "Usage: $0 \"directory_path\""
    exit 1
fi
# Define the input directory
INPUT_DIR="$1"
# Define the output directory
OUTPUT_DIR="${INPUT_DIR}_wav"
mkdir -p "$OUTPUT_DIR"
# Process each audio file in the input directory
for file in "$INPUT_DIR"/*; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        output_file="$OUTPUT_DIR/${filename%.*}.wav"
        echo "Converting $file to $output_file"
        ffmpeg -i "$file" -acodec pcm_s16le -ar 16000 -ac 1 "$output_file" -y
    fi
done
echo "✓ Conversion completed. Files saved in $OUTPUT_DIR/"
