#!/usr/bin/env bash
# This script downloads audio from YouTube URLs using yt-dlp
# Usage: ./get_base_audio_data.sh "URL1 URL2 URL3 ..."

set -e

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "yt-dlp not found. Please install it first."
    exit 1
fi

# Define the output directory
OUTPUT_DIR="base_audio_data"
mkdir -p "$OUTPUT_DIR"

# Check if URLs were provided
if [ $# -eq 0 ]; then
    echo "Error: No URLs provided"
    echo "Usage: $0 \"URL1 URL2 URL3 ...\""
    exit 1
fi

# Process each URL
for url in $1; do
    # Extract video ID (last part of URL) and take first 10 chars
    video_id=$(echo "$url" | grep -oP "(?<=v=)[^&]+" || echo "$url" | grep -oP "(?<=be/)[^&]+")
    filename="${video_id:0:10}"

    echo "Downloading audio from: $url"
    yt-dlp \
        --extract-audio \
        --audio-format wav \
        --audio-quality 0 \
        --output "$OUTPUT_DIR/%(id)s.%(ext)s" \
        --quiet \
        "$url"

    # Rename the downloaded file
    mv "$OUTPUT_DIR/$video_id.wav" "$OUTPUT_DIR/$filename.wav"
done

echo "✓ Downloads completed. Files saved in $OUTPUT_DIR/"