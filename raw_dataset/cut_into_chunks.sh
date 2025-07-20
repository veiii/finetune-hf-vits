#!/usr/bin/env bash

# Script to cut WAV audio files into chunks separated by silence
# with configurable minimum and maximum chunk lengths

set -e  # Exit on error

# Default configuration
INPUT_FILE=""
OUTPUT_DIR="chunks"
MIN_CHUNK_LENGTH="2"      # Minimum chunk length in seconds
MAX_CHUNK_LENGTH="30"     # Maximum chunk length in seconds
SILENCE_THRESHOLD="-30dB" # Silence detection threshold
SILENCE_DURATION="0.5"    # Minimum silence duration to trigger split (seconds)
VERBOSE=false

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -i INPUT_FILE [OPTIONS]

Cut WAV audio file into chunks separated by silence.

Required:
  -i INPUT_FILE     Input WAV file to process

Options:
  -o OUTPUT_DIR     Output directory for chunks (default: chunks)
  -min MIN_LENGTH   Minimum chunk length in seconds (default: 2)
  -max MAX_LENGTH   Maximum chunk length in seconds (default: 30)
  -t THRESHOLD      Silence threshold in dB (default: -30dB)
  -d DURATION       Minimum silence duration in seconds (default: 0.5)
  -v                Verbose output
  -h                Show this help message

Examples:
  $0 -i audio.wav
  $0 -i audio.wav -o output_chunks -min 3 -max 20 -t -25dB -d 1.0
EOF
}

# Function to check if required tools are installed
check_dependencies() {
    if ! command -v ffmpeg &> /dev/null; then
        echo "Error: ffmpeg is required but not installed."
        echo "Please install ffmpeg: sudo apt-get install ffmpeg"
        exit 1
    fi
}

# Function to validate audio file
validate_input() {
    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "Error: Input file '$INPUT_FILE' does not exist."
        exit 1
    fi

    # Check if it's a valid audio file
    if ! ffmpeg -v quiet -i "$INPUT_FILE" -f null - 2>/dev/null; then
        echo "Error: '$INPUT_FILE' is not a valid audio file."
        exit 1
    fi
}

# Function to create output directory
setup_output_dir() {
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        mkdir -p "$OUTPUT_DIR"
        echo "Created output directory: $OUTPUT_DIR"
    fi
}

# Function to detect silence points and split audio
split_audio() {
    local input_file="$1"
    local base_name=$(basename "$input_file" .wav)

    echo "Processing: $input_file"
    echo "Output directory: $OUTPUT_DIR"
    echo "Min chunk length: ${MIN_CHUNK_LENGTH}s"
    echo "Max chunk length: ${MAX_CHUNK_LENGTH}s"
    echo "Silence threshold: $SILENCE_THRESHOLD"
    echo "Silence duration: ${SILENCE_DURATION}s"
    echo ""

    # First, detect silence points
    echo "Detecting silence points..."
    local silence_file=$(mktemp)

    ffmpeg -i "$input_file" -af "silencedetect=noise=${SILENCE_THRESHOLD}:d=${SILENCE_DURATION}" -f null - 2>&1 | \
        grep "silence_start\|silence_end" > "$silence_file"

    if [[ $VERBOSE == true ]]; then
        echo "Silence detection output:"
        cat "$silence_file"
        echo ""
    fi

    # Parse silence points and create segments
    local segments=()
    local start_time=0
    local chunk_count=0

    while IFS= read -r line; do
        if [[ $line == *"silence_start"* ]]; then
            local end_time=$(echo "$line" | grep -o "silence_start: [0-9.]*" | cut -d' ' -f2)
            local duration=$(echo "$end_time - $start_time" | bc -l)

            # Check if segment meets minimum length requirement
            if (( $(echo "$duration >= $MIN_CHUNK_LENGTH" | bc -l) )); then
                # If segment is too long, split it further
                if (( $(echo "$duration > $MAX_CHUNK_LENGTH" | bc -l) )); then
                    local sub_start=$start_time
                    while (( $(echo "$sub_start < $end_time" | bc -l) )); do
                        local sub_end=$(echo "$sub_start + $MAX_CHUNK_LENGTH" | bc -l)
                        if (( $(echo "$sub_end > $end_time" | bc -l) )); then
                            sub_end=$end_time
                        fi
                        local sub_duration=$(echo "$sub_end - $sub_start" | bc -l)

                        if (( $(echo "$sub_duration >= $MIN_CHUNK_LENGTH" | bc -l) )); then
                            segments+=("$sub_start:$sub_end")
                        fi
                        sub_start=$sub_end
                    done
                else
                    segments+=("$start_time:$end_time")
                fi
            fi
        elif [[ $line == *"silence_end"* ]]; then
            start_time=$(echo "$line" | grep -o "silence_end: [0-9.]*" | cut -d' ' -f2)
        fi
    done < "$silence_file"

    # Handle the last segment (from last silence end to file end)
    local total_duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$input_file")
    if [[ -n "$start_time" ]] && (( $(echo "$start_time < $total_duration" | bc -l) )); then
        local final_duration=$(echo "$total_duration - $start_time" | bc -l)
        if (( $(echo "$final_duration >= $MIN_CHUNK_LENGTH" | bc -l) )); then
            segments+=("$start_time:$total_duration")
        fi
    fi

    echo "Found ${#segments[@]} valid segments"
    echo ""

    # Extract segments
    for segment in "${segments[@]}"; do
        local start=$(echo "$segment" | cut -d':' -f1)
        local end=$(echo "$segment" | cut -d':' -f2)
        local duration=$(echo "$end - $start" | bc -l)

        chunk_count=$((chunk_count + 1))
        local output_file="${OUTPUT_DIR}/${base_name}_chunk$(printf "%03d" $chunk_count).wav"

        echo "Extracting chunk $chunk_count: ${start}s to ${end}s (duration: ${duration}s)"

        if [[ $VERBOSE == true ]]; then
            ffmpeg -i "$input_file" -ss "$start" -to "$end" -c copy "$output_file" -y
        else
            ffmpeg -i "$input_file" -ss "$start" -to "$end" -c copy "$output_file" -y 2>/dev/null
        fi

        if [[ $? -eq 0 ]]; then
            echo "  ✓ Saved: $output_file"
        else
            echo "  ✗ Failed to create: $output_file"
        fi
    done

    # Cleanup
    rm -f "$silence_file"

    echo ""
    echo "Processing complete! Created $chunk_count chunks in $OUTPUT_DIR"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -min|--min-length)
            MIN_CHUNK_LENGTH="$2"
            shift 2
            ;;
        -max|--max-length)
            MAX_CHUNK_LENGTH="$2"
            shift 2
            ;;
        -t|--threshold)
            SILENCE_THRESHOLD="$2"
            shift 2
            ;;
        -d|--duration)
            SILENCE_DURATION="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: Input file is required."
    usage
    exit 1
fi

# Main execution
echo "Audio Chunk Splitter"
echo "==================="
echo ""

check_dependencies
validate_input
setup_output_dir
split_audio "$INPUT_FILE"

echo ""
echo "All done!"
