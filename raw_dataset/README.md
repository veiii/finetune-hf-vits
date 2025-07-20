# Cretae dataset for training

## Get audio files
```bash
chmod +x get_base_audio.sh
./get_base_audio.sh URL_TO_AUDIO_FILES
```
## Preprocess with audacity
1. Open Audacity.
2. Import the audio files you downloaded.
3. Select all audio tracks.
4. Voice separation:
   - Go to `Effect` > `Vocal Reduction and Isolation`.
   - Choose `Remove Vocals` and click `OK`.
5. Remove quiet parts:
   - Go to `Effect` > `Truncate Silence`.
   - Set `Silence Length` to 0.1 seconds and click `OK`.
   - Go to `File` > `Export` > `Export Multiple`.
   - Choose `WAV (Microsoft) signed 16-bit PCM` format.
   - Select a destination folder and click `Export`.

## Split audio files into chunks
Use demucs to extract vocal
```bash
chmod +x cut_into_chunks.sh
./cut_into_chunks.sh -i ./processing/output_audacity.wav -o ./processing/output_chunks
```

## Preprocess audio files for training
```bash
python raw_dataset/prepare_voice_audio_file.py --input ./raw_dataset/processing/output_chunks/ --output ./raw_dataset/processing/chunks_cleaned_audio/
```

