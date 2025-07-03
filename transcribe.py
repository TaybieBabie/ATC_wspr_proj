import whisper
import os
import json
import datetime
from config import MODEL_SIZE, AUDIO_DIR, TRANSCRIPT_DIR


def transcribe_audio(audio_file, model_size=MODEL_SIZE):
    """Transcribe audio file using Whisper"""
    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR)

    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    print(f"Transcribing {audio_file}...")
    result = model.transcribe(audio_file)

    # Create transcript filename based on audio filename
    base_name = os.path.basename(audio_file).replace('.wav', '')
    transcript_file = f"{TRANSCRIPT_DIR}{base_name}_transcript.json"

    # Save transcript
    with open(transcript_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Transcript saved to {transcript_file}")
    return result


def batch_transcribe(directory=AUDIO_DIR):
    """Transcribe all WAV files in a directory"""
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            audio_path = os.path.join(directory, file)
            transcribe_audio(audio_path)


if __name__ == "__main__":
    # Test with a single file
    test_file = os.path.join(AUDIO_DIR, os.listdir(AUDIO_DIR)[0])
    if os.path.exists(test_file):
        result = transcribe_audio(test_file)
        print("\nTranscription result:")
        print(result["text"])