# faster_transcribe.py
from faster_whisper import WhisperModel
import torch
from datetime import datetime
import json
import os


class FasterWhisperTranscriber:
    def __init__(self, model_size="base", device="auto", compute_type="float16"):
        # Select device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                self.compute_type = "float16"  # Use half precision for speed
            else:
                self.device = "cpu"
                self.compute_type = "int8"
        else:
            self.device = device

        print(f"🚀 Loading Faster-Whisper {model_size} on {self.device}")

        # Load model
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type
        )

        print("✅ Model loaded successfully")

    def transcribe_audio(self, audio_file):
        """Transcribe with faster-whisper"""
        start_time = datetime.now()

        try:
            segments, info = self.model.transcribe(
                audio_file,
                language="en",
                task="transcribe",
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.3,
                condition_on_previous_text=False,
                initial_prompt="Aviation radio communication with callsigns, altitudes, and flight instructions.",
                word_timestamps=True
            )

            # Combine segments
            text = ""
            words = []
            for segment in segments:
                text += segment.text
                if hasattr(segment, 'words'):
                    words.extend(segment.words)

            result = {
                'text': text.strip(),
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'words': words,
                'metadata': {
                    'model': f"faster-whisper-{self.model.model_size}",
                    'device': self.device,
                    'compute_type': self.compute_type,
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'audio_file': audio_file,
                    'timestamp': datetime.now().isoformat()
                }
            }

            processing_time = result['metadata']['processing_time']
            print(f"✅ Transcription complete in {processing_time:.2f}s")

            return result

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None