import whisper
import torch
import numpy as np
from datetime import datetime
import json
import os
import librosa
import gc

from config import MODEL_SIZE, TRANSCRIPT_DIR, ENABLE_GPU, SAMPLE_RATE, WHISPER_COMPUTE_TYPE, USE_FASTER_WHISPER


class GPUWhisperTranscriber:
    def __init__(self, model_size="base", device="auto", optimize_for_radio=True):
        self.model_size = model_size
        self.optimize_for_radio = optimize_for_radio

        # Determine device with better error handling
        if device == "auto":
            self.setup_device()
        else:
            self.device = device

        # Load model with appropriate optimizations
        self.load_model()

        # Optimize for radio if requested
        if optimize_for_radio:
            self.setup_radio_optimization()

    def setup_device(self):
        """Setup device with detailed diagnostics"""
        if ENABLE_GPU and torch.cuda.is_available():
            try:
                # Try to initialize CUDA explicitly
                torch.cuda.init()
                self.device = "cuda"

                # Print detailed GPU information
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"🚀 Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")

                # Check CUDA version compatibility
                cuda_version = torch.version.cuda
                print(f"🔧 CUDA Version: {cuda_version}")

                # Check available memory
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"💾 GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

            except Exception as e:
                print(f"⚠️ GPU initialization error: {e}")
                print("⚠️ Falling back to CPU")
                self.device = "cpu"
        else:
            self.device = "cpu"
            reason = "disabled in config" if not ENABLE_GPU else "not available"
            print(f"⚠️ Using CPU (GPU {reason})")

    def load_model(self):
        """Load Whisper model with appropriate optimizations"""
        print(f"📦 Loading Whisper model: {self.model_size} on {self.device}")

        try:
            # Clear CUDA cache before loading model
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            if USE_FASTER_WHISPER and self.device == "cuda":
                # Use faster-whisper if available (better GPU performance)
                try:
                    from faster_whisper import WhisperModel

                    # Map compute type based on config
                    compute_type = WHISPER_COMPUTE_TYPE

                    print(f"🚀 Using faster-whisper with compute type: {compute_type}")
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=compute_type
                    )
                    self.using_faster_whisper = True
                    print(f"✅ faster-whisper model loaded successfully")
                    return
                except ImportError:
                    print("⚠️ faster-whisper not installed, falling back to standard whisper")
                except Exception as e:
                    print(f"⚠️ Error loading faster-whisper model: {e}")
                    print("⚠️ Falling back to standard whisper")

            # Standard whisper with GPU optimizations
            if self.device == "cuda":
                # Load model directly to GPU with optimized settings
                self.model = whisper.load_model(self.model_size)
                self.model = self.model.to(self.device)

                # Enable mixed precision for better performance
                if hasattr(torch.cuda, 'amp') and WHISPER_COMPUTE_TYPE == "float16":
                    print("🚀 Enabling mixed precision for faster inference")
                    self.model = self.model.half()  # Convert to half precision
            else:
                # CPU model loading
                self.model = whisper.load_model(self.model_size, device=self.device)

            self.using_faster_whisper = False
            print(f"✅ Standard whisper model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def setup_radio_optimization(self):
        """Setup optimizations for radio communications"""
        # Radio communications typically have:
        # - Limited vocabulary (aviation terms, callsigns, numbers)
        # - Compressed audio
        # - Background noise
        # - Clipped transmissions

        # Common aviation terms to help with recognition
        self.aviation_vocab = [
            "aircraft", "runway", "approach", "departure", "tower", "ground",
            "clearance", "taxi", "takeoff", "landing", "altitude", "heading",
            "squawk", "transponder", "contact", "frequency", "roger", "wilco",
            "unable", "standby", "say again", "flight level", "descend", "climb",
            "maintain", "proceed", "hold", "cleared", "traffic", "report",
            "visual", "radar", "primary", "secondary", "ident", "negative",
            "affirmative", "correction", "verify", "confirm"
        ]

    def preprocess_audio(self, audio_path):
        """Preprocess audio for better radio transcription"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

            if self.optimize_for_radio:
                # Apply radio-specific preprocessing

                # 1. Normalize audio
                audio = librosa.util.normalize(audio)

                # 2. Apply high-pass filter to remove low-frequency noise
                audio = librosa.effects.preemphasis(audio, coef=0.97)

                # 3. Reduce noise (simple spectral subtraction)
                # This is a basic approach - you might want to use more sophisticated methods
                audio = self.reduce_noise(audio, sr)

                # 4. Boost frequencies typical of radio communications (300-3400 Hz)
                # This mimics radio frequency response
                audio = self.radio_filter(audio, sr)

            return audio

        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None

    def reduce_noise(self, audio, sample_rate):
        """Basic noise reduction for radio audio"""
        try:
            # Simple noise gate - reduce very quiet parts
            threshold = np.percentile(np.abs(audio), 20)  # 20th percentile as noise floor
            audio = np.where(np.abs(audio) < threshold, audio * 0.1, audio)
            return audio
        except:
            return audio

    def radio_filter(self, audio, sample_rate):
        """Apply radio-like frequency filtering"""
        try:
            # Apply bandpass filter to mimic radio frequencies
            from scipy import signal

            # Radio typical range: 300-3400 Hz
            nyquist = sample_rate * 0.5
            low = 300 / nyquist
            high = 3400 / nyquist

            b, a = signal.butter(4, [low, high], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio)

            return filtered_audio
        except:
            # If scipy not available, return original
            return audio

    def transcribe_audio(self, audio_file, options=None):
        """Transcribe audio with GPU acceleration and radio optimizations"""
        start_time = datetime.now()

        # Default options optimized for radio
        default_options = {
            "language": "en",
            "task": "transcribe",
            "temperature": 0.0,  # More deterministic
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.3,  # Lower for radio (often quiet)
            "condition_on_previous_text": False,  # Better for short clips
            "initial_prompt": "Aviation radio communication with callsigns, altitudes, and flight instructions.",
            "word_timestamps": True,
        }

        if options:
            default_options.update(options)

        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_file)
            if audio is None:
                return None

            # Transcribe with GPU
            print(f"🎙️  Transcribing {os.path.basename(audio_file)} on {self.device}...")

            # Clear GPU cache before inference if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Handle transcription based on library type
            if hasattr(self, 'using_faster_whisper') and self.using_faster_whisper:
                # faster-whisper has a different API
                segments, info = self.model.transcribe(
                    audio,
                    language=default_options["language"],
                    temperature=default_options["temperature"],
                    word_timestamps=default_options["word_timestamps"],
                    initial_prompt=default_options["initial_prompt"],
                )

                # Convert to standard whisper format
                result = {
                    "text": " ".join([segment.text for segment in segments]),
                    "segments": [{"text": segment.text, "start": segment.start, "end": segment.end}
                                 for segment in segments],
                    "language": info.language
                }
            else:
                # Standard whisper
                with torch.amp.autocast('cuda', enabled=self.device == "cuda" and WHISPER_COMPUTE_TYPE == "float16"):
                    result = self.model.transcribe(
                        audio,
                        **default_options
                    )

            # Post-process result
            if result and result.get('text'):
                result['text'] = self.post_process_text(result['text'])

                # Add metadata
                result['metadata'] = {
                    'model': self.model_size,
                    'device': self.device,
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'audio_file': audio_file,
                    'timestamp': datetime.now().isoformat()
                }

                processing_time = result['metadata']['processing_time']
                print(f"✅ Transcription complete in {processing_time:.2f}s")

                return result
            else:
                print("❌ No speech detected")
                return None

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def post_process_text(self, text):
        """Post-process text for radio communications"""
        if not text:
            return text

        # Clean up common radio transcription issues
        text = text.strip()

        # Fix common number issues
        text = text.replace(" 0 ", " zero ")
        text = text.replace(" 1 ", " one ")
        text = text.replace(" 2 ", " two ")
        text = text.replace(" 3 ", " three ")
        text = text.replace(" 4 ", " four ")
        text = text.replace(" 5 ", " five ")
        text = text.replace(" 6 ", " six ")
        text = text.replace(" 7 ", " seven ")
        text = text.replace(" 8 ", " eight ")
        text = text.replace(" 9 ", " nine ")

        # Fix common aviation terms
        replacements = {
            "rodger": "roger",
            "wilko": "wilco",
            "altitute": "altitude",
            "cleard": "cleared",
            "cont": "contact",
            "freq": "frequency",
            "twr": "tower",
            "gnd": "ground",
            "app": "approach",
            "dep": "departure"
        }

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        return text

    def batch_transcribe(self, audio_files, callback=None):
        """Batch transcribe multiple files efficiently"""
        results = []

        for i, audio_file in enumerate(audio_files):
            result = self.transcribe_audio(audio_file)
            results.append(result)

            if callback:
                callback(i + 1, len(audio_files), result)

        return results


# Global transcriber instance
_transcriber = None


def get_transcriber():
    """Get or create transcriber instance"""
    global _transcriber
    if _transcriber is None:
        _transcriber = GPUWhisperTranscriber(
            model_size=MODEL_SIZE,
            optimize_for_radio=True
        )
    return _transcriber


def transcribe_audio(audio_file, save_transcript=True):
    """Main transcription function"""
    transcriber = get_transcriber()
    result = transcriber.transcribe_audio(audio_file)

    if result and save_transcript:
        # Save transcript
        if not os.path.exists(TRANSCRIPT_DIR):
            os.makedirs(TRANSCRIPT_DIR)

        base_name = os.path.basename(audio_file).replace('.wav', '')
        transcript_file = os.path.join(TRANSCRIPT_DIR, f"{base_name}_transcript.json")

        with open(transcript_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"📄 Transcript saved: {transcript_file}")

    return result


if __name__ == "__main__":
    # Test transcription
    import sys

    if len(sys.argv) > 1:
        result = transcribe_audio(sys.argv[1])
        if result:
            print(f"\nTranscript: {result['text']}")
            print(f"Processing time: {result['metadata']['processing_time']:.2f}s")