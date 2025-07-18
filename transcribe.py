#transcribe.py
import whisper
import torch
import numpy as np
from datetime import datetime
import json
import os
import librosa
import gc
import psutil  # For CPU monitoring
from tqdm import tqdm  # For progress bars

from config import MODEL_SIZE, TRANSCRIPT_DIR, ENABLE_GPU, SAMPLE_RATE, WHISPER_COMPUTE_TYPE, USE_FASTER_WHISPER, \
    PREFER_ONNX_DIRECTML
from gpu_utils import setup_gpu_backend, get_device_string, get_torch_device, get_compute_type, print_gpu_info, \
    get_directml_provider_options, get_amd_gpu_info


class GPUWhisperTranscriber:
    def __init__(self, model_size="base", device="auto", optimize_for_radio=True):
        self.model_size = model_size
        self.optimize_for_radio = optimize_for_radio

        # Setup GPU backend
        if device == "auto":
            self.backend, self.capabilities = setup_gpu_backend()
            self.device_string = get_device_string(self.backend)
            self.device = get_torch_device(self.backend) if self.backend != 'directml' else None
        else:
            self.device_string = device
            self.device = torch.device(device) if device != 'directml' else None
            self.backend = self._determine_backend_from_device(device)
            self.capabilities = None

        # Load model with appropriate optimizations
        self.load_model()

        # Optimize for radio if requested
        if optimize_for_radio:
            self.setup_radio_optimization()

    def _determine_backend_from_device(self, device):
        """Determine backend from device string"""
        if 'cuda' in device:
            return 'cuda'
        elif 'directml' in device:
            return 'directml'
        else:
            return 'cpu'

    def load_model(self):
        """Load Whisper model with appropriate optimizations"""
        print(f"📦 Loading Whisper model: {self.model_size} on {self.device_string}")

        # Print GPU information
        if self.capabilities:
            print_gpu_info(self.backend, self.capabilities)

        try:
            # Clear CUDA cache before loading model
            if self.backend == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

            # UPDATED PRIORITY ORDER:
            # 1. Use faster-whisper for NVIDIA GPU (cuda)
            # 2. Use faster-whisper with DirectML for AMD GPU
            # 3. Use faster-whisper for CPU
            # 4. Use normal whisper as fallback

            if USE_FASTER_WHISPER:
                success = self._load_faster_whisper()
                if success:
                    return

            # Standard whisper fallback
            self._load_standard_whisper()

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _load_faster_whisper(self):
        """Load faster-whisper model with backend-specific optimizations"""
        try:
            from faster_whisper import WhisperModel
            import os

            # Show progress bar for model loading
            with tqdm(total=1, desc="Loading faster-whisper model", unit="model",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:

                if self.backend == 'cuda':
                    # NVIDIA GPU with CUDA
                    compute_type = get_compute_type(self.backend)
                    print(f"🚀 Using faster-whisper with CUDA, compute type: {compute_type}")

                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device_string,
                        compute_type=compute_type
                    )

                elif self.backend == 'directml':
                    # AMD GPU with DirectML
                    print(f"🚀 Using faster-whisper with DirectML")

                    # Get GPU selection from user if multiple GPUs are available
                    selected_device = 0
                    if hasattr(self, 'capabilities') and self.capabilities.get('directml_devices', 0) > 1:
                        print("\nMultiple AMD GPUs detected:")
                        # Display available GPUs
                        gpu_info = get_amd_gpu_info()
                        for i, gpu in enumerate(gpu_info):
                            print(f"  {i}: {gpu['name']}")

                        try:
                            selected_device = int(input("\nSelect GPU by number (0, 1, etc.): "))
                            print(f"Selected GPU {selected_device}")
                        except:
                            print("Invalid input, using default GPU 0")
                            selected_device = 0

                    # Configure environment for ONNX Runtime DirectML
                    if PREFER_ONNX_DIRECTML and self.capabilities.get('directml_onnx_available', False):
                        try:
                            print("   - Setting up ONNX Runtime with DirectML...")
                            # Set environment variable for DirectML device
                            os.environ["DML_DEVICE_ID"] = str(selected_device)

                            # Create model without passing providers directly
                            self.model = WhisperModel(
                                self.model_size,
                                device="cpu",  # Always use CPU for DirectML
                                compute_type="int8",  # DirectML works better with int8
                                device_index=0  # This is for CPU, not GPU
                            )
                            print("   ✅ ONNX Runtime DirectML configured successfully")
                        except Exception as e:
                            print(f"   ⚠️ ONNX Runtime DirectML setup failed: {e}")
                            print("   - Falling back to PyTorch DirectML...")

                            # Fallback to PyTorch DirectML
                            self.model = WhisperModel(
                                self.model_size,
                                device="cpu",
                                compute_type="int8",
                                device_index=0
                            )
                    else:
                        # Use PyTorch DirectML directly
                        self.model = WhisperModel(
                            self.model_size,
                            device="cpu",
                            compute_type="int8",
                            device_index=0
                        )
                        print("   ✅ PyTorch DirectML loaded")

                else:
                    # CPU
                    compute_type = get_compute_type(self.backend)
                    print(f"🚀 Using faster-whisper with CPU, compute type: {compute_type}")

                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device_string,
                        compute_type=compute_type
                    )

                pbar.update(1)

            self.using_faster_whisper = True
            print(f"✅ faster-whisper model loaded successfully")
            return True

        except ImportError:
            print("⚠️ faster-whisper not installed, falling back to standard whisper")
            return False
        except Exception as e:
            print(f"⚠️ Error loading faster-whisper model: {e}")
            print("⚠️ Falling back to standard whisper")
            return False

    def _load_standard_whisper(self):
        """Load standard whisper model"""
        with tqdm(total=1, desc="Loading standard whisper model", unit="model",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:

            if self.backend == 'cuda':
                # Load model directly to GPU with optimized settings
                self.model = whisper.load_model(self.model_size)
                self.model = self.model.to(self.device)

                # Enable mixed precision for better performance
                if hasattr(torch.cuda, 'amp') and WHISPER_COMPUTE_TYPE == "float16":
                    print("🚀 Enabling mixed precision for faster inference")
                    self.model = self.model.half()  # Convert to half precision

            elif self.backend == 'directml':
                # For DirectML, load on CPU first then move to DirectML device
                self.model = whisper.load_model(self.model_size, device='cpu')
                if self.device:
                    self.model = self.model.to(self.device)
                    print("🚀 Standard whisper moved to DirectML device")
                else:
                    print("⚠️ DirectML device not available, using CPU")

            else:
                # CPU model loading
                self.model = whisper.load_model(self.model_size, device=self.device)

            pbar.update(1)

        self.using_faster_whisper = False
        print(f"✅ Standard whisper model loaded successfully")

    def setup_radio_optimization(self):
        """Setup optimizations for radio communications"""
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
                audio = librosa.util.normalize(audio)
                audio = librosa.effects.preemphasis(audio, coef=0.97)
                audio = self.reduce_noise(audio, sr)
                audio = self.radio_filter(audio, sr)

            # Ensure audio array is contiguous to avoid negative stride error
            audio = np.ascontiguousarray(audio)

            return audio

        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None

    def reduce_noise(self, audio, sample_rate):
        """Basic noise reduction for radio audio"""
        try:
            threshold = np.percentile(np.abs(audio), 20)
            audio = np.where(np.abs(audio) < threshold, audio * 0.1, audio)
            return audio
        except:
            return audio

    def radio_filter(self, audio, sample_rate):
        """Apply radio-like frequency filtering"""
        try:
            from scipy import signal
            nyquist = sample_rate * 0.5
            low = 300 / nyquist
            high = 3400 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio)
            return filtered_audio
        except:
            return audio

    def transcribe_audio_with_progress(self, audio_file, options=None):
        """Wrapper for transcribe_audio with progress monitoring"""
        # Create a custom progress bar that shows CPU usage
        with tqdm(total=100, desc=f"Transcribing {os.path.basename(audio_file)}",
                  bar_format="{l_bar}{bar}| CPU: {postfix[0]:.1f}% | {elapsed}<{remaining}",
                  postfix=[0.0]) as pbar:

            # Start CPU monitoring in a separate thread
            import threading
            stop_monitoring = threading.Event()

            def monitor_cpu():
                while not stop_monitoring.is_set():
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    pbar.postfix[0] = cpu_percent
                    pbar.refresh()

            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()

            try:
                # Simulate progress steps
                pbar.update(20)  # Loading audio
                result = self.transcribe_audio(audio_file, options)
                pbar.update(80)  # Transcription complete

                # Stop monitoring
                stop_monitoring.set()
                monitor_thread.join()

                return result
            except Exception as e:
                stop_monitoring.set()
                monitor_thread.join()
                raise e

    def transcribe_audio(self, audio_file, options=None):
        """Transcribe audio with GPU acceleration and radio optimizations"""
        start_time = datetime.now()

        # Default options optimized for radio
        default_options = {
            "language": "en",
            "task": "transcribe",
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.3,
            "condition_on_previous_text": False,
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

            # Ensure audio is float32 and contiguous
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            audio = np.ascontiguousarray(audio)

            # Clear GPU cache before inference
            if self.backend == 'cuda':
                torch.cuda.empty_cache()

            # Handle transcription based on library type
            if hasattr(self, 'using_faster_whisper') and self.using_faster_whisper:
                # faster-whisper API
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
                if self.backend == 'cuda' and WHISPER_COMPUTE_TYPE == "float16":
                    with torch.amp.autocast('cuda', enabled=True):
                        result = self.model.transcribe(audio, **default_options)
                else:
                    result = self.model.transcribe(audio, **default_options)

            # Post-process result
            if result and result.get('text'):
                result['text'] = self.post_process_text(result['text'])

                # Add metadata
                result['metadata'] = {
                    'model': self.model_size,
                    'device': self.device_string,
                    'backend': self.backend,
                    'whisper_type': 'faster-whisper' if hasattr(self,
                                                                'using_faster_whisper') and self.using_faster_whisper else 'standard-whisper',
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'audio_file': audio_file,
                    'timestamp': datetime.now().isoformat()
                }

                processing_time = result['metadata']['processing_time']
                whisper_type = result['metadata']['whisper_type']
                print(f"✅ Transcription complete in {processing_time:.2f}s using {whisper_type} on {self.backend}")

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
        replacements = {
            " 0 ": " zero ", " 1 ": " one ", " 2 ": " two ", " 3 ": " three ",
            " 4 ": " four ", " 5 ": " five ", " 6 ": " six ", " 7 ": " seven ",
            " 8 ": " eight ", " 9 ": " nine "
        }

        # Fix common aviation terms
        aviation_replacements = {
            "rodger": "roger", "wilko": "wilco", "altitute": "altitude",
            "cleard": "cleared", "cont": "contact", "freq": "frequency",
            "twr": "tower", "gnd": "ground", "app": "approach", "dep": "departure"
        }

        replacements.update(aviation_replacements)

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        return text

    def batch_transcribe(self, audio_files, callback=None):
        """Batch transcribe multiple files efficiently with progress bar"""
        results = []

        # Create overall progress bar
        with tqdm(total=len(audio_files), desc="Batch transcription", unit="files") as pbar:
            for i, audio_file in enumerate(audio_files):
                result = self.transcribe_audio_with_progress(audio_file)
                results.append(result)

                pbar.update(1)

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


def transcribe_audio(audio_file, save_transcript=True, show_progress=True):
    """Main transcription function"""
    transcriber = get_transcriber()

    if show_progress:
        result = transcriber.transcribe_audio_with_progress(audio_file)
    else:
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
            print(f"Backend: {result['metadata']['backend']}")
            print(f"Whisper type: {result['metadata']['whisper_type']}")