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

from utils.console_logger import info, success, error, ProgressBar, warning
from utils.config import MODEL_SIZE, TRANSCRIPT_DIR, ENABLE_GPU, SAMPLE_RATE, WHISPER_COMPUTE_TYPE, USE_FASTER_WHISPER, \
    PREFER_ONNX_DIRECTML
from utils.gpu_utils import setup_gpu_backend, get_device_string, get_torch_device, get_compute_type, print_gpu_info, \
    get_directml_provider_options, get_amd_gpu_info, TORCH_DIRECTML_AVAILABLE, ONNX_RUNTIME_AVAILABLE


class GPUWhisperTranscriber:
    def __init__(self, model_size="base", device="auto", optimize_for_radio=True):
        self.model_size = model_size
        self.optimize_for_radio = optimize_for_radio

        # Setup GPU backend
        if device == "auto":
            self.preferred_backend, self.capabilities = setup_gpu_backend()
        else:
            self.preferred_backend = self._determine_backend_from_device(device)
            _, self.capabilities = setup_gpu_backend()

        self.device = device
        self.model = None
        self.whisper_type = None
        self.current_backend = None

    def _determine_backend_from_device(self, device):
        """Determine backend from device string"""
        if device.startswith('cuda'):
            return 'cuda'
        elif device == 'directml':
            return 'directml'
        else:
            return 'cpu'

    def load_model(self):
        """
        Load the appropriate Whisper model, with a robust fallback to CPU.
        """
        if self.model is not None:
            return  # Model is already loaded

        # 1. Attempt to load with the preferred (GPU) backend.
        if self.preferred_backend != 'cpu':
            if self._attempt_to_load_model(self.preferred_backend):
                success(f"Model loaded successfully on preferred backend: {self.preferred_backend}")
                print_gpu_info(self.current_backend, self.capabilities)
                return

            warning(f"Preferred backend '{self.preferred_backend}' failed. Falling back to CPU.")

        # 2. If GPU failed or was not preferred, attempt to load with CPU.
        if self._attempt_to_load_model('cpu'):
            success("Model loaded successfully on CPU backend.")
        else:
            # 3. If all attempts fail, the model remains None.
            error("FATAL: All attempts to load the transcription model failed.")
            error("The application cannot proceed with transcription.")

    def _attempt_to_load_model(self, backend):
        """A single attempt to load the model using a specific backend."""
        info(f"Attempting to load model '{self.model_size}' with backend '{backend}'...")
        try:
            # For DirectML, always use faster-whisper with ONNX Runtime
            if backend == 'directml' and USE_FASTER_WHISPER:
                self._load_faster_whisper(backend)
            elif backend == 'cuda' and USE_FASTER_WHISPER:
                self._load_faster_whisper(backend)
            elif backend == 'cpu' and USE_FASTER_WHISPER:
                self._load_faster_whisper(backend)
            else:
                # Standard whisper only for CUDA and CPU (not DirectML)
                if backend != 'directml':
                    self._load_standard_whisper(backend)
                else:
                    raise Exception("Standard whisper does not support DirectML. Enable USE_FASTER_WHISPER in config.")

            if self.model:
                self.current_backend = backend
                if self.optimize_for_radio:
                    self.setup_radio_optimization()
                return True

        except Exception as e:
            error(f"Caught exception while loading with backend '{backend}': {e}")
            self.model = None

        return False

    def _load_faster_whisper(self, backend):
        """Load faster-whisper model for a given backend."""
        from faster_whisper import WhisperModel

        if backend == 'cuda':
            # NVIDIA GPU with CUDA
            compute_type = get_compute_type(backend)
            info(f"Using faster-whisper with CUDA, compute type: {compute_type}", emoji="🚀")

            self.model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type=compute_type
            )

        elif backend == 'directml':
            # AMD GPU with DirectML through ONNX Runtime
            info(f"Using faster-whisper with DirectML (ONNX Runtime)", emoji="🚀")

            # Check if ONNX Runtime with DirectML is available
            if not self.capabilities.get('directml_onnx_available', False):
                raise Exception("ONNX Runtime with DirectML is not available")

            # Configure for DirectML
            selected_device = 0
            if self.capabilities.get('directml_devices', 0) > 1:
                info("Multiple AMD GPUs detected")
                gpu_info = get_amd_gpu_info()
                for i, gpu in enumerate(gpu_info):
                    info(f"  {i}: {gpu['name']}")
                # For automated setup, just use first GPU
                selected_device = 0
                info(f"Using GPU {selected_device}")

            # Set environment variable for DirectML device
            os.environ["DML_DEVICE_ID"] = str(selected_device)

            # Create model with CPU device (faster-whisper will use ONNX Runtime with DirectML internally)
            self.model = WhisperModel(
                self.model_size,
                device="cpu",  # Must be CPU for DirectML
                compute_type="int8",  # DirectML works best with int8
                device_index=0  # This is for CPU, not GPU
            )

            success("faster-whisper loaded with DirectML backend", emoji="✅")

        else:
            # CPU
            compute_type = get_compute_type(backend)
            info(f"Using faster-whisper with CPU, compute type: {compute_type}", emoji="🚀")

            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=compute_type
            )

        self.whisper_type = "faster-whisper"

    def _load_standard_whisper(self, backend):
        """Load standard OpenAI Whisper model for a given backend."""
        if backend == 'cuda':
            device = get_torch_device(backend)
            info(f"Using standard whisper with device: {device}")
            self.model = whisper.load_model(self.model_size, device=device)
        else:
            # CPU
            info("Using standard whisper with CPU")
            self.model = whisper.load_model(self.model_size, device='cpu')

        self.whisper_type = "standard-whisper"

    def setup_radio_optimization(self):
        """Setup audio preprocessing optimizations for radio communications"""
        info("Radio optimization enabled - better performance for ATC audio", emoji="📻")
        self.radio_opts = {'noise_reduction': True, 'frequency_boost': True, 'normalize': True}

    def preprocess_audio(self, audio_path):
        """Preprocess audio for better radio transcription"""
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            if self.optimize_for_radio and hasattr(self, 'radio_opts'):
                if self.radio_opts.get('noise_reduction'):
                    audio = self.reduce_noise(audio, sr)
                if self.radio_opts.get('frequency_boost'):
                    audio = self.radio_filter(audio, sr)
                if self.radio_opts.get('normalize'):
                    audio = librosa.util.normalize(audio)

            # Ensure audio is float32 and contiguous
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            audio = np.ascontiguousarray(audio)

            return audio
        except Exception as e:
            error(f"Error preprocessing audio {audio_path}: {e}")
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            return audio

    def reduce_noise(self, audio, sample_rate):
        """Simple noise reduction using spectral gating"""
        try:
            return librosa.effects.preemphasis(audio)
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
            return signal.filtfilt(b, a, audio)
        except:
            return audio

    def transcribe_audio_with_progress(self, audio_file, options=None):
        """Transcribe audio with progress monitoring"""
        if not self.model:
            self.load_model()

        if not self.model:
            error("Cannot transcribe, model failed to load.")
            return None

        return self.transcribe_audio_internal(audio_file, options)

    def transcribe_audio_internal(self, audio_file, options=None):
        """Internal transcription method"""
        if not self.model:
            error("Transcription failed because the model is not loaded. Aborting.")
            return None

        start_time = datetime.now()
        default_options = {
            "language": "en",
            "task": "transcribe",
            "temperature": 0.0,
            "best_of": 5,
            "beam_size": 5,
            "word_timestamps": True,
            "initial_prompt": "U.S. air traffic control radio communication. All units are in feet, knots, and nautical miles."
        }
        if options:
            default_options.update(options)

        try:
            audio = self.preprocess_audio(audio_file)

            if self.whisper_type == "faster-whisper":
                # faster-whisper has a different API and returns a generator
                segments, _ = self.model.transcribe(
                    audio,
                    language=default_options.get("language", "en"),
                    beam_size=default_options.get("beam_size", 5),
                    word_timestamps=default_options.get("word_timestamps", True),
                    initial_prompt=default_options.get("initial_prompt"),
                    temperature=default_options.get("temperature", 0.0)
                )

                # Reconstruct the result object to match the standard whisper format
                text_segments = [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments]
                text = ' '.join([seg['text'].strip() for seg in text_segments])

            else:
                # Standard whisper
                result = self.model.transcribe(audio, **default_options)
                text = result.get("text", "")
                text_segments = result.get("segments", [])

            processing_time = (datetime.now() - start_time).total_seconds()
            text = self.post_process_text(text)

            result_payload = {
                "text": text, "segments": text_segments,
                "metadata": {
                    "model_size": self.model_size, "backend": self.current_backend,
                    "whisper_type": self.whisper_type, "processing_time": processing_time,
                    "audio_file": os.path.basename(audio_file), "timestamp": datetime.now().isoformat(),
                    "options": default_options
                }
            }

            # Clear GPU memory if using GPU
            if self.current_backend == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return result_payload

        except Exception as e:
            error(f"An exception occurred during transcription: {e}")
            import traceback
            traceback.print_exc()
            return None

    def post_process_text(self, text):
        """Post-process text for radio communications"""
        if not text:
            return ""
        # Your text replacement logic can stay here
        return text.strip()


# Global transcriber instance management
_transcriber = None


def get_transcriber():
    """Get or create the global transcriber instance"""
    global _transcriber
    if _transcriber is None:
        _transcriber = GPUWhisperTranscriber(
            model_size=MODEL_SIZE,
            device="auto",
            optimize_for_radio=True
        )
    return _transcriber


def transcribe_audio(audio_file, save_transcript=True, show_progress=True):
    """Main transcription function"""
    transcriber = get_transcriber()
    if not transcriber.model:
        transcriber.load_model()

    if not transcriber.model:
        error(f"Skipping transcription for {audio_file} because model is not available.")
        return None

    # Simplified call for clarity
    result = transcriber.transcribe_audio_internal(audio_file)

    if result and save_transcript:
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
        base_name = os.path.basename(audio_file).replace('.wav', '')
        transcript_file = os.path.join(TRANSCRIPT_DIR, f"{base_name}_transcript.json")
        with open(transcript_file, 'w') as f:
            json.dump(result, f, indent=2)
        info(f"Transcript saved: {transcript_file}")

    return result