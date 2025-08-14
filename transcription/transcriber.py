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
import re

# NEW: Added imports for more advanced audio processing
try:
    import noisereduce as nr
    from scipy import signal

    ADVANCED_AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_AUDIO_PROCESSING_AVAILABLE = False


from utils.console_logger import info, success, error, ProgressBar, warning
from utils.config import MODEL_SIZE, TRANSCRIPT_DIR, ENABLE_GPU, SAMPLE_RATE, WHISPER_COMPUTE_TYPE, USE_FASTER_WHISPER, \
    PREFER_ONNX_DIRECTML
from utils.gpu_utils import setup_gpu_backend, get_device_string, get_torch_device, get_compute_type, print_gpu_info, \
    get_directml_provider_options, get_amd_gpu_info, TORCH_DIRECTML_AVAILABLE, ONNX_RUNTIME_AVAILABLE


RADIO_INITIAL_PROMPT = (
    "U.S. air traffic control radio communication. All units are in feet, knots, and nautical miles."
)


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
                # MODIFIED: setup_radio_optimization is now called here
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
            compute_type = get_compute_type(backend)
            info(f"Using faster-whisper with CUDA, compute type: {compute_type}", emoji="🚀")
            self.model = WhisperModel(self.model_size, device="cuda", compute_type=compute_type)

        elif backend == 'directml':
            info(f"Using faster-whisper with DirectML (ONNX Runtime)", emoji="🚀")
            if not self.capabilities.get('directml_onnx_available', False):
                raise Exception("ONNX Runtime with DirectML is not available")

            selected_device = 0
            if self.capabilities.get('directml_devices', 0) > 1:
                info("Multiple AMD GPUs detected, using device 0.")
            os.environ["DML_DEVICE_ID"] = str(selected_device)

            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            success("faster-whisper loaded with DirectML backend", emoji="✅")

        else:  # CPU
            compute_type = get_compute_type(backend)
            info(f"Using faster-whisper with CPU, compute type: {compute_type}", emoji="🚀")
            self.model = WhisperModel(self.model_size, device="cpu", compute_type=compute_type)

        self.whisper_type = "faster-whisper"

    def _load_standard_whisper(self, backend):
        """Load standard OpenAI Whisper model for a given backend."""
        if backend == 'cuda':
            device = get_torch_device(backend)
            info(f"Using standard whisper with device: {device}")
            self.model = whisper.load_model(self.model_size, device=device)
        else:
            info("Using standard whisper with CPU")
            self.model = whisper.load_model(self.model_size, device='cpu')

        self.whisper_type = "standard-whisper"

    # MODIFIED: Updated setup to reflect new capabilities
    def setup_radio_optimization(self):
        """Setup audio preprocessing optimizations for radio communications"""
        info("Radio optimization enabled - using advanced audio filtering for ATC.", emoji="📻")
        if not ADVANCED_AUDIO_PROCESSING_AVAILABLE:
            warning("`noisereduce` or `scipy` not found. Audio filtering will be limited.")
            warning("For best results, run: pip install noisereduce scipy")

    # MODIFIED: Replaced with a multi-stage, more effective pipeline
    def preprocess_audio(self, audio_path):
        """Preprocess audio for better radio transcription using a multi-stage pipeline."""
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

            if self.optimize_for_radio:
                # 1. High-pass filter to remove low-frequency rumble
                audio = self.high_pass_filter(audio, cutoff_freq=250, sample_rate=sr)

                # 2. Spectral noise reduction to remove static and background noise
                audio = self.reduce_noise_spectral(audio, sample_rate=sr)

                # 3. Band-pass filter to isolate the voice frequency range
                audio = self.radio_filter(audio, sample_rate=sr)

                # 4. Normalize audio volume
                audio = librosa.util.normalize(audio)

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            return np.ascontiguousarray(audio)

        except Exception as e:
            error(f"Error preprocessing audio {audio_path}: {e}")
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            return audio

    # NEW: Helper function for spectral noise reduction
    def reduce_noise_spectral(self, audio, sample_rate):
        """Reduce noise using spectral gating. Requires `noisereduce`."""
        if not ADVANCED_AUDIO_PROCESSING_AVAILABLE:
            return audio
        try:
            return nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.95, stationary=False)
        except Exception as e:
            warning(f"Could not apply spectral noise reduction: {e}")
            return audio

    # NEW: Helper function for high-pass filtering
    def high_pass_filter(self, audio, cutoff_freq, sample_rate):
        """Apply a high-pass filter to remove low-frequency noise."""
        if not ADVANCED_AUDIO_PROCESSING_AVAILABLE:
            return audio
        try:
            nyquist = 0.5 * sample_rate
            norm_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(6, norm_cutoff, btype='high', analog=False)
            return signal.filtfilt(b, a, audio)
        except Exception as e:
            warning(f"Could not apply high-pass filter: {e}")
            return audio

    # MODIFIED: Updated band-pass filter with slightly different parameters
    def radio_filter(self, audio, sample_rate):
        """Apply a band-pass filter to isolate voice frequencies."""
        if not ADVANCED_AUDIO_PROCESSING_AVAILABLE:
            return audio
        try:
            lowcut = 250.0
            highcut = 3800.0
            nyquist = 0.5 * sample_rate
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(6, [low, high], btype='band')
            return signal.filtfilt(b, a, audio)
        except Exception as e:
            warning(f"Could not apply radio band-pass filter: {e}")
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

        # MODIFIED: Greatly enhanced initial prompt and tuned parameters for accuracy
        RADIO_INITIAL_PROMPT = (
            "U.S. air traffic control radio communication. This transcript contains standard ATC phraseology, "
            "aircraft callsigns (e.g., 'November one two three alpha bravo', 'United one two three'), and the phonetic alphabet "
            "(Alpha, Bravo, Charlie, Delta, Echo, Foxtrot, Golf, Hotel, India, Juliett, Kilo, Lima, Mike, November, Oscar, "
            "Papa, Quebec, Romeo, Sierra, Tango, Uniform, Victor, Whiskey, X-ray, Yankee, Zulu). "
            "Numbers are spoken digit by digit, such as 'one two one point five' for 121.5, or 'niner' for the digit 9. "
            "Common terms include: squawk, runway, taxiway, takeoff, landing, wind, traffic, roger, wilco, approach, departure, "
            "vector, hold short, descend, climb, maintain, flight level, heading, cleared, contact, tower, ground. "
            "Altitudes are in feet, speeds in knots, and distances in nautical miles."
        )

        # MODIFIED: Tuned parameters for higher accuracy on noisy, specific-domain audio
        transcribe_options = {
            "language": "en",
            "task": "transcribe",
            "temperature": 0.0,
            "beam_size": 7,  # Increased from 5 for more thorough searching
            "best_of": 7,  # Match beam_size when temp is 0
            "patience": 1.0,  # Added to help with challenging audio
            "suppress_tokens": [-1],  # Suppress tokens that are often errors/hallucinations
            "log_prob_threshold": -0.8,  # Filter out low-confidence (likely garbage) segments
            "no_speech_threshold": 0.4,  # More sensitive to faint speech than the default (0.6)
            "word_timestamps": True,
            "initial_prompt": RADIO_INITIAL_PROMPT,
        }
        if options:
            transcribe_options.update(options)

        try:
            audio = self.preprocess_audio(audio_file)

            if self.whisper_type == "faster-whisper":
                # faster-whisper uses log_prob_threshold, not logprob_threshold
                segments, _ = self.model.transcribe(
                    audio,
                    language=transcribe_options["language"],
                    task=transcribe_options["task"],
                    beam_size=transcribe_options["beam_size"],
                    best_of=transcribe_options["best_of"],
                    patience=transcribe_options["patience"],
                    temperature=transcribe_options["temperature"],
                    initial_prompt=transcribe_options["initial_prompt"],
                    log_prob_threshold=transcribe_options["log_prob_threshold"],
                    no_speech_threshold=transcribe_options["no_speech_threshold"],
                    word_timestamps=transcribe_options["word_timestamps"],
                    suppress_tokens=transcribe_options["suppress_tokens"]
                )

                # Reconstruct the result object to match the standard whisper format
                text_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
                text_segments = self._clean_segments(text_segments)
                text = ' '.join([seg['text'].strip() for seg in text_segments])
                
            else:
                # Standard whisper uses logprob_threshold (no underscore)
                standard_whisper_options = transcribe_options.copy()
                standard_whisper_options["logprob_threshold"] = standard_whisper_options.pop("log_prob_threshold")
                result = self.model.transcribe(audio, **standard_whisper_options)
                text = result.get("text", "")
                text_segments = self._clean_segments(result.get("segments", []))

            processing_time = (datetime.now() - start_time).total_seconds()
            text = self.post_process_text(text)

            result_payload = {
                "text": text, "segments": text_segments,
                "metadata": {
                    "model_size": self.model_size, "backend": self.current_backend,
                    "whisper_type": self.whisper_type, "processing_time": processing_time,
                    "audio_file": os.path.basename(audio_file), "timestamp": datetime.now().isoformat(),
                    "options": transcribe_options
                }
            }

            if self.current_backend == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return result_payload

        except Exception as e:
            error(f"An exception occurred during transcription: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _clean_segments(self, segments):
        """Remove boilerplate prompts and collapse repeated segments."""
        cleaned = []
        prev_text = None
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text or text == RADIO_INITIAL_PROMPT:
                continue
            if prev_text is None or text.lower() != prev_text.lower():
                cleaned.append({**seg, "text": text})
                prev_text = text
        return cleaned

    def post_process_text(self, text):
        """Post-process text for radio communications"""
        if not text:
            return ""

        text = text.replace(RADIO_INITIAL_PROMPT, "")
        text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

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

    result = transcriber.transcribe_audio_internal(audio_file)

    if result and save_transcript:
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
        base_name = os.path.basename(audio_file).replace('.wav', '')
        transcript_file = os.path.join(TRANSCRIPT_DIR, f"{base_name}_transcript.json")
        with open(transcript_file, 'w') as f:
            json.dump(result, f, indent=2)
        info(f"Transcript saved: {transcript_file}")

    return result
