# config.py
# Configuration settings for ATC WSPR Multi-Channel Monitor
#
# ⚠️  IMPORTANT: For GPU worker configuration help, see:
#     GPU_WORKER_CONFIGURATION_GUIDE.md (in project root)
#
# Quick guide to worker configuration:
#   - 4GB GPU:  MODEL_SIZE = "small", NUM_TRANSCRIPTION_WORKERS = 1
#   - 8GB GPU:  MODEL_SIZE = "small", NUM_TRANSCRIPTION_WORKERS = 3
#   - 12GB GPU: MODEL_SIZE = "medium", NUM_TRANSCRIPTION_WORKERS = 2
#   - 16GB GPU: MODEL_SIZE = "medium", NUM_TRANSCRIPTION_WORKERS = 3
#   - 24GB GPU: MODEL_SIZE = "large", NUM_TRANSCRIPTION_WORKERS = 2
#
MODEL_SIZE = "distil-large-v3"  # Options: tiny, base, small, medium, large
PROCESSED_DIR = "audio/processed/"
TRANSCRIPT_DIR = "transcripts/"
ANALYSIS_DIR = "analysis/"
ENABLE_GPU = True

# GPU Configuration
GPU_BACKEND = "auto"  # Options: "auto", "cuda", "directml", "cpu"
PREFER_AMD_GPU = False  # Set to True to prefer AMD over NVIDIA when both available
DIRECTML_ENABLED = True  # Enable DirectML support for AMD GPUs
PREFER_ONNX_DIRECTML = True  # Prefer ONNX Runtime DirectML over PyTorch DirectML

# Performance tuning
USE_FASTER_WHISPER = True  # Use faster-whisper library
WHISPER_COMPUTE_TYPE = "float16"  # float16 for GPU, int8 for CPU/DirectML
BATCH_SIZE = 1  # For batch processing

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-CHANNEL TRANSCRIPTION WORKER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
# Number of parallel GPU transcription workers for multi-channel monitoring
# Each worker loads a separate model instance into GPU VRAM
#
# IMPORTANT: Set this based on your GPU's VRAM capacity and model size!
# 
# VRAM REQUIREMENTS PER MODEL (Approximate):
# ┌────────────┬──────────────┬──────────────┬──────────────────────────────┐
# │ Model Size │ VRAM (CUDA)  │ VRAM (DirectML) │ Quality                    │
# ├────────────┼──────────────┼──────────────┼──────────────────────────────┤
# │ tiny       │ ~1 GB        │ ~1.2 GB      │ Basic (fast, less accurate) │
# │ base       │ ~1.5 GB      │ ~1.8 GB      │ Good (balanced)             │
# │ small      │ ~2 GB        │ ~2.5 GB      │ Better (recommended min)    │
# │ medium     │ ~5 GB        │ ~6 GB        │ Great (good accuracy)       │
# │ large      │ ~10 GB       │ ~12 GB       │ Best (highest accuracy)     │
# │ large-v2   │ ~10 GB       │ ~12 GB       │ Best (latest)               │
# │ large-v3   │ ~10 GB       │ ~12 GB       │ Best (newest)               │
# └────────────┴──────────────┴──────────────┴──────────────────────────────┘
#
# WORKER RECOMMENDATIONS BY GPU VRAM:
# ┌──────────────┬─────────────────────────────────────────────────────────┐
# │ GPU VRAM     │ Recommended Configuration                               │
# ├──────────────┼─────────────────────────────────────────────────────────┤
# │ 4 GB         │ tiny: 3 workers, base: 2 workers, small: 1 worker       │
# │ 6 GB         │ tiny: 5 workers, base: 3 workers, small: 2 workers      │
# │              │ medium: 1 worker                                        │
# │ 8 GB         │ tiny: 7 workers, base: 5 workers, small: 3 workers      │
# │              │ medium: 1 worker                                        │
# │ 12 GB        │ tiny: 10 workers, base: 7 workers, small: 5 workers     │
# │              │ medium: 2 workers, large: 1 worker                      │
# │ 16 GB        │ small: 6 workers, medium: 3 workers, large: 1 worker    │
# │ 24 GB        │ small: 9 workers, medium: 4 workers, large: 2 workers   │
# │ 32 GB+       │ medium: 6 workers, large: 3 workers                     │
# └──────────────┴─────────────────────────────────────────────────────────┘
#
# EXAMPLES:
#   RTX 3060 (12GB):    NUM_TRANSCRIPTION_WORKERS = 1 with large model
#                       NUM_TRANSCRIPTION_WORKERS = 2 with medium model
#                       NUM_TRANSCRIPTION_WORKERS = 5 with small model
#
#   RTX 3080 (10GB):    NUM_TRANSCRIPTION_WORKERS = 1 with large model
#                       NUM_TRANSCRIPTION_WORKERS = 1 with medium model
#                       NUM_TRANSCRIPTION_WORKERS = 4 with small model
#
#   RTX 4090 (24GB):    NUM_TRANSCRIPTION_WORKERS = 2 with large model
#                       NUM_TRANSCRIPTION_WORKERS = 4 with medium model
#                       NUM_TRANSCRIPTION_WORKERS = 9 with small model
#
#   AMD RX 6800 (16GB): NUM_TRANSCRIPTION_WORKERS = 1 with large model
#                       NUM_TRANSCRIPTION_WORKERS = 3 with medium model
#                       NUM_TRANSCRIPTION_WORKERS = 6 with small model
#
#   AMD RX 7900 XTX (24GB): NUM_TRANSCRIPTION_WORKERS = 2 with large model
#                           NUM_TRANSCRIPTION_WORKERS = 4 with medium model
#
# NOTES:
# - Leave ~1-2GB VRAM headroom for system and display
# - DirectML (AMD) typically uses ~20% more VRAM than CUDA (NVIDIA)
# - More workers = faster transcription but uses more VRAM
# - If you get OOM errors, reduce NUM_TRANSCRIPTION_WORKERS or use smaller model
# - For testing, start with 1 worker and increase gradually
# - CPU mode: Can use more workers (4-8) but transcription is much slower
#
NUM_TRANSCRIPTION_WORKERS = 5  # Default: 3 workers (suitable for 8-16GB VRAM with small/medium model)

# Audio preprocessing
OPTIMIZE_FOR_RADIO = True
APPLY_NOISE_REDUCTION = True
RADIO_FREQUENCY_BOOST = True

# Audio settings
SAMPLE_RATE = 16000  # Good for speech recognition
CHANNELS = 1  # Mono for ATC
AUDIO_DIR = "audio/raw/"

# VAD settings
VAD_THRESHOLD = 0.1  # Adjust based on your audio levels
SILENCE_DURATION = 3.0  # Seconds of silence before ending recording
MIN_TRANSMISSION_LENGTH = 1.0  # Minimum seconds to save a transmission

# Airport/Monitoring Area Configuration
LOCATION_NAME = "Default"
AIRPORT_LAT = 44.8851  # Your airport latitude (45.588699 for pdx, 44.8851 for msp)
AIRPORT_LON = -93.2144  # Your airport longitude (-122.5975 for , -93.2144 for msp)
SEARCH_RADIUS_NM = 50  # Nautical miles radius to monitor


def apply_location_settings(location: dict) -> None:
    """Apply location-specific settings from a configuration dictionary."""
    global LOCATION_NAME, AIRPORT_LAT, AIRPORT_LON, SEARCH_RADIUS_NM

    airport = location.get("airport", {}) if isinstance(location, dict) else {}
    if "name" in location:
        LOCATION_NAME = location["name"]
    if "lat" in airport:
        AIRPORT_LAT = airport["lat"]
    if "lon" in airport:
        AIRPORT_LON = airport["lon"]
    if "search_radius_nm" in location:
        SEARCH_RADIUS_NM = location["search_radius_nm"]

# ADS-B Configuration
ENABLE_ADSB = True
ADSB_SOURCE = 'opensky'  # Options: 'opensky', 'adsbx', 'local'

# OpenSky Network OAuth2 credentials
# Path to JSON file containing `client_id` and `client_secret`
OPENSKY_CREDENTIALS_FILE = 'credentials.json'

# ADS-B Exchange (optional, requires API key)
ADSBX_API_KEY = ''

# Local dump1090 (if running RTL-SDR receiver)
DUMP1090_URL = 'http://localhost:8080'

# Correlation Settings
CORRELATION_WINDOW = 30  # seconds
ALTITUDE_TOLERANCE = 500  # feet
POSITION_TOLERANCE = 5  # nautical miles

# LLM Correlator Settings
ENABLE_LLM_CORRELATION = True
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://localhost:11434"
# Timeout for Ollama correlation requests (kept higher for large batches)
OLLAMA_REQUEST_TIMEOUT = 220  # seconds
LLM_MAX_ADSB_CONTACTS = 100
LLM_MAX_TRANSMISSIONS = 15

# Advanced Ollama correlator tuning (these were previously hardcoded)
# Maximum tokens supported by the Ollama model context window
OLLAMA_CONTEXT_WINDOW_TOKENS = 12400
# Tokens reserved for the model's response to avoid truncation
OLLAMA_MAX_RESPONSE_TOKENS = 6400
# Conservative character-to-token ratio for prompt budgeting
OLLAMA_CHARS_PER_TOKEN = 4.0
# Extra buffer added to token estimates for safety
OLLAMA_TOKEN_ESTIMATE_BUFFER = 20
# Estimated tokens consumed per correlation entry in the response JSON
OLLAMA_TOKENS_PER_CORRELATION = 180
# Tokens held back for JSON wrappers and metadata in the response
OLLAMA_RESPONSE_JSON_OVERHEAD = 200
# Maximum transmissions to include per request (prevents oversized responses)
OLLAMA_MAX_TRANSMISSION_BATCH = 10
# Portion of prompt token budget dedicated to ADS-B contact context
OLLAMA_ADSB_PROMPT_RATIO = 0.70
# Character limit for previewing transmission text inside prompts
OLLAMA_TRANSMISSION_PREVIEW_CHARS = 150
# Temperature used for Ollama generation (lower for deterministic output)
OLLAMA_TEMPERATURE = 0.3
# Top-p nucleus sampling value for Ollama generation
OLLAMA_TOP_P = 0.9
# Repeat penalty applied to discourage looping responses
OLLAMA_REPEAT_PENALTY = 1.1
# Number of responses tracked when computing moving average latency
OLLAMA_RESPONSE_TIME_WINDOW = 100
# Safety buffer before assuming the response hit the token ceiling
OLLAMA_RESPONSE_SAFETY_MARGIN = 50
# Minimum alert confidence the LLM must report before surfacing
OLLAMA_ALERT_CONFIDENCE_THRESHOLD = 0.7
# Delay used when bringing up the debug monitor to ensure Tk is ready
OLLAMA_DEBUG_MONITOR_DELAY = 0.5
# Enable the rich Tk debug monitor when Tkinter is available
OLLAMA_ENABLE_DEBUG_MONITOR = True
