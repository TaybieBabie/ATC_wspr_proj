# Configuration settings
MODEL_SIZE = "large"  # Options: tiny, base, small, medium, large
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

# Audio preprocessing
OPTIMIZE_FOR_RADIO = True
APPLY_NOISE_REDUCTION = True
RADIO_FREQUENCY_BOOST = True

# Audio settings
SAMPLE_RATE = 16000  # Good for speech recognition
CHANNELS = 1  # Mono for ATC
AUDIO_DIR = "audio/raw/"

# ATC settings
ATC_FREQUENCY = "121.9"  # Your local frequency

# VAD settings
VAD_THRESHOLD = 0.2  # Adjust based on your audio levels
SILENCE_DURATION = 3.0  # Seconds of silence before ending recording
MIN_TRANSMISSION_LENGTH = 1.0  # Minimum seconds to save a transmission

# LiveATC settings
LIVEATC_STREAM_URL = "https://s1-bos.liveatc.net/kmsp3_twr_12r?nocache=2025071302515286063"  # Replace with your local feed

# Airport/Monitoring Area Configuration
AIRPORT_LAT = 40.6413  # Your airport latitude (e.g., JFK)
AIRPORT_LON = -73.7781  # Your airport longitude
SEARCH_RADIUS_NM = 30  # Nautical miles radius to monitor

# ADS-B Configuration
ENABLE_ADSB = True
ADSB_SOURCE = 'opensky'  # Options: 'opensky', 'adsbx', 'local'

# OpenSky Network (free, requires account for better rate limits)
OPENSKY_USERNAME = ''  # Leave empty for anonymous access
OPENSKY_PASSWORD = ''

# ADS-B Exchange (optional, requires API key)
ADSBX_API_KEY = ''

# Local dump1090 (if running RTL-SDR receiver)
DUMP1090_URL = 'http://localhost:8080'

# Correlation Settings
CORRELATION_WINDOW = 30  # seconds
ALTITUDE_TOLERANCE = 500  # feet
POSITION_TOLERANCE = 5  # nautical miles