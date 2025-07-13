# Configuration settings
MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large
PROCESSED_DIR = "audio/processed/"
TRANSCRIPT_DIR = "transcripts/"
ANALYSIS_DIR = "analysis/"

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