import pyaudio
import wave
import datetime
import os
import threading
import queue
import numpy as np
import requests
import subprocess
import tempfile
from config import SAMPLE_RATE, CHANNELS, AUDIO_DIR


class LiveATCRecorder:
    def __init__(self, stream_url, vad_threshold=0.01, silence_duration=2.0):
        self.stream_url = stream_url
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.current_audio_buffer = []
        self.silence_counter = 0
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS

        if not os.path.exists(AUDIO_DIR):
            os.makedirs(AUDIO_DIR)

    def capture_stream_audio(self):
        """Capture audio from LiveATC stream using ffmpeg"""
        cmd = [
            'ffmpeg',
            '-i', self.stream_url,
            '-f', 'wav',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-'  # Output to stdout
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1024
            )
            return self.ffmpeg_process.stdout
        except Exception as e:
            print(f"Error starting stream capture: {e}")
            return None

    def detect_voice_activity(self, audio_data):
        """Simple VAD based on RMS energy"""
        if len(audio_data) == 0:
            return False

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

        # Normalize RMS (adjust based on your audio levels)
        normalized_rms = rms / 32768.0

        return normalized_rms > self.vad_threshold

    def save_audio_segment(self, audio_data, frequency=None):
        """Save recorded audio segment to file"""
        if not audio_data:
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        freq_str = f"_{frequency}" if frequency else ""
        filename = f"{AUDIO_DIR}/transmission_{timestamp}{freq_str}.wav"

        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(audio_data))

            print(f"Saved transmission: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

    def record_with_vad(self, frequency=None, max_duration=3600):
        """Record audio with voice activity detection"""
        print(f"Starting stream capture from: {self.stream_url}")

        stream = self.capture_stream_audio()
        if not stream:
            return

        chunk_size = 1024 * 2  # 2 bytes per sample (16-bit)
        chunks_per_second = self.sample_rate // (chunk_size // 2)
        silence_chunks_threshold = int(self.silence_duration * chunks_per_second)

        recording_transmission = False
        current_transmission = []
        silence_count = 0
        total_chunks = 0
        max_chunks = max_duration * chunks_per_second

        print("Listening for transmissions...")

        try:
            while total_chunks < max_chunks:
                # Read audio chunk
                chunk = stream.read(chunk_size)
                if not chunk:
                    break

                total_chunks += 1

                # Check for voice activity
                has_voice = self.detect_voice_activity(chunk)

                if has_voice:
                    if not recording_transmission:
                        print("🎙️  Transmission detected - recording...")
                        recording_transmission = True
                        current_transmission = []

                    current_transmission.append(chunk)
                    silence_count = 0

                elif recording_transmission:
                    # Still recording but no voice detected
                    current_transmission.append(chunk)
                    silence_count += 1

                    # If silence exceeds threshold, end recording
                    if silence_count >= silence_chunks_threshold:
                        print("📁 Transmission ended - saving...")
                        self.save_audio_segment(current_transmission, frequency)
                        recording_transmission = False
                        current_transmission = []
                        silence_count = 0
                        print("👂 Listening for transmissions...")

                # Optional: Print status every 30 seconds
                if total_chunks % (30 * chunks_per_second) == 0:
                    print(f"⏱️  Monitoring... ({total_chunks // chunks_per_second}s elapsed)")

        except KeyboardInterrupt:
            print("\n🛑 Recording stopped by user")

        finally:
            # Save any remaining transmission
            if current_transmission:
                print("📁 Saving final transmission...")
                self.save_audio_segment(current_transmission, frequency)

            # Clean up
            if hasattr(self, 'ffmpeg_process'):
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()

            print("✅ Recording session completed")


# Alternative: Using pyaudio for system audio capture
class SystemAudioRecorder:
    def __init__(self, vad_threshold=0.01, silence_duration=2.0):
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration

        if not os.path.exists(AUDIO_DIR):
            os.makedirs(AUDIO_DIR)

    def find_audio_input_device(self, device_name_contains=""):
        """Find audio input device (useful for virtual audio cables)"""
        audio = pyaudio.PyAudio()

        print("Available audio devices:")
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            print(f"  {i}: {info['name']} - {info['maxInputChannels']} inputs")

            if device_name_contains.lower() in info['name'].lower():
                audio.terminate()
                return i

        audio.terminate()
        return None

    def record_system_audio_with_vad(self, device_index=None, frequency=None):
        """Record from system audio with VAD"""
        format = pyaudio.paInt16
        chunk = 1024

        audio = pyaudio.PyAudio()

        try:
            stream = audio.open(
                format=format,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk
            )

            print("🎧 Recording system audio with VAD...")

            recording_transmission = False
            current_transmission = []
            silence_count = 0
            chunks_per_second = SAMPLE_RATE // chunk
            silence_chunks_threshold = int(self.silence_duration * chunks_per_second)

            while True:
                data = stream.read(chunk, exception_on_overflow=False)

                # Simple VAD
                audio_array = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                normalized_rms = rms / 32768.0
                has_voice = normalized_rms > self.vad_threshold

                if has_voice:
                    if not recording_transmission:
                        print("🎙️  Transmission detected - recording...")
                        recording_transmission = True
                        current_transmission = []

                    current_transmission.append(data)
                    silence_count = 0

                elif recording_transmission:
                    current_transmission.append(data)
                    silence_count += 1

                    if silence_count >= silence_chunks_threshold:
                        print("📁 Transmission ended - saving...")
                        self.save_audio_segment(current_transmission, frequency)
                        recording_transmission = False
                        current_transmission = []
                        silence_count = 0
                        print("👂 Listening for transmissions...")

        except KeyboardInterrupt:
            print("\n🛑 Recording stopped")

        finally:
            if current_transmission:
                self.save_audio_segment(current_transmission, frequency)

            stream.stop_stream()
            stream.close()
            audio.terminate()

    def save_audio_segment(self, audio_data, frequency=None):
        """Save recorded audio segment"""
        if not audio_data:
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        freq_str = f"_{frequency}" if frequency else ""
        filename = f"{AUDIO_DIR}/transmission_{timestamp}{freq_str}.wav"

        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(audio_data))

            print(f"Saved: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving: {e}")
            return None


if __name__ == "__main__":
    from config import ATC_FREQUENCY

    # Option 1: Record from LiveATC stream
    # Find your local LiveATC stream URL (e.g., https://d.liveatc.net/klax_twr)
    stream_url = "https://s1-bos.liveatc.net/kmsp3_dep_ne?nocache=2025071301083069248"  # Replace with your local stream

    recorder = LiveATCRecorder(
        stream_url=stream_url,
        vad_threshold=0.2,  # Adjust based on your audio levels
        silence_duration=3.0  # Seconds of silence before ending recording
    )

    recorder.record_with_vad(frequency=ATC_FREQUENCY)

    # Option 2: Record system audio (if you have LiveATC playing in browser)
    # system_recorder = SystemAudioRecorder(vad_threshold=0.02)
    # system_recorder.record_system_audio_with_vad(frequency=ATC_FREQUENCY)