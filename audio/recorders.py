# recording.py
import pyaudio
import wave
import datetime
import os
import threading
import queue
import numpy as np
import requests
import subprocess
import time

from utils.console_logger import info, success, warning, error
from utils.config import SAMPLE_RATE, CHANNELS, AUDIO_DIR


class LiveATCRecorder:
    def __init__(self, stream_url, vad_threshold=0.01, silence_duration=2.0):
        self.stream_url = stream_url
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.is_recording = False
        self.ffmpeg_process = None
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS

        if not os.path.exists(AUDIO_DIR):
            os.makedirs(AUDIO_DIR)

    def capture_stream_audio(self):
        """Capture audio from LiveATC stream using ffmpeg"""
        cmd = [
            'ffmpeg',
            '-i', self.stream_url,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-'
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            return self.ffmpeg_process.stdout
        except FileNotFoundError:
            error("ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
            return None
        except Exception as e:
            error(f"Error starting stream capture: {e}")
            return None

    def detect_voice_activity(self, audio_data):
        """Simple VAD based on RMS energy"""
        if not audio_data:
            return False

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if audio_array.size == 0:
            return False

        rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
        normalized_rms = rms / 32768.0
        return normalized_rms > self.vad_threshold

    def save_audio_segment(self, audio_data, frequency=None):
        """Save recorded audio segment to file"""
        if not audio_data:
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        freq_str = f"_{frequency.replace('.', 'p')}" if frequency else ""
        filename = os.path.join(AUDIO_DIR, f"transmission_{timestamp}{freq_str}.wav")

        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(audio_data))
            success(f"Saved transmission: {os.path.basename(filename)}", emoji="💾")
            return filename
        except Exception as e:
            error(f"Error saving audio to {filename}: {e}")
            return None

    def record_with_vad(self, frequency=None, max_duration=None):
        """Record audio with voice activity detection."""
        info(f"Starting stream capture from: {self.stream_url}")

        stream = self.capture_stream_audio()
        if not stream:
            return

        chunk_size = 1024 * 2
        samples_per_chunk = chunk_size // 2
        chunks_per_second = self.sample_rate / samples_per_chunk
        silence_chunks_threshold = int(self.silence_duration * chunks_per_second)

        recording_transmission = False
        current_transmission = []
        silence_count = 0

        start_time = time.time()
        info("Listening for transmissions...", emoji="👂")

        try:
            # The main loop now simply runs until the parent thread stops it.
            while True:
                # If a max_duration is set, check if we've exceeded it.
                if max_duration is not None and (time.time() - start_time) > max_duration:
                    info("Maximum recording duration reached.")
                    break

                chunk = stream.read(chunk_size)
                if not chunk:
                    info("Stream ended.")
                    break

                has_voice = self.detect_voice_activity(chunk)

                if has_voice:
                    if not recording_transmission:
                        info("Transmission detected - recording...", emoji="🎙️")
                        recording_transmission = True
                        current_transmission = []

                    current_transmission.append(chunk)
                    silence_count = 0

                elif recording_transmission:
                    current_transmission.append(chunk)
                    silence_count += 1

                    if silence_count >= silence_chunks_threshold:
                        info("Transmission ended - saving...", emoji="📁")
                        self.save_audio_segment(current_transmission, frequency)
                        recording_transmission = False
                        current_transmission = []
                        silence_count = 0
                        info("Listening for transmissions...", emoji="👂")

        except Exception as e:
            error(f"An error occurred during recording: {e}")
        finally:
            if recording_transmission and current_transmission:
                info("Saving final transmission...", emoji="📁")
                self.save_audio_segment(current_transmission, frequency)

            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_process.kill()
            success("Recording session completed.", emoji="🎬")


class SystemAudioRecorder:
    def __init__(self, vad_threshold=0.01, silence_duration=2.0):
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        if not os.path.exists(AUDIO_DIR):
            os.makedirs(AUDIO_DIR)

    def record_system_audio_with_vad(self, frequency=None, device_index=None):
        """Record from system audio with VAD"""
        p = pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
        except Exception as e:
            error(f"Failed to open audio stream: {e}")
            p.terminate()
            return

        info("Recording system audio with VAD...", emoji="🎧")
        # Identical logic to LiveATC recorder can be applied here if needed
        # For now, keeping it simple
        # ...

    def save_audio_segment(self, audio_data, frequency=None):
        """Save recorded audio segment"""
        if not audio_data:
            return None
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        freq_str = f"_{frequency.replace('.', 'p')}" if frequency else ""
        filename = os.path.join(AUDIO_DIR, f"transmission_{timestamp}{freq_str}.wav")
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(audio_data))
            success(f"Saved: {os.path.basename(filename)}", emoji="💾")
            return filename
        except Exception as e:
            error(f"Error saving audio to {filename}: {e}")
            return None


class EnhancedLiveATCRecorder(LiveATCRecorder):
    """Enhanced version of LiveATCRecorder with callback support."""
    
    def __init__(self, stream_url, vad_threshold=0.01, silence_duration=2.0, callback=None):
        super().__init__(stream_url, vad_threshold, silence_duration)
        self.callback = callback

    def save_audio_segment(self, audio_data, frequency=None):
        """Save audio segment and call callback if provided."""
        filename = super().save_audio_segment(audio_data, frequency)
        if filename and self.callback:
            self.callback(filename)
        return filename


class EnhancedSystemAudioRecorder(SystemAudioRecorder):
    """Enhanced version of SystemAudioRecorder with callback support."""
    
    def __init__(self, vad_threshold=0.01, silence_duration=2.0, callback=None):
        super().__init__(vad_threshold, silence_duration)
        self.callback = callback

    def save_audio_segment(self, audio_data, frequency=None):
        """Save audio segment and call callback if provided."""
        filename = super().save_audio_segment(audio_data, frequency)
        if filename and self.callback:
            self.callback(filename)
        return filename
