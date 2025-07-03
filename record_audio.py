import pyaudio
import wave
import datetime
import os
from config import SAMPLE_RATE, CHANNELS, AUDIO_DIR


def record_audio(duration=60, frequency=None):
    """Record audio for specified duration in seconds"""
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)

    # Generate filename with timestamp and frequency
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    freq_str = f"_{frequency}" if frequency else ""
    filename = f"{AUDIO_DIR}{timestamp}{freq_str}.wav"

    # Configure audio recording
    format = pyaudio.paInt16
    chunk = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=CHANNELS,
                        rate=SAMPLE_RATE, input=True,
                        frames_per_buffer=chunk)

    print(f"Recording for {duration} seconds...")
    frames = []

    for i in range(0, int(SAMPLE_RATE / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to {filename}")
    return filename


if __name__ == "__main__":
    from config import ATC_FREQUENCY

    record_audio(duration=120, frequency=ATC_FREQUENCY)