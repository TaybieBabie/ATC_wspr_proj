# test_amd_gpu.py
import time
from faster_whisper import WhisperModel

# Test CPU
model_cpu = WhisperModel("tiny", device="cpu", compute_type="int8")
start = time.time()
result = model_cpu.transcribe("test_audio.wav")
cpu_time = time.time() - start

# Test DirectML
model_dml = WhisperModel("tiny", device="cpu", compute_type="int8",
                         provider="DmlExecutionProvider")
start = time.time()
result = model_dml.transcribe("test_audio.wav")
dml_time = time.time() - start

print(f"CPU: {cpu_time:.2f}s, DirectML: {dml_time:.2f}s")
print(f"Speedup: {cpu_time/dml_time:.2f}x")