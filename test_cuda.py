# test_cuda.py
import torch
import whisper

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test tensor operation
    x = torch.rand(5, 3).cuda()
    print(f"Test tensor on GPU: {x.device}")

    # Test whisper model loading
    model = whisper.load_model("base", device="cuda")
    print("Whisper model loaded successfully on CUDA!")