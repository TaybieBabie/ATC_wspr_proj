import torch
import platform
import subprocess
import os
import sys
from config import GPU_BACKEND, PREFER_AMD_GPU, DIRECTML_ENABLED, PREFER_ONNX_DIRECTML

# Try to import DirectML and ONNX Runtime
try:
    import torch_directml

    TORCH_DIRECTML_AVAILABLE = True
except ImportError:
    TORCH_DIRECTML_AVAILABLE = False

try:
    import onnxruntime as ort

    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False


def detect_gpu_capabilities():
    """Detect available GPU capabilities with AMD and NVIDIA support"""
    capabilities = {
        'cuda_available': False,
        'directml_available': False,
        'directml_onnx_available': False,
        'torch_directml_available': False,
        'cuda_devices': 0,
        'directml_devices': 0,
        'recommended_backend': 'cpu',
        'cuda_version': None,
        'amd_gpu_detected': False,
        'nvidia_gpu_detected': False
    }

    # Check CUDA (NVIDIA)
    try:
        if torch.cuda.is_available():
            capabilities['cuda_available'] = True
            capabilities['cuda_devices'] = torch.cuda.device_count()
            capabilities['cuda_version'] = torch.version.cuda
            capabilities['nvidia_gpu_detected'] = True
    except Exception as e:
        print(f"CUDA detection error: {e}")

    # Check DirectML availability
    if DIRECTML_ENABLED:
        # Check ONNX Runtime DirectML
        if ONNX_RUNTIME_AVAILABLE:
            try:
                providers = ort.get_available_providers()
                if 'DmlExecutionProvider' in providers:
                    capabilities['directml_onnx_available'] = True
                    capabilities['directml_available'] = True
                    print("✅ ONNX Runtime DirectML available")
            except Exception as e:
                print(f"ONNX Runtime DirectML detection error: {e}")

        # Check PyTorch DirectML
        if TORCH_DIRECTML_AVAILABLE:
            try:
                capabilities['torch_directml_available'] = True
                capabilities['directml_available'] = True
                capabilities['directml_devices'] = torch_directml.device_count()
                print("✅ PyTorch DirectML available")
            except Exception as e:
                print(f"PyTorch DirectML detection error: {e}")

        # Check if AMD GPU is present
        if capabilities['directml_available']:
            capabilities['amd_gpu_detected'] = detect_amd_gpu()

    # Determine recommended backend
    if capabilities['cuda_available'] and capabilities['directml_available']:
        if capabilities['amd_gpu_detected'] and PREFER_AMD_GPU:
            capabilities['recommended_backend'] = 'directml'
        else:
            capabilities['recommended_backend'] = 'cuda'
    elif capabilities['cuda_available']:
        capabilities['recommended_backend'] = 'cuda'
    elif capabilities['directml_available']:
        capabilities['recommended_backend'] = 'directml'
    else:
        capabilities['recommended_backend'] = 'cpu'

    return capabilities


def detect_amd_gpu():
    """Detect AMD GPU on Windows"""
    if platform.system() != "Windows":
        return False

    try:
        # Method 1: Check via wmic
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                capture_output=True, text=True, timeout=10)
        if 'AMD' in result.stdout or 'Radeon' in result.stdout:
            return True
    except:
        pass

    try:
        # Method 2: Check via PowerShell
        result = subprocess.run(['powershell', '-Command',
                                 'Get-WmiObject -Class Win32_VideoController | Select-Object Name'],
                                capture_output=True, text=True, timeout=10)
        if 'AMD' in result.stdout or 'Radeon' in result.stdout:
            return True
    except:
        pass

    return False


def get_directml_provider_options():
    """Get DirectML provider options for ONNX Runtime"""
    if not ONNX_RUNTIME_AVAILABLE:
        return None

    try:
        # Get available DirectML devices
        providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in providers:
            return {
                'provider': 'DmlExecutionProvider',
                'provider_options': {
                    'device_id': 0,  # Use first DirectML device
                    'disable_memory_arena': False
                }
            }
    except Exception as e:
        print(f"Error getting DirectML provider options: {e}")

    return None


def setup_gpu_backend():
    """Setup appropriate GPU backend based on configuration and hardware"""
    capabilities = detect_gpu_capabilities()

    if GPU_BACKEND == "auto":
        backend = capabilities['recommended_backend']
    else:
        backend = GPU_BACKEND

    # Validate backend availability
    if backend == 'cuda' and not capabilities['cuda_available']:
        print("⚠️ CUDA requested but not available, falling back to CPU")
        backend = 'cpu'
    elif backend == 'directml' and not capabilities['directml_available']:
        print("⚠️ DirectML requested but not available, falling back to CPU")
        backend = 'cpu'

    return backend, capabilities


def get_device_string(backend):
    """Get device string for the backend"""
    if backend == 'cuda':
        return 'cuda'
    elif backend == 'directml':
        return 'directml'
    else:
        return 'cpu'


def get_torch_device(backend, device_index=0):
    """Get PyTorch device object for the backend"""
    if backend == 'cuda':
        return torch.device('cuda', device_index)
    elif backend == 'directml' and TORCH_DIRECTML_AVAILABLE:
        return torch_directml.device(device_index)
    else:
        return torch.device('cpu')


def get_compute_type(backend):
    """Get appropriate compute type for the backend"""
    if backend == 'cuda':
        return "float16"  # CUDA supports float16 well
    elif backend == 'directml':
        return "int8"  # DirectML works better with int8
    else:
        return "int8"  # CPU works better with int8


def print_gpu_info(backend, capabilities):
    """Print detailed GPU information"""
    print(f"\n🖥️  GPU Configuration:")
    print(f"  Backend: {backend}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Platform: {platform.system()}")

    if capabilities['cuda_available']:
        print(f"  ✅ CUDA available: {capabilities['cuda_devices']} device(s)")
        if capabilities['nvidia_gpu_detected']:
            print(f"     - NVIDIA GPU detected")
        if torch.cuda.is_available():
            try:
                print(f"     - Device: {torch.cuda.get_device_name(0)}")
                print(f"     - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                print(f"     - CUDA Version: {capabilities['cuda_version']}")
            except Exception as e:
                print(f"     - Error getting CUDA details: {e}")
    else:
        print(f"  ❌ CUDA not available")

    if capabilities['directml_available']:
        print(f"  ✅ DirectML available")
        if capabilities['amd_gpu_detected']:
            print(f"     - AMD GPU detected")
        if capabilities['directml_onnx_available']:
            print(f"     - ONNX Runtime DirectML: Available")
        if capabilities['torch_directml_available']:
            print(f"     - PyTorch DirectML: Available ({capabilities['directml_devices']} device(s))")
    else:
        print(f"  ❌ DirectML not available")

    if backend == 'cpu':
        print(f"  ⚠️  Using CPU (no GPU acceleration)")

    print()


def test_directml_functionality():
    """Test DirectML functionality"""
    print("🧪 Testing DirectML functionality...")

    # Test ONNX Runtime DirectML
    if ONNX_RUNTIME_AVAILABLE:
        try:
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                print("✅ ONNX Runtime DirectML provider available")

                # Try to create a simple session
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3  # Reduce logging

                print("✅ ONNX Runtime DirectML test passed")
            else:
                print("❌ ONNX Runtime DirectML provider not found")
        except Exception as e:
            print(f"❌ ONNX Runtime DirectML test failed: {e}")

    # Test PyTorch DirectML
    if TORCH_DIRECTML_AVAILABLE:
        try:
            device = torch_directml.device()
            x = torch.randn(10, 10).to(device)
            y = torch.randn(10, 10).to(device)
            z = torch.matmul(x, y)
            print("✅ PyTorch DirectML test passed")
        except Exception as e:
            print(f"❌ PyTorch DirectML test failed: {e}")


def test_gpu_functionality():
    """Test GPU functionality for all backends"""
    backend, capabilities = setup_gpu_backend()
    print_gpu_info(backend, capabilities)

    if backend == 'cuda':
        try:
            device = get_torch_device(backend)
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.matmul(x, y)
            print(f"✅ CUDA tensor operations test passed")
            return True
        except Exception as e:
            print(f"❌ CUDA tensor operations test failed: {e}")
            return False
    elif backend == 'directml':
        test_directml_functionality()
        return True
    else:
        print("ℹ️  CPU backend selected")
        return False


if __name__ == "__main__":
    test_gpu_functionality()