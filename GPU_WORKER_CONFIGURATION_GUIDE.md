# GPU Worker Configuration Guide for ATC WSPR Multi-Channel Monitor

## Overview

This guide helps you configure the optimal number of transcription workers and model size based on your GPU's VRAM capacity. Each worker loads a separate Whisper model instance into GPU memory for parallel transcription processing.

## Quick Configuration

Edit `utils/config.py`:

```python
# Set your model size
MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large

# Set number of workers based on your GPU
NUM_TRANSCRIPTION_WORKERS = 3  # Adjust based on tables below
```

Or override via command line:
```bash
python main.py --multi --workers 5
```

---

## VRAM Requirements Per Model

These are **per-worker** VRAM requirements (approximate):

| Model Size | Parameters | VRAM (CUDA/NVIDIA) | VRAM (DirectML/AMD) | Quality Level | Best For |
|------------|------------|-------------------|---------------------|---------------|----------|
| **tiny**   | 39M        | ~1 GB             | ~1.2 GB            | Basic         | Testing, very limited VRAM |
| **base**   | 74M        | ~1.5 GB           | ~1.8 GB            | Good          | Backup option |
| **small**  | 244M       | **~2 GB**         | **~2.5 GB**        | **Better**    | **Recommended minimum** |
| **medium** | 769M       | **~5 GB**         | **~6 GB**          | **Great**     | **Best balance** |
| **large**  | 1550M      | **~10 GB**        | **~12 GB**         | Best          | Maximum accuracy |
| large-v2   | 1550M      | ~10 GB            | ~12 GB             | Best          | Improved large |
| large-v3   | 1550M      | ~10 GB            | ~12 GB             | Best          | Latest large |

**Notes:**
- DirectML (AMD GPUs) typically uses 15-20% more VRAM than CUDA (NVIDIA)
- Always leave 1-2 GB VRAM headroom for system/display
- More workers = faster parallel processing but more VRAM usage

---

## Recommended Configurations by GPU

### NVIDIA GPUs (CUDA)

#### Budget/Entry Level (4-6 GB VRAM)
**Examples:** GTX 1650, GTX 1660, RTX 3050

| GPU Model | VRAM | Recommended Config |
|-----------|------|-------------------|
| GTX 1650  | 4 GB | `MODEL_SIZE = "tiny"`, `NUM_TRANSCRIPTION_WORKERS = 3` |
| GTX 1660  | 6 GB | `MODEL_SIZE = "small"`, `NUM_TRANSCRIPTION_WORKERS = 2` |
| RTX 3050  | 8 GB | `MODEL_SIZE = "small"`, `NUM_TRANSCRIPTION_WORKERS = 3` |

```python
# Example config for GTX 1660 (6GB)
MODEL_SIZE = "small"
NUM_TRANSCRIPTION_WORKERS = 2
```

---

#### Mid-Range (8-12 GB VRAM)
**Examples:** RTX 2070, RTX 3060, RTX 3070

| GPU Model       | VRAM  | Recommended Config |
|-----------------|-------|-------------------|
| RTX 2070        | 8 GB  | `MODEL_SIZE = "small"`, `NUM_TRANSCRIPTION_WORKERS = 3` |
| RTX 3060 12GB   | 12 GB | `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 2` |
|                 |       | OR `MODEL_SIZE = "small"`, `NUM_TRANSCRIPTION_WORKERS = 5` |
|                 |       | OR `MODEL_SIZE = "large"`, `NUM_TRANSCRIPTION_WORKERS = 1` |
| RTX 3070        | 8 GB  | `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 1` |
|                 |       | OR `MODEL_SIZE = "small"`, `NUM_TRANSCRIPTION_WORKERS = 3` |

```python
# Example config for RTX 3060 12GB - Best balance
MODEL_SIZE = "medium"
NUM_TRANSCRIPTION_WORKERS = 2
```

---

#### High-End (16-24 GB VRAM)
**Examples:** RTX 3080, RTX 3090, RTX 4080, RTX 4090

| GPU Model   | VRAM  | Recommended Config |
|-------------|-------|-------------------|
| RTX 3080    | 10 GB | `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 1-2` |
|             |       | OR `MODEL_SIZE = "small"`, `NUM_TRANSCRIPTION_WORKERS = 4` |
| RTX 3090    | 24 GB | `MODEL_SIZE = "large"`, `NUM_TRANSCRIPTION_WORKERS = 2` |
|             |       | OR `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 4` |
| RTX 4080    | 16 GB | `MODEL_SIZE = "large"`, `NUM_TRANSCRIPTION_WORKERS = 1` |
|             |       | OR `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 3` |
| RTX 4090    | 24 GB | `MODEL_SIZE = "large"`, `NUM_TRANSCRIPTION_WORKERS = 2` |
|             |       | OR `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 4` |

```python
# Example config for RTX 4090 24GB - Maximum performance
MODEL_SIZE = "large"
NUM_TRANSCRIPTION_WORKERS = 2

# Or for faster processing with slight quality trade-off
MODEL_SIZE = "medium"
NUM_TRANSCRIPTION_WORKERS = 4
```

---

### AMD GPUs (DirectML)

**Note:** DirectML uses slightly more VRAM than CUDA, so reduce workers by ~20% compared to NVIDIA equivalents.

#### Mid-Range AMD (8-12 GB VRAM)
**Examples:** RX 6600 XT, RX 6700 XT

| GPU Model    | VRAM  | Recommended Config |
|--------------|-------|-------------------|
| RX 6600 XT   | 8 GB  | `MODEL_SIZE = "small"`, `NUM_TRANSCRIPTION_WORKERS = 2-3` |
| RX 6700 XT   | 12 GB | `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 1-2` |
|              |       | OR `MODEL_SIZE = "small"`, `NUM_TRANSCRIPTION_WORKERS = 4` |

```python
# Example config for RX 6700 XT (12GB)
MODEL_SIZE = "medium"
NUM_TRANSCRIPTION_WORKERS = 2
PREFER_AMD_GPU = True
DIRECTML_ENABLED = True
```

---

#### High-End AMD (16-24 GB VRAM)
**Examples:** RX 6800, RX 6800 XT, RX 6900 XT, RX 7900 XT, RX 7900 XTX

| GPU Model      | VRAM  | Recommended Config |
|----------------|-------|-------------------|
| RX 6800        | 16 GB | `MODEL_SIZE = "large"`, `NUM_TRANSCRIPTION_WORKERS = 1` |
|                |       | OR `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 2-3` |
| RX 6900 XT     | 16 GB | `MODEL_SIZE = "large"`, `NUM_TRANSCRIPTION_WORKERS = 1` |
|                |       | OR `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 3` |
| RX 7900 XT     | 20 GB | `MODEL_SIZE = "large"`, `NUM_TRANSCRIPTION_WORKERS = 1-2` |
|                |       | OR `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 3` |
| RX 7900 XTX    | 24 GB | `MODEL_SIZE = "large"`, `NUM_TRANSCRIPTION_WORKERS = 2` |
|                |       | OR `MODEL_SIZE = "medium"`, `NUM_TRANSCRIPTION_WORKERS = 4` |

```python
# Example config for RX 7900 XTX (24GB)
MODEL_SIZE = "large"
NUM_TRANSCRIPTION_WORKERS = 2
PREFER_AMD_GPU = True
DIRECTML_ENABLED = True
PREFER_ONNX_DIRECTML = True
```

---

## Worker Count Decision Matrix

Use this table to quickly determine worker count:

| Available VRAM | tiny | base | small | medium | large |
|----------------|------|------|-------|--------|-------|
| 4 GB           | 3    | 2    | 1     | ❌     | ❌    |
| 6 GB           | 5    | 3    | 2     | 1      | ❌    |
| 8 GB           | 7    | 5    | 3     | 1      | ❌    |
| 12 GB          | 10   | 7    | 5     | 2      | 1     |
| 16 GB          | 14   | 10   | 6     | 3      | 1     |
| 24 GB          | 20   | 14   | 9     | 4      | 2     |
| 32 GB+         | 28   | 20   | 12    | 6      | 3     |

**Key:**
- ✅ Numbers = Recommended worker count
- ❌ = Not enough VRAM (use smaller model)

---

## Performance vs. Quality Trade-offs

### Scenario 1: Maximum Accuracy (Recommended)
**Goal:** Best transcription quality for critical ATC communications

```python
MODEL_SIZE = "large"  # or "medium" if VRAM limited
NUM_TRANSCRIPTION_WORKERS = 1  # Quality over speed
```

**Best for:** Single busy frequency, critical operations, archival quality

---

### Scenario 2: Balanced (Recommended for Most Users)
**Goal:** Good quality with reasonable processing speed

```python
MODEL_SIZE = "medium"
NUM_TRANSCRIPTION_WORKERS = 2-3
```

**Best for:** 2-4 channels, typical ATC monitoring

---

### Scenario 3: Maximum Throughput
**Goal:** Handle many channels simultaneously

```python
MODEL_SIZE = "small"  # Still quite accurate
NUM_TRANSCRIPTION_WORKERS = 5-8
```

**Best for:** 5+ channels, busy airspace, real-time monitoring

---

### Scenario 4: Testing/Development
**Goal:** Quick iterations, don't care about quality

```python
MODEL_SIZE = "tiny"
NUM_TRANSCRIPTION_WORKERS = 3-5
```

**Best for:** Development, testing, debugging

---

## CPU Mode Configuration

If no GPU is available or `ENABLE_GPU = False`:

```python
ENABLE_GPU = False
MODEL_SIZE = "small"  # or "base" for slower systems
NUM_TRANSCRIPTION_WORKERS = 4-8  # CPU can handle more parallel workers
```

**Warning:** CPU transcription is 10-50x slower than GPU!

**Typical CPU transcription times:**
- tiny: 10-20 seconds for 5-second audio
- small: 30-60 seconds for 5-second audio
- medium: 60-120 seconds for 5-second audio
- large: 120-300 seconds for 5-second audio

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```
or
```
DmlExecutionProvider error: Out of memory
```

**Solutions:**
1. **Reduce workers:**
   ```python
   NUM_TRANSCRIPTION_WORKERS = 1  # Start with 1 and increase gradually
   ```

2. **Use smaller model:**
   ```python
   MODEL_SIZE = "medium"  # or "small"
   ```

3. **Check VRAM usage:**
   ```bash
   # NVIDIA
   nvidia-smi
   
   # AMD (Windows)
   Get-Counter "\GPU Process Memory(*)\Local Usage"
   ```

4. **Close other GPU applications** (browsers, games, etc.)

5. **For DirectML, try:**
   ```python
   WHISPER_COMPUTE_TYPE = "int8"  # Uses less VRAM
   ```

---

### Slow Transcription

**If transcription is taking too long:**

1. **Increase workers** (if you have VRAM):
   ```python
   NUM_TRANSCRIPTION_WORKERS = 4
   ```

2. **Use smaller model:**
   ```python
   MODEL_SIZE = "small"  # Still quite accurate
   ```

3. **Verify GPU is being used:**
   - Check startup logs for "Using CUDA" or "Using DirectML"
   - If says "Using CPU", check GPU configuration

---

### Worker Starvation

**Symptoms:** Workers sitting idle while transmissions queue up

**Solutions:**
1. Increase workers if you have VRAM
2. Channels are too slow (not a problem)
3. Check for errors in worker threads

---

## Monitoring Worker Performance

The GUI shows real-time worker status:
- **Green box:** Worker idle (ready for work)
- **Orange box pulsing:** Worker busy transcribing
- **Queue size:** Number of pending transcriptions

**Optimal setup:** Queue rarely exceeds 1-2 items

**Too few workers:** Queue consistently 3+ items

**Too many workers:** Workers mostly idle (wasting VRAM)

---

## Example Configurations

### RTX 3060 12GB - Monitoring PDX (3 channels)
```python
# utils/config.py
MODEL_SIZE = "medium"
NUM_TRANSCRIPTION_WORKERS = 2
ENABLE_GPU = True
GPU_BACKEND = "auto"
WHISPER_COMPUTE_TYPE = "float16"
```

**Expected performance:**
- ~5-10 seconds transcription time per 5-second transmission
- Can handle 2 simultaneous transmissions
- Excellent quality for ATC communications

---

### RTX 4090 24GB - Monitoring Busy Airport (8 channels)
```python
# utils/config.py
MODEL_SIZE = "medium"  # large if you want max quality
NUM_TRANSCRIPTION_WORKERS = 4
ENABLE_GPU = True
GPU_BACKEND = "auto"
WHISPER_COMPUTE_TYPE = "float16"
```

**Expected performance:**
- ~3-5 seconds transcription time
- Can handle 4 simultaneous transmissions
- Nearly real-time processing

---

### AMD RX 7900 XTX 24GB - Maximum Quality
```python
# utils/config.py
MODEL_SIZE = "large"
NUM_TRANSCRIPTION_WORKERS = 2
ENABLE_GPU = True
GPU_BACKEND = "auto"
PREFER_AMD_GPU = True
DIRECTML_ENABLED = True
PREFER_ONNX_DIRECTML = True
WHISPER_COMPUTE_TYPE = "int8"  # DirectML works better with int8
```

**Expected performance:**
- ~8-12 seconds transcription time
- Highest quality transcriptions
- Can handle 2 simultaneous transmissions

---

### Limited 4GB GPU - Testing Setup
```python
# utils/config.py
MODEL_SIZE = "small"
NUM_TRANSCRIPTION_WORKERS = 1
ENABLE_GPU = True
WHISPER_COMPUTE_TYPE = "int8"
```

**Expected performance:**
- ~5-8 seconds transcription time
- Good quality for testing
- Single worker (no parallelism)

---

## Advanced: Dynamic Worker Scaling (Future Feature)

Currently, worker count is fixed at startup. Future versions may support:
- Auto-scaling based on queue length
- GPU memory monitoring
- Automatic model size adjustment
- Load balancing across multiple GPUs

---

## Verification

After configuration, verify setup at startup:

```
🎙️ ═══════════════════════════════════════════════
   Starting Multi-Channel ATC Monitor
═══════════════════════════════════════════════════

✓ GPU: NVIDIA GeForce RTX 3060 (12GB)
✓ Backend: CUDA
✓ Model: medium (769M parameters)
✓ Workers: 2
✓ VRAM per worker: ~5GB
✓ Total estimated VRAM: ~10GB + 2GB overhead = 12GB

Monitoring 3 channels
```

---

## Quick Reference Card

**Print this and keep by your computer:**

```
┌─────────────────────────────────────────────────────────┐
│ QUICK REFERENCE: Worker Configuration                  │
├─────────────────────────────────────────────────────────┤
│ Your GPU VRAM: _____ GB                                 │
│                                                         │
│ For MAXIMUM QUALITY:                                    │
│   MODEL_SIZE = "large"                                  │
│   NUM_TRANSCRIPTION_WORKERS = 1                         │
│                                                         │
│ For BALANCED (RECOMMENDED):                             │
│   MODEL_SIZE = "medium"                                 │
│   NUM_TRANSCRIPTION_WORKERS = (VRAM ÷ 6)               │
│                                                         │
│ For MAXIMUM SPEED:                                      │
│   MODEL_SIZE = "small"                                  │
│   NUM_TRANSCRIPTION_WORKERS = (VRAM ÷ 2.5)             │
│                                                         │
│ Remember: Leave 1-2GB VRAM headroom!                    │
└─────────────────────────────────────────────────────────┘
```

---

## Still Need Help?

1. Run GPU test: `python utils/gpu_utils.py`
2. Start with 1 worker and increase gradually
3. Monitor VRAM usage during operation
4. Check logs for OOM errors

**Default safe configuration (works on most 8GB+ GPUs):**
```python
MODEL_SIZE = "small"
NUM_TRANSCRIPTION_WORKERS = 2
```

This will work on virtually any modern GPU while providing good transcription quality.
