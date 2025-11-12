#!/usr/bin/env python3
"""
GPU Worker Configuration Helper
Automatically detects your GPU and recommends optimal worker/model settings
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.gpu_utils import detect_gpu_capabilities, print_gpu_info, setup_gpu_backend
from utils.console_logger import info, success, warning, error, section
import torch


def get_gpu_vram():
    """Detect available GPU VRAM in GB"""
    backend, capabilities = setup_gpu_backend()
    
    vram_gb = 0
    gpu_name = "Unknown"
    
    if backend == 'cuda' and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as e:
            error(f"Error getting CUDA VRAM: {e}")
    elif backend == 'directml' and capabilities['amd_gpu_detected']:
        # Get AMD GPU info
        if capabilities['amd_gpus']:
            gpu_info = capabilities['amd_gpus'][0]
            vram_gb = gpu_info['vram']
            gpu_name = gpu_info['name']
    
    return vram_gb, gpu_name, backend


def recommend_configuration(vram_gb, backend):
    """Recommend optimal configuration based on VRAM"""
    
    configs = []
    
    # Model VRAM requirements (per worker)
    if backend == 'directml':
        # DirectML uses more VRAM
        vram_reqs = {
            'tiny': 1.2,
            'base': 1.8,
            'small': 2.5,
            'medium': 6.0,
            'large': 12.0
        }
    else:
        # CUDA/CPU
        vram_reqs = {
            'tiny': 1.0,
            'base': 1.5,
            'small': 2.0,
            'medium': 5.0,
            'large': 10.0
        }
    
    # Reserve 1.5-2GB for system
    available_vram = vram_gb - 2.0
    
    if available_vram < 2:
        warning(f"Only {vram_gb:.1f}GB VRAM detected - limited options available")
        available_vram = max(vram_gb - 1, 1)
    
    # Calculate configurations
    for model_size, vram_per_worker in vram_reqs.items():
        max_workers = int(available_vram / vram_per_worker)
        if max_workers > 0:
            configs.append({
                'model': model_size,
                'workers': max_workers,
                'vram_used': max_workers * vram_per_worker,
                'vram_per_worker': vram_per_worker
            })
    
    return configs


def display_recommendations(vram_gb, gpu_name, backend, configs):
    """Display configuration recommendations"""
    
    section("GPU Configuration Recommendations", emoji="🎮")
    
    info(f"GPU Detected: {gpu_name}")
    info(f"Total VRAM: {vram_gb:.1f} GB")
    info(f"Backend: {backend.upper()}")
    
    print()
    section("Recommended Configurations", emoji="⚙️")
    
    if not configs:
        error("Insufficient VRAM detected!")
        error("Consider using CPU mode or upgrading GPU")
        return
    
    # Group by use case
    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│ MAXIMUM QUALITY (Recommended for critical operations)                  │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    
    # Find best quality config
    best_quality = None
    for config in reversed(configs):
        if config['workers'] >= 1:
            best_quality = config
            break
    
    if best_quality:
        print(f"\n  Model Size: {best_quality['model'].upper()}")
        print(f"  Workers: {best_quality['workers']}")
        print(f"  VRAM Usage: ~{best_quality['vram_used']:.1f} GB / {vram_gb:.1f} GB")
        print(f"\n  In config.py:")
        print(f"    MODEL_SIZE = \"{best_quality['model']}\"")
        print(f"    NUM_TRANSCRIPTION_WORKERS = {best_quality['workers']}")
    
    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│ BALANCED (Recommended for most users)                                  │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    
    # Find balanced config (medium with 2-3 workers or small with 3-5)
    balanced = None
    for config in configs:
        if config['model'] == 'medium' and config['workers'] >= 2:
            balanced = config
            balanced['workers'] = min(config['workers'], 3)
            break
        elif config['model'] == 'small' and config['workers'] >= 3:
            balanced = config
            balanced['workers'] = min(config['workers'], 5)
            break
    
    if balanced:
        print(f"\n  Model Size: {balanced['model'].upper()}")
        print(f"  Workers: {balanced['workers']}")
        print(f"  VRAM Usage: ~{balanced['workers'] * balanced['vram_per_worker']:.1f} GB / {vram_gb:.1f} GB")
        print(f"\n  In config.py:")
        print(f"    MODEL_SIZE = \"{balanced['model']}\"")
        print(f"    NUM_TRANSCRIPTION_WORKERS = {balanced['workers']}")
    
    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│ MAXIMUM THROUGHPUT (For handling many channels)                        │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    
    # Find throughput config (small with max workers)
    throughput = None
    for config in configs:
        if config['model'] == 'small':
            throughput = config
            break
    
    if throughput:
        print(f"\n  Model Size: {throughput['model'].upper()}")
        print(f"  Workers: {throughput['workers']}")
        print(f"  VRAM Usage: ~{throughput['vram_used']:.1f} GB / {vram_gb:.1f} GB")
        print(f"\n  In config.py:")
        print(f"    MODEL_SIZE = \"{throughput['model']}\"")
        print(f"    NUM_TRANSCRIPTION_WORKERS = {throughput['workers']}")
    
    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│ ALL POSSIBLE CONFIGURATIONS                                             │")
    print("└─────────────────────────────────────────────────────────────────────────┘\n")
    
    print(f"  {'Model':<10} {'Workers':<10} {'VRAM Used':<15} {'Config'}")
    print(f"  {'-'*10} {'-'*10} {'-'*15} {'-'*40}")
    
    for config in reversed(configs):
        if config['workers'] > 0:
            vram_str = f"{config['vram_used']:.1f}/{vram_gb:.1f} GB"
            config_str = f"MODEL_SIZE=\"{config['model']}\", WORKERS={config['workers']}"
            print(f"  {config['model']:<10} {config['workers']:<10} {vram_str:<15} {config_str}")
    
    print()


def generate_config_file(config):
    """Generate a snippet for config.py"""
    
    section("Configuration Snippet", emoji="📝")
    
    print("\nCopy this to your utils/config.py:\n")
    print("─" * 60)
    print(f"MODEL_SIZE = \"{config['model']}\"")
    print(f"NUM_TRANSCRIPTION_WORKERS = {config['workers']}")
    print("─" * 60)
    
    print(f"\nOr run with command line:")
    print(f"  python main.py --multi --workers {config['workers']}")
    print()


def interactive_mode(configs):
    """Interactive configuration selection"""
    
    print("\n" + "─" * 70)
    print("Select a configuration:")
    print("─" * 70)
    
    options = []
    idx = 1
    
    # Add preset options
    for config in reversed(configs):
        if config['workers'] >= 1:
            print(f"  [{idx}] {config['model'].upper()} with {config['workers']} worker(s) "
                  f"(~{config['vram_used']:.1f} GB)")
            options.append(config)
            idx += 1
    
    print(f"  [0] Exit without applying")
    print()
    
    try:
        choice = input("Enter choice [0-{}]: ".format(len(options)))
        choice = int(choice)
        
        if choice == 0:
            info("Exiting without changes")
            return None
        elif 1 <= choice <= len(options):
            selected = options[choice - 1]
            success(f"Selected: {selected['model'].upper()} with {selected['workers']} workers")
            return selected
        else:
            error("Invalid choice")
            return None
    except (ValueError, KeyboardInterrupt):
        print()
        info("Cancelled")
        return None


def apply_configuration(config):
    """Apply configuration to config.py"""
    
    config_path = os.path.join(os.path.dirname(__file__), 'utils', 'config.py')
    
    if not os.path.exists(config_path):
        error(f"Could not find config.py at {config_path}")
        return False
    
    try:
        # Read existing config
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Modify relevant lines
        new_lines = []
        for line in lines:
            if line.strip().startswith('MODEL_SIZE ='):
                new_lines.append(f'MODEL_SIZE = "{config["model"]}"  # Updated by configure_gpu_workers.py\n')
            elif line.strip().startswith('NUM_TRANSCRIPTION_WORKERS ='):
                new_lines.append(f'NUM_TRANSCRIPTION_WORKERS = {config["workers"]}  # Updated by configure_gpu_workers.py\n')
            else:
                new_lines.append(line)
        
        # Write back
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        
        success(f"Configuration applied to {config_path}")
        return True
    
    except Exception as e:
        error(f"Failed to apply configuration: {e}")
        return False


def main():
    """Main configuration helper"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              ATC WSPR GPU Worker Configuration Helper                   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Detect GPU
    info("Detecting GPU capabilities...")
    vram_gb, gpu_name, backend = get_gpu_vram()
    
    if vram_gb == 0:
        warning("No GPU detected or GPU not accessible")
        warning("Configuring for CPU mode...")
        
        print("\nFor CPU mode, use:")
        print("  MODEL_SIZE = \"small\"  # or \"base\" for slower systems")
        print("  NUM_TRANSCRIPTION_WORKERS = 4-8  # CPU can handle more workers")
        print("  ENABLE_GPU = False")
        print("\nNote: CPU transcription is 10-50x slower than GPU!")
        return
    
    # Calculate recommendations
    configs = recommend_configuration(vram_gb, backend)
    
    # Display recommendations
    display_recommendations(vram_gb, gpu_name, backend, configs)
    
    # Additional info
    print("\n💡 Tips:")
    print("  • Start with recommended 'BALANCED' configuration")
    print("  • Monitor VRAM usage: nvidia-smi (NVIDIA) or Task Manager (AMD)")
    print("  • If you get OOM errors, reduce workers or use smaller model")
    print("  • See GPU_WORKER_CONFIGURATION_GUIDE.md for detailed information")
    print()
    
    # Interactive mode
    try:
        apply = input("Would you like to apply a configuration? [y/N]: ").strip().lower()
        
        if apply == 'y':
            selected = interactive_mode(configs)
            if selected:
                generate_config_file(selected)
                
                apply_now = input("\nApply to config.py now? [y/N]: ").strip().lower()
                if apply_now == 'y':
                    if apply_configuration(selected):
                        success("✓ Configuration applied successfully!")
                        info("You can now run: python main.py --multi")
                else:
                    info("Configuration snippet generated above - copy to config.py manually")
    except KeyboardInterrupt:
        print()
        info("Cancelled")
    
    print()


if __name__ == "__main__":
    main()
