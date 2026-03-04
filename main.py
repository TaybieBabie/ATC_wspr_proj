import os
import ctypes
import importlib

def preload_cudnn():
    """Load cuDNN shared libraries before ctranslate2 tries to find them."""
    # Try to find the nvidia.cudnn pip package
    try:
        nvidia_cudnn = importlib.import_module("nvidia.cudnn")
        cudnn_lib_dir = os.path.join(os.path.dirname(nvidia_cudnn.__file__), "lib")
    except ImportError:
        # Fallback: hardcode the path
        cudnn_lib_dir = os.path.expanduser(
            "~/.local/lib/python3.13/site-packages/nvidia/cudnn/lib"
        )

    if not os.path.isdir(cudnn_lib_dir):
        print(f"WARNING: cuDNN lib directory not found at {cudnn_lib_dir}")
        return

    # These must be loaded in dependency order
    libs_to_load = [
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_graph.so.9",
        "libcudnn.so.9",
    ]

    for lib_name in libs_to_load:
        lib_path = os.path.join(cudnn_lib_dir, lib_name)
        if os.path.exists(lib_path):
            try:
                ctypes.cdll.LoadLibrary(lib_path)
                print(f"  Loaded: {lib_name}")
            except OSError as e:
                print(f"  FAILED to load {lib_name}: {e}")
        else:
            print(f"  Not found: {lib_path}")

print("Pre-loading cuDNN libraries...")
preload_cudnn()

# ---- NOW import everything else ----
import argparse
import json
import os
import threading
import time
from core.multi_channel_monitor import MultiChannelATCMonitor
from utils import config
from utils.console_logger import section


def load_channel_config(config_file):
    """Load channel configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)


def load_location_config(config_file):
    """Load location configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    print(r"""
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣤⣤⣴⣦⣶⣤⣤⣀⣀⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣾⠿⠿⣶⣦⣄⡀⠀⢀⣠⣴⣶⠿⠿⠛⠋⠉⠉⠉⠉⠀⠀⠈⠉⠛⠋⠉⠉⠙⠻⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⠏⠀⣠⣄⡀⠈⠙⣿⡾⠟⠋⠁⢀⣀⡤⠶⠖⠛⠛⠲⠞⠋⠉⠛⠛⠒⠲⢿⣷⣦⣄⠀⠉⠻⣷⣶⣤⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⡶⠾⠾⠿⠃⢀⣼⣿⣻⣿⡏⠀⠉⣀⣤⠶⠞⠉⢁⠠⠀⠄⡒⠀⠄⠂⠌⢀⠡⠐⠀⠀⠀⠈⠙⢿⣿⣦⣀⠀⠀⠈⠙⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⡿⠟⠃⠀⣀⣀⡀⠀⠀⣼⡿⣧⣿⣿⠇⢀⡼⠛⠀⠀⡀⠘⢀⠀⠄⢠⠃⠠⢀⠘⠀⠄⠀⠄⡀⠀⠀⠀⠀⠀⠸⢿⣿⣧⡀⠛⢧⡀⠘⢿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⡿⠋⠀⡠⠞⠋⠀⠀⠀⢀⣾⣿⣽⣿⣿⡿⣿⠋⠀⠄⠂⢁⠠⠈⡀⠄⢈⡏⢀⠐⠠⠀⡁⠂⢁⠠⠀⠀⠀⠀⢀⠀⡀⠈⠹⣟⣿⣦⠀⢻⡄⠈⢿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡟⠀⣠⠞⠀⡀⠀⠀⠀⠀⣾⣿⣳⣿⡿⢃⡾⠁⠀⠀⠂⢈⠀⢄⠞⠀⡐⣘⠀⠄⠂⠠⠁⡀⠌⠀⠀⠀⠀⠀⠀⠀⠁⠀⠐⡀⠙⣿⣿⠀⡀⢿⠀⠈⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡟⠀⢰⢏⣾⣿⣶⣎⣠⣤⣼⣿⣿⣿⡿⠁⣼⠃⠀⠀⠀⠀⠀⣀⡞⠀⡐⠀⣷⠀⠀⢈⠀⠐⠀⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⢂⠠⠀⠸⡟⢀⠐⣸⡇⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⠃⢀⣿⣾⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⢃⢰⡇⠀⠀⠀⠀⠀⠀⡼⠀⠀⠄⠂⣽⣄⠀⠀⠀⢀⠀⠙⠧⣄⠀⠠⠀⠄⢀⠠⠐⠀⡀⢧⡐⠀⢂⠰⢸⡇⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣄⣀⣼⡟⠀⣸⣟⣿⣽⣿⣟⠻⢿⣿⣼⣿⣿⣿⢊⣼⢡⠀⠀⠀⠀⠀⣸⠇⠀⠀⠐⢀⡇⠸⣦⡀⢈⠀⠀⠀⣇⠈⠛⣦⣐⠈⠀⠄⠂⡐⠀⡀⢧⠐⡀⢎⣹⠇⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⠿⡷⡟⠿⢋⠿⢿⡿⠛⠀⢀⣼⠟⢸⣿⣿⢥⡟⣿⠀⠀⠀⠀⣰⠏⠀⡀⠄⠀⣸⠁⠀⡼⣿⣦⡀⠡⠀⠹⡄⠀⠀⠽⢷⣌⠀⠂⠄⠂⡀⠘⡄⠸⣄⢻⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⣿⣿⣧⣛⠴⣩⠝⡊⠅⠀⠀⠀⣠⣿⠟⠀⣼⣿⣿⡎⣽⣿⠀⠀⠠⣱⠏⠀⢂⢠⠀⢠⣿⣀⡀⠀⠜⣿⣷⣦⡐⠀⢻⡄⠀⠀⣹⣿⣷⣅⠐⠠⢀⠁⡸⢐⠬⣻⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⢀⣠⣶⡿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠗⡌⠀⠀⠀⣠⣾⣿⠋⠀⢸⡇⠀⢸⣷⣾⣿⠇⠀⢡⡟⠀⠌⠀⡎⠀⣼⠋⠈⠉⠓⠲⢿⣿⣿⣿⣄⠈⣧⠀⠚⠁⠺⣹⢿⣷⡆⡄⢂⡘⡇⢎⣽⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⣠⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣭⣛⢩⠁⠀⠀⠀⣠⣾⣿⡟⠁⠀⣰⣾⡇⠀⠈⣿⣿⣿⠁⢨⡿⠁⠐⠠⢁⡇⢐⣿⠀⠀⣄⠀⠀⠈⢻⣿⣿⣿⣇⢾⡇⠀⠀⣀⣼⣿⣿⣿⣷⢆⡜⣹⢂⣿⡄⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⢿⣿⣷⣄⠀⢀⣴⣿⡿⠋⠀⣰⣴⣿⣿⣻⡶⢤⣼⣿⡿⢀⣿⠃⢀⠁⠂⢸⣧⢨⣿⣿⣶⣦⣄⠀⠀⠀⠻⣿⣿⣿⣿⡇⢠⣾⣿⣿⡽⢿⣿⣿⣿⡜⣳⠐⣾⡇⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣟⣛⠻⢿⣷⣝⢿⣿⣧⣾⣿⠟⠁⠀⠀⣿⣿⣿⣋⣼⣷⡛⡟⠿⣿⣼⣟⠀⠤⠈⢀⣿⣿⣸⣿⣿⡟⣻⠝⢧⠀⠀⠀⠈⠻⣿⣿⢷⣿⣿⣿⣼⡇⠈⣿⣿⣿⣞⠱⠈⣾⡇⠀⢻⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⢸⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣷⣤⠙⣿⣧⢿⣿⣿⠃⠀⡀⠀⠀⣿⡟⣿⣧⣿⣿⣿⣼⢲⡌⡽⣿⣤⠀⢂⢰⣿⣿⣿⠿⡿⠛⣿⠄⠀⠀⠀⠀⠀⠀⠈⠹⠎⢯⣙⠉⣸⠃⠸⢫⣿⣿⣯⠆⠁⢾⡇⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⢸⣿⣿⣿⣿⡏⣼⣿⣿⣿⣿⣿⣿⣿⣇⠸⣿⣾⣿⡇⠀⠠⠆⣠⣼⣿⢀⣿⣿⣿⣿⣿⣿⣿⣼⠡⠎⠻⣿⣤⣼⡗⠻⠻⠦⣤⠴⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠛⠁⠀⠀⣾⠘⣿⠛⡇⢈⢚⡇⠀⢸⣯⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⢸⣿⣿⣿⣿⡅⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⡇⠀⢸⣿⣿⢿⣧⣼⣿⣿⣿⣿⣿⣿⣿⣿⣧⠉⠋⠙⠙⢻⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⡁⠠⠁⠂⡸⣿⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⢸⣿⣿⡽⣿⣧⠹⣿⣿⣿⣿⣿⣿⣿⠏⣼⣿⣽⣿⡇⣰⠏⢸⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⡿⢛⠷⠈⠀⠀⠀⠀⠹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⢤⣿⡇⠐⠀⡁⠄⢁⠰⣿⠀⠈⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠘⢿⣿⣿⣿⣿⣷⣬⣙⡛⠿⣻⣿⣫⣾⣿⣏⣿⣿⡶⠋⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⠃⠀⠀⠀⠀⠀⠀⢠⣿⣟⣳⣄⠀⠀⠀⠀⠀⠠⣴⣶⣶⣶⣾⠗⠀⠀⠀⣴⣿⣿⣿⠀⠂⠠⠐⢀⠐⣿⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠻⣿⣼⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⢿⠏⠀⣴⢿⡝⠁⠈⠉⠙⢿⣿⣿⣿⣿⣟⠀⠀⠀⠀⠀⠀⠀⣀⣼⣯⣿⡟⠻⣷⣄⡀⠀⠀⠀⠈⠉⠉⠉⠁⠀⢀⣴⣾⣿⣿⣿⣿⡇⠀⡁⠐⡀⠂⣽⡇⠀⢿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠐⣿⡧⣿⡿⢿⣻⣯⡿⣿⣯⣿⣾⢧⡟⠃⢠⣼⡟⡎⠀⠉⠉⠙⠛⠳⡞⣿⣿⣿⡇⠀⢀⣤⣤⣀⠀⣴⣿⡿⠟⠉⠙⢦⣽⣿⠿⢦⣤⣀⠀⠀⠀⠀⣰⠛⢻⣿⣿⣿⣿⣿⣿⣷⠀⠄⢁⠠⠐⡸⡇⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⣿⣷⣉⣿⣿⣿⣿⣿⣶⡝⣷⢿⣾⠃⣠⣾⣿⢇⡟⢀⡀⠐⠀⠀⠀⣇⢸⣿⣿⠇⠐⠋⠀⠀⢘⣿⡿⠋⠀⣠⡆⠀⣾⡟⣿⣿⣾⣿⣿⣿⣷⣾⣿⢿⡇⢸⣿⣿⣿⣿⣿⣿⡿⠀⢈⠀⠄⠂⠱⣿⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⣿⣿⣿⣟⣿⣿⣿⣿⣿⣿⢸⣿⣿⢠⣿⣿⣿⢸⣷⡸⢿⣦⣄⠀⣰⣇⣾⣿⣿⠂⠈⡀⠀⠀⣸⠋⠀⣠⣾⠟⠀⢠⡟⠻⠿⠛⠋⠀⠉⠉⠙⠿⠿⣦⡁⢿⣿⣿⣿⣿⣿⣿⡇⠀⢂⠀⠂⠌⠐⣿⠀⠘⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⣽⣿⣿⣿⢹⣿⣿⣿⣿⡿⣸⣿⣏⣾⣿⣿⡯⣾⡿⠷⢮⣽⣿⣶⣿⠿⠻⡉⠟⣶⣀⠑⠤⠀⡿⠲⣶⡿⠋⠀⣀⠾⠁⠀⠀⠀⠀⠀⠀⠀⢰⡤⠴⠟⢧⡌⠻⢿⣿⣿⣿⣿⠃⠀⠌⠐⡀⠄⠐⣻⡀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠠⣿⣿⣯⠻⣷⣍⣛⣛⢻⣴⣿⠟⣱⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠠⣅⣜⠠⠙⣿⣶⣄⣼⢧⡀⠈⠀⣀⣾⣟⣓⠂⠄⠀⠀⠀⠀⠀⠀⠘⠷⣤⣀⢨⡿⢷⣤⡈⠛⠛⠉⠋⠻⠶⣶⣦⣄⠂⢹⡇⠀⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⢻⣿⣿⣷⣮⣙⡛⢛⢻⣻⣥⣿⣿⣿⣿⠟⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⣠⣿⣿⣿⣿⡄⠙⣄⠚⠉⠁⠀⠉⠙⣆⠈⠐⠠⠀⠀⠀⠀⠀⠈⢷⡄⠘⢦⡉⠉⠳⢦⡙⢦⠀⠀⠀⠉⠺⡛⢶⣿⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢁⠠⢠⠀⡄⢠⠀⠠⢀⠀⠀⠀⠀⠀⢀⡞⠙⣿⣿⣿⣿⡄⠹⡆⠀⠀⠀⠀⠀⢯⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡄⢀⠳⣤⠀⠀⠁⠈⠳⡀⠀⠀⠀⠈⢢⡻⡄⠸⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣤⢃⠦⣡⠒⠤⠘⠠⠀⠀⠀⠀⣀⣤⠟⠁⢸⣿⣿⣿⣿⣿⡀⢹⡄⠀⠀⠀⠀⠩⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣆⠀⡈⠳⣄⠀⠀⠀⠙⢦⡀⡀⠀⠀⠙⣇⠀⠈⢻⡿⢿⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠶⠿⠾⠷⠷⠶⠾⠟⠛⠉⠁⠀⡘⣿⣿⣿⣿⣿⣿⡇⠘⣧⠀⠀⠀⠀⢸⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣧⡀⠐⠘⢷⣄⠉⠦⠿⣇⠁⠀⠀⠀⠈⠳⢦⣀⣧⡀⠈⠻⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡼⢅⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⣼⣿⣿⣿⣿⣿⣿⡿⢀⣿⠀⠀⠀⠀⢠⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣷⡈⢀⠂⠙⢿⡶⠶⠿⠶⢦⡀⠀⠀⠀⠀⠀⠙⣿⣦⠀⠙⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣈⠰⢠⢂⠠⢀⡀⠀⠀⠀⠀⠀⢀⣡⣿⣿⣿⣿⣿⣿⣿⡇⢸⡇⠀⠀⠀⠀⢸⡇⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡆⠠⠁⡈⠱⡆⠀⠀⠲⣕⡀⠀⠀⠀⠀⢀⡟⣿⣷⡀⠈⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢦⡙⡆⢬⡑⢢⠠⣉⣄⣠⡴⠞⠋⣿⣿⣿⣿⣿⣿⡟⠀⡾⠀⠀⠀⠀⠀⣼⠄⢰⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⡀⠐⡀⠁⠹⡄⠀⠀⠘⢿⡄⠀⠀⣠⡾⢁⣿⣿⣿⣦⡀⠨⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠈⣿⠛⣿⣿⣿⣿⡟⡟⠛⠛⠛⠿⠛⠿⠛⠛⠟⠛⠉⠁⠠⣄⣾⣿⣿⣿⣿⣿⣿⡟⠁⣼⠁⠀⠀⠀⠀⢸⡏⠀⣾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡇⠠⢀⠁⠂⢿⠀⠀⠀⠀⣿⣦⡾⢋⣴⣿⣿⣿⣿⣿⣿⣦⡀⠈⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⣘⣿⣿⣿⣿⡱⠡⠌⢂⣄ ⠀⠄⢂⡀⠀⠀⠄⣱⣾⣿⣿⣿⣿⣿⡿⠋⣠⢾⡇⠀⠀⠀⠀⢠⣞⢁⣸⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠐⡀⠈⠄⡘⣇⠀⠀⠀⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠈⢻⣧⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⢸⡏⠀⢹⣿⣿⣿⣿⣇⠻⣌⠲⣌⠷⢬⠐⠠⢀⠂⣀⣴⣿⣿⣿⣿⠿⠛⢁⣤⠞⠁⠈⣿⣄⠀⠀⢀⣾⠇⣼⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠐⡀⠡⠀⠄⢿⠀⠀⠀⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠈⣿⡆⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⣿⡇⠀⡿⠈⠻⣿⡿⢿⢷⣾⣿⣾⣽⣦⣭⣶⠟⠛⠛⠛⠛⢉⣉⣠⡴⠞⠋⠀⣠⣶⡀⠘⢿⣦⣶⡿⠋⠚⠛⠓⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢨⡗⠠⠐⠀⠡⠀⢼⡆⠀⠀⢹⣿⡏⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢰⣿⠁⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⢰⣿⠀⢰⡇⠠⢀⠀⢀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣾⠛⠛⠉⠁⣀⣤⣶⠿⠛⠻⣿⡀⠸⣿⡿⡖⠒⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⢀⠂⢁⠂⢁⠀⣧⠀⠀⠸⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⣼⡏⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⣸⡏⠀⣸⠀⠠⠀⠌⠀⠠⠈⢀⠐⠀⠂⠀⠈⠉⢉⣿⣿⣿⠀⢰⣾⠿⠛⠉⠀⠀⠀⠀⢿⣧⠀⢸⣷⣷⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣇⠀⡐⠠⠐⠀⢂⢻⠀⠀⢸⣿⣿⣿⠟⣻⣿⣿⣿⣻⡟⣿⣿⡟⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⣿⡇⠀⣿⠀⡀⢁⠂⠈⠀⠄⠂⠀⠌⠀⠁⠀⠀⣾⣿⣿⣿⠀⠈⣿⠄⠀⠀⠀⠀⠀⠀⠩⣿⣆⠀⢻⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠠⠐⡀⠡⢀⢸⡆⠀⢸⣿⣿⣥⣿⣿⣿⣿⣿⣿⡿⠈⣿⠀⢠⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⢀⣿⠀⢰⡇⠀⠄⡀⠂⢁⠀⠂⢀⠐⠀⣈⣤⣄⣾⣿⣿⣿⣿⡄⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠸⣿⡄⠈⢿⣧⠤⠀⠀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠡⠐⢀⠐⡀⠈⣷⠀⢸⣿⣿⣿⣿⣿⣿⣿⡿⡿⠁⣰⠇⠀⣾⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⣸⡟⠀⢸⠃⢀⠂⠠⠐⠀⡀⠌⠀⠀⠄⣾⣿⣿⣿⢿⣿⣿⣿⡇⠀⢻⡇⠀⠀⠀⠀⠀⠀⠀⠀⢻⣷⡀⠘⣿⣆⠀⢀⠠⠐⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡅⠂⢈⠠⠀⠄⡁⠸⡆⢸⣿⣿⣿⣿⣿⣿⡿⢱⣇⣀⡟⠀⣼⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⣿⡇⠀⡿⠀⠠⠀⡁⢀⠂⠀⠠⠐⠈⠀⣿⣿⣿⣯⣿⣿⣿⣿⡇⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣧⠀⢹⣿⡄⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡇⠀⢂⠠⠈⡀⠄⠁⢿⣻⣿⣿⣿⣿⣿⡿⢠⣟⣼⡿⠁⢰⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⢠⣿⠀⢰⡇⠀⡁⠠⠐⠀⠀⠀⠂⠀⠄⣠⣿⣿⣿⣟⣿⣿⣿⣿⣿⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡄⠀⢻⣧⠀⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣻⣇⠈⡀⠄⠂⡀⠂⢁⠘⣿⣿⣿⣿⣿⣿⣥⡟⡴⣼⠃⢀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⢸⣿⠀⢸⡇⠀⡄⠐⢠⠀⠁⠀⡄⣴⣾⣿⣿⣿⣿⣿⢻⣿⣿⣿⣿⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⡄⠈⣿⡆⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⠀⡄⠂⠁⢠⠈⢠⠀⢹⣿⣿⣿⢻⣿⣾⢳⢳⡏⠀⣼⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⢀⣾⡇⠀⣾⣄⣀⣄⣐⣀⣀⣀⣠⣾⣿⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⡀⢸⣿⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣧⠀⣹⣿⣤⣀⣀⣀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⣀⣀⣸⣿⣀⣠⣔⣈⣤⣐⣀⣈⣀⣈⣹⣿⣿⣾⣷⣏⣾⠁⣠⣿⡃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠈⠀⠈⠁⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠈⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

                88     888888 888888 .dP"Y8     8b    d8 888888 88     888888     88    dP""b8   888888 
                88     88__     88   `Ybo."     88b  d88 88__   88       88       88   dP   `"   88__   
                88  .o 88""     88   A `Y8b     88Y\/P88 88""   88  .o   88       88   Yb        88""   
                88ood8 888888   88   8bodP'     88 Y7 88 888888 88ood8   88       88 0  YioodP 0 888888 0                                                   ⠀⠀⠀⠀⠀⠀⠀
            """)

    time.sleep(3.72)

    parser = argparse.ArgumentParser(description='ATC Communication Monitor')
    parser.add_argument('--multi', action='store_true', help='Start multi-channel monitoring')
    parser.add_argument('--channels', type=str, help='Path to channels configuration JSON file')
    parser.add_argument('--location', type=str, help='Path to location configuration JSON file')
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of transcription workers (default: {config.NUM_TRANSCRIPTION_WORKERS} from config)'
    )
    args = parser.parse_args()

    location_config = None
    if args.location:
        location_config = load_location_config(args.location)
    elif args.multi and not args.channels:
        default_location = "location_pdx.json"
        if os.path.exists(default_location):
            location_config = load_location_config(default_location)

    if location_config:
        config.apply_location_settings(location_config)

    if args.multi:
        section("Multi-Channel ATC Monitor", emoji="✈️")

        # Load channel configuration
        if args.channels:
            channels = load_channel_config(args.channels)
        elif location_config and location_config.get("channels"):
            channels = location_config["channels"]
        else:
            # Default PDX area channels
            channels = [
                {
                    "name": "PDX Tower",
                    "frequency": "118.7",
                    "stream_url": "https://s1-fmt2.liveatc.net/kpdx_twr",
                    "color": "#FF0000"
                },
                {
                    "name": "PDX Approach West",
                    "frequency": "124.35",
                    "stream_url": "https://s1-fmt2.liveatc.net/kpdx_app_124350",
                    "color": "#00FF00"
                },
                {
                    "name": "PDX Approach East",
                    "frequency": "118.1",
                    "stream_url": "https://s1-fmt2.liveatc.net/kpdx_app_118100",
                    "color": "#0000FF"
                },
                # Add more channels as needed
            ]

        # Create multi-channel monitor
        # If --workers specified, use it; otherwise uses NUM_TRANSCRIPTION_WORKERS from config
        num_workers = args.workers if args.workers is not None else config.NUM_TRANSCRIPTION_WORKERS
        atc_monitor = MultiChannelATCMonitor(channels, num_transcription_workers=num_workers)

        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=atc_monitor.start_monitoring)
        monitor_thread.start()

        # Run GUI
        from gui.map_app_webview import run_webview_app
        run_webview_app(atc_monitor)
        monitor_thread.join()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
