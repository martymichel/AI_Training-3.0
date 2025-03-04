import torch
import torchvision
import ultralytics
import sys
import platform
import psutil
import os

def get_system_info():
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

def get_cuda_info():
    print("\n=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

def get_package_versions():
    print("\n=== Package Versions ===")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"ultralytics: {ultralytics.__version__}")

if __name__ == "__main__":
    get_system_info()
    get_cuda_info()
    get_package_versions()

#### PC Output ####
"""
PS P:\PY\AI_Training 2.0> uv run check_versions.py

=== System Information ===
Python version: 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)]
Platform: Windows-11-10.0.26100-SP0
CPU cores: 16
RAM: 63.9 GB

=== CUDA Information ===
CUDA available: True
CUDA version: 12.6
GPU device: NVIDIA GeForce RTX 3070 Ti
GPU memory: 8.0 GB
cuDNN version: 90501
cuDNN enabled: True

=== Package Versions ===
torch: 2.6.0+cu126
torchvision: 0.21.0+cu126
ultralytics: 8.3.78
"""

#### Notebook Output ####
"""
PS C:\Users\miche\Documents\GitHub\AI_Training-3.0> uv run check_versions.py

=== System Information ===
Python version: 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)]
Platform: Windows-11-10.0.22631-SP0
CPU cores: 22
RAM: 23.4 GB

=== CUDA Information ===
CUDA available: True
CUDA version: 12.6
GPU device: NVIDIA GeForce RTX 4050 Laptop GPU
GPU memory: 6.0 GB
cuDNN version: 90501
cuDNN enabled: True

=== Package Versions ===
torch: 2.6.0+cu126
torchvision: 0.21.0+cpu
ultralytics: 8.3.78
"""