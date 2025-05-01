import torch
import torchvision
import ultralytics
import sys
import platform
import psutil
import os
import traceback

def get_system_info():
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'not set')}")
    print(f"PATH: {os.environ.get('PATH', 'not set')}")

def get_cuda_info():
    print("\n=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        print(f"PyTorch version: {torch.__version__}")
        print("No CUDA available")
        try:
            torch.cuda.init()
        except Exception as e:
            print(f"CUDA initialization error: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

def get_package_versions():
    print("\n=== Package Versions ===")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"ultralytics: {ultralytics.__version__}")

if __name__ == "__main__":
    get_system_info()
    get_cuda_info()
get_package_versions()