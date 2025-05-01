import torch
import torchvision
import torchaudio

'''
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
'''
def get_device_settings():
    """Get optimal device and batch settings."""
    if not torch.cuda.is_available():
        return 'cpu', 1

    try:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        # Adjust batch size based on GPU memory
        if gpu_mem >= 10:  # High-end GPUs (>= 10GB)
            return 'cuda:0', 1.0
        elif gpu_mem >= 6:  # Mid-range GPUs (6-8GB)
            return 'cuda:0', 0.5
        else:  # Low memory GPUs
            return 'cuda:0', 0.3
    except Exception as e:
        return 'cuda:0', 0.5


def main():
    # Überprüfen, ob CUDA verfügbar ist
    if torch.cuda.is_available():
        print("CUDA ist verfügbar!")
        device = torch.device("cuda")
    else:
        print("CUDA ist nicht verfügbar.")
        device = torch.device("cpu")

    # Beispiel für die Verwendung von PyTorch
    x = torch.randn(3, 3).to(device)
    print(x)

    dev, batch_scale = get_device_settings()
    print(f"Optimaler Gerätetyp: {dev}")
    print(f"Optimaler Batch-Skalierungsfaktor: {batch_scale}")

if __name__ == "__main__":
    main()
