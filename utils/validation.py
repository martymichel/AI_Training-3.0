"""Validation utilities for the YOLO trainer."""

import os
import yaml
import torch
from pathlib import Path 
import logging
import traceback
# Disable debug logging for validation
logging.getLogger().setLevel(logging.WARNING)

from typing import Tuple, Optional

def validate_yaml(path: str) -> Tuple[bool, Optional[str]]:
    """Validate YAML dataset file.
    
    Args:
        path (str): Path to YAML file
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not os.path.exists(path):
        return False, f"File not found: {path}"
        
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
            # Validate required keys
        required_keys = ['path', 'train', 'val', 'names']
        missing = [k for k in required_keys if k not in data]
        if missing:
            error_msg = f"Missing required keys in YAML: {', '.join(missing)}"
            return False, error_msg

            # Convert paths to absolute using YAML file location as base
            yaml_dir = Path(path).parent
            data_path = Path(data['path'])
            if not data_path.is_absolute():
                data_path = yaml_dir / data_path

            # Check if train and val paths exist
            train_path = Path(data['train'])
            val_path = Path(data['val'])
            
            # Convert relative paths to absolute
            if not train_path.is_absolute():
                train_path = data_path / train_path.name
            if not val_path.is_absolute():
                val_path = data_path / val_path.name

            # Check if paths exist and contain data
            if not data_path.exists():
                error_msg = f"Dataset path not found: {data_path}"
                return False, error_msg

            # Check training data
            if not train_path.exists() or not any(train_path.glob('*.txt')):
                error_msg = f"Training data not found or empty: {train_path}"
                return False, error_msg

            # Check validation data
            if not val_path.exists() or not any(val_path.glob('*.txt')):
                error_msg = f"Validation data not found or empty: {val_path}"
                return False, error_msg

            # Verify image-label pairs exist
            train_labels = list(train_path.glob('*.txt'))
            val_labels = list(val_path.glob('*.txt'))
            
            if not train_labels:
                error_msg = "No training labels found"
                return False, error_msg
                
            if not val_labels:
                error_msg = "No validation labels found"
                return False, error_msg
                
            # Check for corresponding images
            for label_path in train_labels + val_labels:
                img_stem = label_path.stem
                # Read and validate label file content
                try:
                    with open(label_path) as f:
                        content = f.read().strip()
                        if not content:  # Skip empty files
                            error_msg = f"Empty label file found: {label_path}"
                            return False, error_msg
                        
                        # Validate each line
                        for line_num, line in enumerate(content.split('\n'), 1):
                            try:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    error_msg = f"Invalid format in {label_path} line {line_num}: Expected 5 values (class x y w h)"
                                    return False, error_msg
                                
                                # Validate values
                                class_id = int(float(parts[0]))
                                x, y, w, h = map(float, parts[1:])
                                
                                # Check ranges
                                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                    error_msg = f"Invalid coordinates in {label_path} line {line_num}: Values must be between 0 and 1"
                                    return False, error_msg
                                    
                            except ValueError as ve:
                                error_msg = f"Invalid number format in {label_path} line {line_num}: {str(ve)}"
                                return False, error_msg
                                
                except Exception as e:
                    error_msg = f"Error reading label file {label_path}: {str(e)}"
                    return False, error_msg

                img_found = False
                for ext in ['.jpg', '.jpeg', '.png']:
                    if label_path.with_name(f"{img_stem}{ext}").exists():
                        img_found = True
                        # Validate image can be opened
                        try:
                            import cv2
                            img = cv2.imread(str(label_path.with_name(f"{img_stem}{ext}")))
                            if img is None:
                                error_msg = f"Cannot read image file: {img_stem}{ext}"
                                return False, error_msg
                        except Exception as e:
                            error_msg = f"Error validating image {img_stem}{ext}: {str(e)}"
                            return False, error_msg
                        break
                if not img_found:
                    error_msg = f"No matching image found for label: {label_path}"
                    return False, error_msg
            
        return True, None
        
    except Exception as e:
        error_msg = f"Error parsing YAML: {str(e)}"
        return False, error_msg

def check_gpu() -> Tuple[bool, str]:
    """Check GPU availability and CUDA status.
    
    Returns:
        Tuple[bool, str]: (gpu_available, message)
    """
    try:
        # Check PyTorch version
        torch_version = torch.__version__
        
        # Check CUDA environment variables
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible == '-1':
            return False, "CUDA ist durch Umgebungsvariable deaktiviert"

        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            return False, (
                f"CUDA nicht verfügbar.\n"
                f"PyTorch Version: {torch_version}\n"
                f"CUDA_VISIBLE_DEVICES: {cuda_visible}\n"
                "Training wird CPU nutzen (langsam)."
            )

        # Try to initialize CUDA to force detection
        torch.cuda.init()
        
        # Get device information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        cuda_version = torch.version.cuda
        message = (
            f"GPU gefunden: {gpu_name} ({memory:.1f}GB)\n"
            f"Anzahl GPUs: {gpu_count}\n"
            f"CUDA Version: {cuda_version}\n"
            f"PyTorch Version: {torch_version}\n"
            f"CUDA_VISIBLE_DEVICES: {cuda_visible}"
        )
        return True, message
        
    except Exception as e:
        error_msg = (
            f"Fehler bei GPU-Erkennung: {str(e)}\n"
            f"PyTorch Version: {torch_version}\n"
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'nicht gesetzt')}\n"
            f"PyTorch CUDA verfügbar: {torch.cuda.is_available()}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        return False, error_msg