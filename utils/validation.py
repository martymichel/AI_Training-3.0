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

def detect_annotation_format(label_files: list) -> str:
    """Detect annotation format from label files.

    Args:
        label_files: List of label file paths

    Returns:
        str: 'bbox', 'polygon', 'mixed', or 'unknown'
    """
    has_bbox = False
    has_polygon = False

    for label_path in label_files[:10]:  # Check first 10 files for efficiency
        try:
            if not label_path.exists():
                continue

            with open(label_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    continue

                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    if len(parts) == 5:
                        has_bbox = True
                    elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                        has_polygon = True

                    # Early exit if both found
                    if has_bbox and has_polygon:
                        return 'mixed'
        except:
            continue

    if has_polygon:
        return 'polygon'
    elif has_bbox:
        return 'bbox'
    else:
        return 'unknown'

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

        # Check training data - train_path should be a .txt file, not a directory
        if not train_path.exists():
            error_msg = f"Training data file not found: {train_path}"
            return False, error_msg

        # Check if train.txt file has content
        try:
            with open(train_path, 'r') as f:
                train_content = f.read().strip()
                if not train_content:
                    error_msg = f"Training data file is empty: {train_path}"
                    return False, error_msg
        except Exception as e:
            error_msg = f"Error reading training data file {train_path}: {str(e)}"
            return False, error_msg

        # Check validation data - val_path should be a .txt file, not a directory
        if not val_path.exists():
            error_msg = f"Validation data file not found: {val_path}"
            return False, error_msg

        # Check if val.txt file has content
        try:
            with open(val_path, 'r') as f:
                val_content = f.read().strip()
                if not val_content:
                    error_msg = f"Validation data file is empty: {val_path}"
                    return False, error_msg
        except Exception as e:
            error_msg = f"Error reading validation data file {val_path}: {str(e)}"
            return False, error_msg

        # Verify image-label pairs exist by reading the train.txt and val.txt files
        # Parse train.txt to get image paths and check corresponding label files
        train_image_paths = []
        with open(train_path, 'r') as f:
            for line in f:
                img_path = line.strip()
                if img_path:
                    train_image_paths.append(Path(img_path))

        val_image_paths = []
        with open(val_path, 'r') as f:
            for line in f:
                img_path = line.strip()
                if img_path:
                    val_image_paths.append(Path(img_path))

        if not train_image_paths:
            error_msg = "No training images found in train.txt"
            return False, error_msg

        if not val_image_paths:
            error_msg = "No validation images found in val.txt"
            return False, error_msg

        # Get corresponding label files for verification
        train_labels = []
        val_labels = []

        for img_path in train_image_paths:
            # Convert image path to label path
            label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            train_labels.append(label_path)

        for img_path in val_image_paths:
            # Convert image path to label path
            label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            val_labels.append(label_path)
            
        # Detect annotation format first
        all_label_paths = train_labels + val_labels
        annotation_format = detect_annotation_format(all_label_paths)

        # Check for corresponding images and validate label files
        for i, (img_path, label_path) in enumerate(zip(train_image_paths + val_image_paths, train_labels + val_labels)):
            # Check if image file exists
            if not img_path.exists():
                error_msg = f"Image file not found: {img_path}"
                return False, error_msg

            # Check if label file exists
            if not label_path.exists():
                error_msg = f"Label file not found: {label_path}"
                return False, error_msg

            # Read and validate label file content
            try:
                with open(label_path) as f:
                    content = f.read().strip()
                    # Allow empty label files (background images)
                    if content:  # Only validate if file has content
                        # Validate each line
                        for line_num, line in enumerate(content.split('\n'), 1):
                            if line.strip():  # Skip empty lines
                                try:
                                    parts = line.strip().split()
                                    if len(parts) < 5:
                                        error_msg = f"Invalid format in {label_path} line {line_num}: Expected at least 5 values"
                                        return False, error_msg

                                    # Validate class ID
                                    class_id = int(float(parts[0]))

                                    if len(parts) == 5:
                                        # Bounding box format: class x_center y_center width height
                                        x, y, w, h = map(float, parts[1:])
                                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                            error_msg = f"Invalid bounding box coordinates in {label_path} line {line_num}: Values must be between 0 and 1"
                                            return False, error_msg
                                    elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                                        # Polygon/segmentation format: class x1 y1 x2 y2 x3 y3 ...
                                        # Check that coordinate count (excluding class) is even
                                        coords = list(map(float, parts[1:]))
                                        # Validate all coordinates are between 0 and 1
                                        for coord in coords:
                                            if not (0 <= coord <= 1):
                                                error_msg = f"Invalid polygon coordinates in {label_path} line {line_num}: All values must be between 0 and 1"
                                                return False, error_msg
                                        # Check minimum 3 points for polygon (6 coordinates)
                                        if len(coords) < 6:
                                            error_msg = f"Invalid polygon in {label_path} line {line_num}: Minimum 3 points (6 coordinates) required"
                                            return False, error_msg
                                    else:
                                        error_msg = f"Invalid format in {label_path} line {line_num}: Expected 5 values (bbox) or ≥7 with even coordinate count (polygon)"
                                        return False, error_msg

                                except ValueError as ve:
                                    error_msg = f"Invalid number format in {label_path} line {line_num}: {str(ve)}"
                                    return False, error_msg

            except Exception as e:
                error_msg = f"Error reading label file {label_path}: {str(e)}"
                return False, error_msg

            # Validate image can be opened
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    error_msg = f"Cannot read image file: {img_path}"
                    return False, error_msg
            except Exception as e:
                error_msg = f"Error validating image {img_path}: {str(e)}"
                return False, error_msg
            
        # Add format information to success message
        format_info = f" (Detected format: {annotation_format})"
        return True, format_info

    except Exception as e:
        error_msg = f"Error parsing YAML: {str(e)}"
        return False, error_msg

def validate_yaml_for_model_type(path: str, model_type: str = None) -> Tuple[bool, Optional[str]]:
    """Validate YAML dataset file with model type awareness.

    Args:
        path (str): Path to YAML file
        model_type (str): 'detection' or 'segmentation', auto-detected if None

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # First do basic YAML validation
    is_valid, message = validate_yaml(path)
    if not is_valid:
        return is_valid, message

    # Extract annotation format from message
    if "Detected format:" in message:
        detected_format = message.split("Detected format: ")[1].rstrip(")")

        # If model type is specified, check compatibility
        if model_type:
            if model_type.lower() == 'segmentation' and detected_format == 'bbox':
                warning_msg = (f"Warning: Segmentation model selected but only bounding box annotations detected. "
                             f"Segmentation models work best with polygon annotations, but can also train on bounding boxes.")
                return True, warning_msg
            elif model_type.lower() == 'detection' and detected_format in ['polygon', 'mixed']:
                warning_msg = (f"Warning: Detection model selected but polygon annotations detected. "
                             f"Consider using a segmentation model for better results with polygon annotations.")
                return True, warning_msg

    return True, f"YAML validation successful. {message}"

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