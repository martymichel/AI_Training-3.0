"""Optimized YOLO training module."""

import torch
import os
import sys
from ultralytics import YOLO
import traceback
import logging
import platform
from pathlib import Path

# Configure logging
logger = logging.getLogger("yolo_training")
logger.setLevel(logging.INFO)

def setup_logging():
    """Set up logging with proper formatting."""
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def get_optimal_workers():
    """Get optimal number of workers based on system."""
    cpu_count = os.cpu_count() or 1
    # Leave some cores free for system
    return max(1, min(8, cpu_count - 2))

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
        logger.warning(f"Error getting GPU info: {e}")
        return 'cuda:0', 0.5

def start_training(data_path, epochs, imgsz, batch, lr0, resume, multi_scale, cos_lr, close_mosaic,
                   momentum, warmup_epochs, warmup_momentum, box, dropout, project, name,
                   progress_callback=None, log_callback=None):
    """Start YOLO training with GPU support."""
    try:
        setup_logging()
        
        # Get optimal device and batch settings
        device, batch_scale = get_device_settings()
        workers = get_optimal_workers()
        
        # Adjust batch size based on GPU memory
        if batch > batch_scale:
            logger.warning(
                f"Reducing batch size from {batch} to {batch_scale} "
                "due to GPU memory constraints"
            )
            batch = batch_scale

        logger.info(f"Using device: {device}")
        logger.info(f"Batch size: {batch}")
        logger.info(f"Workers: {workers}")
        
        # Initialize model
        model = YOLO("yolov8n.pt")  # Use standard YOLOv8 nano model
        
        # Start training
        model.train(
            resume=resume,
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            optimizer="AdamW",  # Fixed default
            device=device,
            project=project,
            name=name,
            workers=workers,
            exist_ok=True,
            plots=True,  # Fixed default
            multi_scale=multi_scale,
            cos_lr=cos_lr,
            close_mosaic=close_mosaic,
            momentum=momentum,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            box=box,
            dropout=dropout,
            seed=42  # Fixed seed for reproducibility
        )
        
        if progress_callback:
            progress_callback(100)
    
    except torch.cuda.OutOfMemoryError:
        error_msg = (
            "GPU memory exhausted! The batch size has been automatically reduced. "
            "If the error persists, try reducing the image size."
        )
        logger.error(error_msg)
        if progress_callback:
            progress_callback(0, error_msg)
    except Exception:
        error_msg = f"Training error:\n{traceback.format_exc()}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback(0, error_msg)