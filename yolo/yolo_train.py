"""Optimized YOLO training module."""

import torch
import os
import sys
from ultralytics import YOLO
import traceback
import logging
import platform
from pathlib import Path
import time

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
        if gpu_mem >= 64:  # High-end GPUs (>= 10GB)
            return 'cuda:0', 1.0
        elif gpu_mem >= 16:  # Mid-range GPUs (6-8GB)
            return 'cuda:0', 0.8
        else:  # Low memory GPUs
            return 'cuda:0', 0.5
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
        
        # Initialize model VERSION 11 - newest version
        model = YOLO("yolo11n.pt")  # Use standard YOLOv8 nano model
        
        # Log progress using the callback
        if log_callback: 
            log_callback("Starting training with the following parameters:")
            log_callback(f"Data path: {data_path}")
            log_callback(f"Epochs: {epochs}")
            log_callback(f"Image size: {imgsz}")
            log_callback(f"Batch: {batch}")
            log_callback(f"Learning rate: {lr0}")
            log_callback(f"Device: {device}")

        # Check if resume is requested and checkpoint exists
        if resume:
            ckpt_path = Path(project) / name / "weights" / "last0980/1..2.pt"
            if not ckpt_path.is_file():
                logger.warning(f"No checkpoint found at {ckpt_path}, starting new training.")
                if log_callback:
                    log_callback(f"Checkpoint not found at {ckpt_path}. Starting new training.")
                resume = False            
        
        # Start training - REMOVED callbacks parameter as it's not supported
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
        
        # Manual progress tracking 
        if progress_callback:
            progress_callback(100, "Training completed successfully")
        
        if log_callback:
            log_callback("Training completed successfully")
    
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