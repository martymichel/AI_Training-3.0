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
                   model_path="yolo11n.pt", progress_callback=None, log_callback=None):
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
        model = YOLO(model_path)  # Use specified model
        logger.info(f"Loaded model: {model_path}")
        
        # Log progress using the callback
        if log_callback: 
            log_callback("Starting training with the following parameters:")
            log_callback(f"Data path: {data_path}")
            log_callback(f"Epochs: {epochs}")
            log_callback(f"Image size: {imgsz}")
            log_callback(f"Batch: {batch}")
            log_callback(f"Learning rate: {lr0}")
            log_callback(f"Device: {device}")
            log_callback(f"Model: {model_path}")

        # Check if resume is requested and checkpoint exists
        if resume:
            ckpt_path = Path(project) / name / "weights" / "last.pt"
            if not ckpt_path.is_file():
                logger.warning(f"No checkpoint found at {ckpt_path}, starting new training.")
                if log_callback:
                    log_callback(f"Checkpoint not found at {ckpt_path}. Starting new training.")
                resume = False            
        
        # Detect if segmentation model
        is_segmentation = 'seg' in model_path.lower()

        # Base training parameters
        train_args = {
            'resume': resume,
            'data': data_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'lr0': lr0,
            'optimizer': 'auto',  # Let YOLO11 choose automatically
            'device': device,
            'project': project,
            'name': name,
            'workers': workers,
            'exist_ok': True,
            'plots': True,
            'save': True,
            'val': True,
            'verbose': True,
            # Loss weights (YOLO11 defaults)
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Classification loss gain
            'dfl': 1.5,  # Distribution focal loss gain
            # Training hyperparameters
            'cos_lr': cos_lr,
            'close_mosaic': close_mosaic,
            'momentum': momentum,
            'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            # Augmentation parameters
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation augmentation
            'degrees': 0.0,   # Image rotation (+/- deg)
            'translate': 0.1, # Image translation (+/- fraction)
            'scale': 0.5,     # Image scale (+/- gain)
            'mosaic': 1.0,    # Mosaic augmentation probability
            'mixup': 0.0,     # Mixup augmentation probability
            'seed': 42        # Fixed seed for reproducibility
        }

        # Add segmentation-specific parameters
        if is_segmentation:
            train_args.update({
                'copy_paste': 0.0,        # Copy-paste augmentation
                'copy_paste_mode': 'flip', # Copy-paste mode
                'overlap_mask': True,      # Overlap masks during training
                'mask_ratio': 4           # Mask downsampling ratio
            })

        # Add multi-scale training if requested
        if multi_scale:
            train_args['rect'] = False  # Disable rectangular training for multi-scale

        if log_callback:
            log_callback(f"Training {'segmentation' if is_segmentation else 'detection'} model: {model_path}")
            log_callback(f"Segmentation-specific parameters: {is_segmentation}")

        # Start training with optimized parameters for YOLO11
        model.train(**train_args)
        
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


def start_training_threaded(data_path, epochs, imgsz, batch, lr0, optimizer, augment,
                           project, name, model_path="yolo11n.pt", callback=None):
    """Start training in a separate thread with simplified parameters."""
    import threading

    def training_worker():
        try:
            start_training(
                data_path=data_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr0,
                resume=False,
                multi_scale=False,
                cos_lr=True,
                close_mosaic=10,
                momentum=0.937,
                warmup_epochs=3,
                warmup_momentum=0.8,
                box=7.5,  # Use YOLO11 default
                dropout=0.0,  # Keep for backward compatibility
                project=project,
                name=name,
                model_path=model_path,
                progress_callback=callback
            )
            if callback:
                callback(100)  # Signal completion
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg)
            if callback:
                callback(0, error_msg)

    # Start training in separate thread
    thread = threading.Thread(target=training_worker)
    thread.daemon = True
    thread.start()
    return thread