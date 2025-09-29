"""Optimized YOLO training module with separate detection and segmentation training."""

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
        if gpu_mem >= 16:  # High-end GPUs (>= 16GB)
            return 'cuda:0', 1.0
        elif gpu_mem >= 8:  # Mid-range GPUs (8-16GB)
            return 'cuda:0', 0.8
        else:  # Low memory GPUs
            return 'cuda:0', 0.5
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
        return 'cuda:0', 0.5

def start_detection_training(data_path, epochs, imgsz, batch, lr0, resume, multi_scale, cos_lr, 
                           close_mosaic, momentum, warmup_epochs, warmup_momentum, box, dropout, 
                           project, name, model_path="yolo11n.pt", progress_callback=None, log_callback=None):
    """Start YOLO detection training optimized for object detection."""
    try:
        setup_logging()
        
        # Get optimal device and batch settings
        device, batch_scale = get_device_settings()
        workers = get_optimal_workers()
        
        # Adjust batch size based on GPU memory
        if batch > batch_scale:
            logger.warning(f"Reducing batch size from {batch} to {batch_scale} due to GPU memory constraints")
            batch = batch_scale

        logger.info(f"Starting Detection Training")
        logger.info(f"Using device: {device}")
        logger.info(f"Batch size: {batch}")
        logger.info(f"Workers: {workers}")
        
        # Initialize detection model
        model = YOLO(model_path)
        logger.info(f"Loaded detection model: {model_path}")
        
        if log_callback: 
            log_callback("Starting DETECTION training with the following parameters:")
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

        # Detection-specific training parameters
        train_args = {
            'resume': resume,
            'data': data_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'lr0': lr0,
            'optimizer': 'AdamW',  # Best optimizer for detection
            'device': device,
            'project': project,
            'name': name,
            'workers': workers,
            'exist_ok': True,
            'plots': True,
            'save': True,
            'val': True,
            'verbose': True,
            'patience': 50,  # Early stopping patience for detection
            # Detection-specific loss weights
            'box': box,      # Box loss gain (user-defined)
            'cls': 0.5,      # Classification loss gain
            'dfl': 1.5,      # Distribution focal loss gain
            # Training hyperparameters optimized for detection
            'cos_lr': cos_lr,
            'close_mosaic': close_mosaic,
            'momentum': momentum,
            'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            'weight_decay': 0.0005,  # L2 regularization
            'dropout': dropout,
            # Detection-optimized augmentation
            'hsv_h': 0.015,    # HSV-Hue augmentation
            'hsv_s': 0.7,      # HSV-Saturation augmentation
            'hsv_v': 0.4,      # HSV-Value augmentation
            'degrees': 0.0,     # Image rotation (use 0 for controlled training)
            'translate': 0.1,   # Image translation
            'scale': 0.5,       # Image scale
            'shear': 0.0,       # Image shear
            'perspective': 0.0, # Image perspective
            'flipud': 0.0,      # Vertical flip probability
            'fliplr': 0.5,      # Horizontal flip probability
            'mosaic': 1.0,      # Mosaic augmentation probability
            'mixup': 0.0,       # Mixup augmentation (usually 0 for detection)
            'copy_paste': 0.0,  # Copy-paste augmentation
            'seed': 42          # Fixed seed for reproducibility
        }

        # Add multi-scale training if requested
        if multi_scale:
            train_args['rect'] = False  # Disable rectangular training for multi-scale
            train_args['cache'] = False  # Disable cache for multi-scale

        # Start detection training
        if log_callback:
            log_callback("Initializing detection training...")
        
        results = model.train(**train_args)
        
        if progress_callback:
            progress_callback(100, "Detection training completed successfully")
        
        if log_callback:
            log_callback("Detection training completed successfully")
            
        return results
    
    except torch.cuda.OutOfMemoryError:
        error_msg = (
            "GPU memory exhausted! Try reducing batch size or image size for detection training."
        )
        logger.error(error_msg)
        if progress_callback:
            progress_callback(0, error_msg)
        raise
    except Exception:
        error_msg = f"Detection training error:\n{traceback.format_exc()}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback(0, error_msg)
        raise

def start_segmentation_training(data_path, epochs, imgsz, batch, lr0, resume, multi_scale, cos_lr,
                              close_mosaic, momentum, warmup_epochs, warmup_momentum, box, dropout,
                              project, name, model_path="yolo11n-seg.pt", progress_callback=None, log_callback=None):
    """Start YOLO segmentation training optimized for instance segmentation."""
    try:
        setup_logging()
        
        # Get optimal device and batch settings
        device, batch_scale = get_device_settings()
        workers = get_optimal_workers()
        
        # Segmentation typically needs more memory, so be more conservative
        segmentation_batch_scale = batch_scale * 0.7  # Reduce by 30% for segmentation
        if batch > segmentation_batch_scale:
            logger.warning(f"Reducing batch size from {batch} to {segmentation_batch_scale} for segmentation training")
            batch = segmentation_batch_scale

        logger.info(f"Starting Segmentation Training")
        logger.info(f"Using device: {device}")
        logger.info(f"Batch size: {batch}")
        logger.info(f"Workers: {workers}")
        
        # Initialize segmentation model
        model = YOLO(model_path)
        logger.info(f"Loaded segmentation model: {model_path}")
        
        if log_callback: 
            log_callback("Starting SEGMENTATION training with the following parameters:")
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

        # Segmentation-specific training parameters
        train_args = {
            'resume': resume,
            'data': data_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'lr0': lr0,
            'optimizer': 'AdamW',  # Best optimizer for segmentation
            'device': device,
            'project': project,
            'name': name,
            'workers': workers,
            'exist_ok': True,
            'plots': True,
            'save': True,
            'val': True,
            'verbose': True,
            'patience': 100,  # Higher patience for segmentation (more complex task)
            # Segmentation-specific loss weights
            'box': box * 0.8,    # Slightly reduced box loss for segmentation
            'cls': 0.5,          # Classification loss gain
            'dfl': 1.5,          # Distribution focal loss gain
            # Training hyperparameters optimized for segmentation
            'cos_lr': cos_lr,
            'close_mosaic': close_mosaic,
            'momentum': momentum,
            'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            'weight_decay': 0.0005,
            'dropout': dropout,
            # Segmentation-optimized augmentation (more conservative)
            'hsv_h': 0.01,      # Reduced HSV augmentation for segmentation
            'hsv_s': 0.5,       # Reduced saturation changes
            'hsv_v': 0.3,       # Reduced value changes
            'degrees': 0.0,     # No rotation for segmentation (can distort masks)
            'translate': 0.05,  # Reduced translation
            'scale': 0.3,       # Reduced scale changes
            'shear': 0.0,       # No shear (bad for masks)
            'perspective': 0.0, # No perspective (bad for masks)
            'flipud': 0.0,      # No vertical flip
            'fliplr': 0.3,      # Reduced horizontal flip probability
            'mosaic': 0.8,      # Reduced mosaic for better mask quality
            'mixup': 0.0,       # No mixup (can corrupt masks)
            'copy_paste': 0.3,  # Copy-paste augmentation (segmentation-specific)
            'seed': 42,         # Fixed seed for reproducibility
            # Segmentation-specific parameters
            'overlap_mask': True,     # Allow overlapping masks
            'mask_ratio': 4,          # Mask downsampling ratio
            'retina_masks': False,    # Use standard mask resolution
        }

        # Multi-scale is generally not recommended for segmentation due to mask complexity
        if multi_scale:
            logger.warning("Multi-scale training is not recommended for segmentation due to mask complexity")
            if log_callback:
                log_callback("Warning: Multi-scale disabled for segmentation training")
        
        # Always disable rect for segmentation (rectangular training doesn't work well with masks)
        train_args['rect'] = False
        train_args['cache'] = False  # Disable cache for segmentation (memory intensive)

        # Start segmentation training
        if log_callback:
            log_callback("Initializing segmentation training...")
        
        results = model.train(**train_args)
        
        if progress_callback:
            progress_callback(100, "Segmentation training completed successfully")
        
        if log_callback:
            log_callback("Segmentation training completed successfully")
            
        return results
    
    except torch.cuda.OutOfMemoryError:
        error_msg = (
            "GPU memory exhausted! Segmentation requires more memory. "
            "Try reducing batch size or image size for segmentation training."
        )
        logger.error(error_msg)
        if progress_callback:
            progress_callback(0, error_msg)
        raise
    except Exception:
        error_msg = f"Segmentation training error:\n{traceback.format_exc()}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback(0, error_msg)
        raise

def start_training(data_path, epochs, imgsz, batch, lr0, resume, multi_scale, cos_lr, close_mosaic,
                   momentum, warmup_epochs, warmup_momentum, box, dropout, project, name,
                   model_path="yolo11n.pt", progress_callback=None, log_callback=None):
    """Start YOLO training with automatic detection/segmentation selection."""
    try:
        # Determine if this is a segmentation model
        is_segmentation = 'seg' in model_path.lower()
        
        if is_segmentation:
            logger.info("Detected segmentation model, using segmentation training pipeline")
            if log_callback:
                log_callback("Segmentation model detected - using optimized segmentation training")
            return start_segmentation_training(
                data_path, epochs, imgsz, batch, lr0, resume, multi_scale, cos_lr,
                close_mosaic, momentum, warmup_epochs, warmup_momentum, box, dropout,
                project, name, model_path, progress_callback, log_callback
            )
        else:
            logger.info("Detected detection model, using detection training pipeline")
            if log_callback:
                log_callback("Detection model detected - using optimized detection training")
            return start_detection_training(
                data_path, epochs, imgsz, batch, lr0, resume, multi_scale, cos_lr,
                close_mosaic, momentum, warmup_epochs, warmup_momentum, box, dropout,
                project, name, model_path, progress_callback, log_callback
            )
            
    except Exception as e:
        error_msg = f"Training initialization error:\n{traceback.format_exc()}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback(0, error_msg)
        raise

def start_training_threaded(data_path, epochs, imgsz, batch, lr0, optimizer, augment,
                           project, name, model_path="yolo11n.pt", callback=None):
    """Start training in a separate thread with simplified parameters (legacy support)."""
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
                box=7.5,
                dropout=0.0,
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