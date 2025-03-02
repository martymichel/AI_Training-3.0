"""Optimized YOLO training module."""

import torch
import os
import sys
import gc
from ultralytics import YOLO
import traceback
import logging
# Disable debug logging for training
logging.getLogger().setLevel(logging.WARNING)

from rich.console import Console

# Rich Console for colored output
console = Console()

# Get CPU core count
workers_found = os.cpu_count()

def optimize_memory():
    """Optimize memory before training."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def start_training(data_path, epochs, imgsz, batch, lr0, resume, multi_scale, cos_lr, close_mosaic,
                   momentum, warmup_epochs, warmup_momentum, box, dropout, project, name,
                   progress_callback=None, log_callback=None):
    """Start YOLO training with GPU support."""
    try:
        optimize_memory()

        # CUDA configuration
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        console.log(f"[bold green]Using device:[/bold green] {device}")
        
        # Initialize model
        model = YOLO("yolo11n.pt")
        
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
            workers=min(8, workers_found-2),  # Limit workers
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
        error_msg = "GPU memory exhausted! Reduce batch size or image size."
        if progress_callback:
            progress_callback(0, error_msg)
    except Exception:
        error_msg = f"Training error:\n{traceback.format_exc()}"
        if progress_callback:
            progress_callback(0, error_msg)