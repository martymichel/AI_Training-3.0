"""Thread management for the training process."""

import threading
import logging
import time
import os
from PyQt6.QtCore import QObject, pyqtSignal

# Configure logging
logger = logging.getLogger("training_thread")
logger.setLevel(logging.INFO)

class TrainingSignals(QObject):
    """Signal class for thread-safe communication."""
    progress_updated = pyqtSignal(int, str)
    log_updated = pyqtSignal(str)
    results_updated = pyqtSignal(object)  # Will carry the DataFrame

def start_training_thread(signals, data_path, epochs, imgsz, batch, lr0, resume, multi_scale,
                        cos_lr, close_mosaic, momentum, warmup_epochs, warmup_momentum,
                        box, dropout, project, experiment):
    """Start training in a separate thread."""
    training_thread = threading.Thread(
        target=run_training,
        args=(
            signals,
            data_path, epochs, imgsz, batch, lr0, resume, multi_scale,
            cos_lr, close_mosaic, momentum, warmup_epochs, warmup_momentum,
            box, dropout, project, experiment
        )
    )
    training_thread.daemon = True
    training_thread.start()
    return training_thread

def run_training(signals, data_path, epochs, imgsz, batch, lr0, resume, multi_scale,
               cos_lr, close_mosaic, momentum, warmup_epochs, warmup_momentum,
               box, dropout, project, experiment):
    """Run the training process in a separate thread."""
    from yolo.yolo_train import start_training
    
    try:
        # Define progress callback
        def progress_callback(progress, message=""):
            signals.progress_updated.emit(progress, message)
            
        # Define log callback
        def log_callback(message):
            signals.log_updated.emit(message)
            
        # Start actual training
        start_training(
            data_path, epochs, imgsz, batch, lr0, resume, multi_scale,
            cos_lr, close_mosaic, momentum, warmup_epochs, warmup_momentum,
            box, dropout, project, experiment,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
    except Exception as e:
        import traceback
        error_msg = f"Training error:\n{traceback.format_exc()}"
        signals.progress_updated.emit(0, error_msg)