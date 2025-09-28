import threading
import logging
import os
import shutil
import subprocess
import sys
from PyQt6.QtCore import QObject, pyqtSignal

# Configure logging
logger = logging.getLogger("training_thread")
logger.setLevel(logging.INFO)

# Global process handle and stop flag
training_process = None
stop_event = threading.Event()

class TrainingSignals(QObject):
    """Signal class for thread-safe communication."""
    progress_updated = pyqtSignal(int, str)
    log_updated = pyqtSignal(str)
    results_updated = pyqtSignal(object)  # Will carry the DataFrame

def start_training_thread(signals, data_path, epochs, imgsz, batch, lr0, resume, multi_scale,
                        cos_lr, close_mosaic, momentum, warmup_epochs, warmup_momentum,
                        box, dropout, project, experiment, model_path="yolo11n.pt"):
    """Start training in a separate thread."""
    stop_event.clear()
    training_thread = threading.Thread(
        target=run_training,
        args=(
            signals,
            data_path, epochs, imgsz, batch, lr0, resume, multi_scale,
            cos_lr, close_mosaic, momentum, warmup_epochs, warmup_momentum,
            box, dropout, project, experiment, model_path
        )
    )
    training_thread.daemon = True
    training_thread.start()
    return training_thread

def run_training(signals, data_path, epochs, imgsz, batch, lr0, resume, multi_scale,
                 cos_lr, close_mosaic, momentum, warmup_epochs, warmup_momentum,
                 box, dropout, project, experiment, model_path="yolo11n.pt"):
    """Run the training process in a separate thread using a subprocess."""

    global training_process

    try:
        def progress_callback(progress, message=""):
            signals.progress_updated.emit(progress, message)

        def log_callback(message):
            signals.log_updated.emit(message)

        # Build command for subprocess
        cmd = [
            sys.executable, "-m", "yolo.train_runner",
            "--data", data_path,
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--lr0", str(lr0),
            "--close_mosaic", str(close_mosaic),
            "--momentum", str(momentum),
            "--warmup_epochs", str(warmup_epochs),
            "--warmup_momentum", str(warmup_momentum),
            "--box", str(box),
            "--dropout", str(dropout),
            "--project", project,
            "--experiment", experiment,
            "--model", model_path,
        ]
        if resume:
            cmd.append("--resume")
        if multi_scale:
            cmd.append("--multi_scale")
        if cos_lr:
            cmd.append("--cos_lr")

        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Relay subprocess output and handle stop signal
        for line in iter(training_process.stdout.readline, ""):
            if line:
                log_callback(line.strip())
            if stop_event.is_set():
                training_process.terminate()
                break

        return_code = training_process.wait()
        training_process = None

        if stop_event.is_set():
            progress_callback(0, "Training stopped by user")
        elif return_code == 0:
            progress_callback(100, "Training completed successfully")
        else:
            progress_callback(0, f"Training failed with code {return_code}")

    except Exception:
        import traceback
        error_msg = f"Training error:\n{traceback.format_exc()}"
        progress_callback(0, error_msg)


def stop_training(project=None, experiment=None):
    """Signal the current training process to stop and clean up files."""
    stop_event.set()
    global training_process
    if training_process and training_process.poll() is None:
        logger.info("Terminating training process")
        training_process.terminate()
        try:
            training_process.wait(timeout=10)
        except Exception:
            training_process.kill()
    training_process = None

    if project and experiment:
        try:
            base_path = os.path.join(project, experiment)
            if os.path.isdir(base_path):
                shutil.rmtree(base_path, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error cleaning up training files: {e}")