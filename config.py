"""Central configuration for the YOLO trainer application."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingDefaults:
    """Default values for training parameters."""
    epochs: int = 100
    image_size: int = 1280
    batch: float = 0.9
    lr0: float = 0.005
    resume: bool = False
    multi_scale: bool = False
    cos_lr: bool = True
    close_mosaic: int = 0
    momentum: float = 0.9
    warmup_epochs: int = 3
    warmup_momentum: float = 0.9
    box: int = 7
    dropout: float = 0.1
    project_dir: str = "yolo_training_results"
    experiment_name: str = "experiment"

@dataclass
class UIConfig:
    """UI-related configuration."""
    window_width: int = 600
    window_height: int = 700
    update_interval: int = 5000  # ms
    dark_mode: bool = False

class Config:
    """Global configuration singleton."""
    training = TrainingDefaults()
    ui = UIConfig()
    
    @staticmethod
    def get_training_params() -> Dict[str, Any]:
        """Get training parameters as dictionary."""
        return {
            "epochs": Config.training.epochs,
            "imgsz": Config.training.image_size,
            "batch": Config.training.batch,
            "lr0": Config.training.lr0,
            "resume": Config.training.resume,
            "multi_scale": Config.training.multi_scale,
            "cos_lr": Config.training.cos_lr,
            "close_mosaic": Config.training.close_mosaic,
            "momentum": Config.training.momentum,
            "warmup_epochs": Config.training.warmup_epochs,
            "warmup_momentum": Config.training.warmup_momentum,
            "box": Config.training.box,
            "dropout": Config.training.dropout
        }