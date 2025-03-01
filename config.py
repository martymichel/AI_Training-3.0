"""Central configuration for the YOLO trainer application."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingDefaults:
    """Default values for training parameters."""
    epochs: int = 100
    image_size: int = 640
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = "AdamW"
    augmentation: bool = False
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
            "batch": Config.training.batch_size,
            "lr0": Config.training.learning_rate,
            "optimizer": Config.training.optimizer,
            "augment": Config.training.augmentation
        }