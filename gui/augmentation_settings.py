"""Settings management for the augmentation application."""

import os
import json
import logging
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLabel, QSpinBox, QDoubleSpinBox, 
    QPushButton, QHBoxLayout
)

logger = logging.getLogger(__name__)

class SettingsDialog(QDialog):
    """Dialog for augmentation settings."""
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Augmentation Settings")
        self.setModal(True)
        
        layout = QFormLayout(self)
        
        # Box validation settings
        self.min_visibility = QDoubleSpinBox()
        self.min_visibility.setRange(0.1, 1.0)
        self.min_visibility.setSingleStep(0.1)
        self.min_visibility.setValue(settings.get('min_visibility', 0.3))
        self.min_visibility.setToolTip(
            "Minimum required visibility of a box after augmentation (0.1-1.0)\n"
            "Higher values ensure better object visibility"
        )
        layout.addRow("Minimum Visibility:", self.min_visibility)
        
        self.min_size = QSpinBox()
        self.min_size.setRange(10, 100)
        self.min_size.setValue(settings.get('min_size', 20))
        self.min_size.setToolTip(
            "Minimum size of a box in pixels\n"
            "Boxes smaller than this will be discarded"
        )
        layout.addRow("Minimum Box Size (px):", self.min_size)
        
        # Method-specific settings
        self.max_shift = QSpinBox()
        self.max_shift.setRange(5, 50)
        self.max_shift.setValue(settings.get('max_shift', 30))
        self.max_shift.setToolTip("Maximum shift as percentage of image size")
        layout.addRow("Max Shift (%):", self.max_shift)
        
        self.max_rotation = QSpinBox()
        self.max_rotation.setRange(5, 45)
        self.max_rotation.setValue(settings.get('max_rotation', 30))
        self.max_rotation.setToolTip("Maximum rotation angle in degrees")
        layout.addRow("Max Rotation (°):", self.max_rotation)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addRow(button_layout)

    def get_settings(self):
        """Get current settings as dictionary."""
        return {
            'min_visibility': self.min_visibility.value(),
            'min_size': self.min_size.value(),
            'max_shift': self.max_shift.value(),
            'max_rotation': self.max_rotation.value()
        }

def load_settings():
    """Load settings from JSON file."""
    try:
        if os.path.exists('augmentation_settings.json'):
            with open('augmentation_settings.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
    return {}

def save_settings(settings):
    """Save settings to JSON file."""
    try:
        with open('augmentation_settings.json', 'w') as f:
            json.dump(settings, f)
    except Exception as e:
        logger.error(f"Error saving settings: {e}")