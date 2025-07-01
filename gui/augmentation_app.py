"""Main application module for image augmentation."""

import os
import logging
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QCheckBox,
    QFileDialog, QProgressBar, QStackedWidget, QDialog
)

from PyQt6.QtCore import Qt, QTimer

from gui.augmentation_settings import SettingsDialog, load_settings, save_settings
from gui.augmentation_preview import (
    display_image, load_sample_image, generate_preview, toggle_preview_mode
)
from gui.augmentation_methods import (
    create_method_controls, get_method_key, show_method_info
)
from gui.augmentation_processor import start_augmentation_process, calculate_augmentation_count
from project_manager import ProjectManager, WorkflowStep

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageAugmentationApp(QMainWindow):
    """Main window for the image augmentation application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Augmentation Tool")
        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Global style for all widgets in the app
        self.setStyleSheet("""
            QWidget {
                color: black;
            }
            QProgressBar {
                text-align: center;
                font-size: 16px;
                font-weight: bold;
                min-height: 30px;
                color: black;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 5px;
            }
            QLabel {
                color: black;
            }
            QCheckBox {
                color: black;
            }
            QSpinBox {
                color: black;
            }
            QGroupBox {
                color: black;
            }
        """)

        # Initialize attributes
        self.source_path = None
        self.dest_path = None
        self.sample_image = None
        self.sample_boxes = []
        self.settings = load_settings()
        self.methods = ["Verschiebung", "Rotation", "Zoom", "Helligkeit", "Unschärfe"]
        self.method_checkboxes = []
        self.method_levels = {}
        
        # Create left panel (settings/controls)
        self.left_panel = self.create_left_panel()
        
        # Create right panel (preview/processing)
        self.right_panel, self.stack = self.create_right_panel()
        
        # Add panels to main layout
        self.layout.addWidget(self.left_panel)
        self.layout.addWidget(self.right_panel, stretch=1)
        
        # Setup preview timer
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(lambda: generate_preview(self))
        
        # Initialize preview mode
        toggle_preview_mode(self, self.preview_checkbox.isChecked())

    def create_left_panel(self):
        """Create and return the left panel with settings and controls."""
        left_panel = QWidget()
        left_panel.setFixedWidth(380)  # Reduced width
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        left_layout.setSpacing(4)  # Reduced spacing
        left_panel.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-right: 1px solid #ddd;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 3px;
                margin: 3px;
            }
            QLabel {
                padding: 2px;
            }
        """)

        # Source directory
        self.source_label = QLabel("Quellverzeichnis:")
        self.source_label.setStyleSheet("color: black;")
        self.source_path_button = QPushButton("Durchsuchen")
        self.source_layout = QHBoxLayout()
        self.source_layout.addWidget(self.source_label)
        self.source_layout.addWidget(self.source_path_button)
        self.source_path_button.clicked.connect(self.browse_source)
        left_layout.addLayout(self.source_layout)

        # Destination directory
        self.dest_label = QLabel("Zielverzeichnis:")
        self.dest_label.setStyleSheet("color: black;")
        self.dest_path_button = QPushButton("Durchsuchen")
        self.dest_layout = QHBoxLayout()
        self.dest_layout.addWidget(self.dest_label)
        self.dest_layout.addWidget(self.dest_path_button)
        self.dest_path_button.clicked.connect(self.browse_dest)
        left_layout.addLayout(self.dest_layout)

        # Combined count info
        self.count_info = QLabel("Aktuelle Anzahl Bilder: -\nErwartete Anzahl Bilder: -")
        self.count_info.setStyleSheet("""
            font-size: 13px;
            padding: 5px;
            background: rgba(33, 150, 243, 0.1);
            border-radius: 5px;
            margin: 3px;
            min-height: 0;
        """)
        left_layout.addWidget(self.count_info)

        # Settings button
        self.settings_button = QPushButton("Einstellungen...")
        self.settings_button.clicked.connect(self.show_settings)
        left_layout.addWidget(self.settings_button)

        # Augmentation method selection
        self.method_group = QGroupBox("Augmentierungsmethoden auswählen")
        self.method_layout = QVBoxLayout()
        self.method_layout.setContentsMargins(4, 4, 4, 4)
        self.method_layout.setSpacing(3)
        
        # Preview toggle
        preview_layout = QHBoxLayout()
        self.preview_checkbox = QCheckBox("Bild Vorschau anzeigen")
        # Set to false by default as per requirement
        self.preview_checkbox.setChecked(False)
        self.preview_checkbox.stateChanged.connect(lambda state: toggle_preview_mode(self, state))
        preview_layout.addWidget(self.preview_checkbox)
        
        # Add "Neue Vorschau" button
        self.refresh_preview_button = QPushButton("Neue Vorschau")
        self.refresh_preview_button.clicked.connect(lambda: generate_preview(self))
        self.refresh_preview_button.setEnabled(True)
        preview_layout.addWidget(self.refresh_preview_button)
        
        self.method_layout.addLayout(preview_layout)
        
        self.method_group.setLayout(self.method_layout)
        left_layout.addWidget(self.method_group)
        
        # Create method controls
        create_method_controls(self)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        # Start button
        self.start_button = QPushButton("Augmentierung starten")
        self.start_button.clicked.connect(lambda: start_augmentation_process(self))
        left_layout.addWidget(self.start_button)
        
        return left_panel

    def create_right_panel(self):
        """Create and return the right panel with preview/processing areas."""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # StackedWidget to switch between processing and preview modes
        stack = QStackedWidget()
        right_layout.addWidget(stack)
        
        # Page 1: Processing view (during augmentation)
        processing_widget = QWidget()
        processing_layout = QVBoxLayout(processing_widget)
        
        # Preview label for augmentation process
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: black;")
        processing_layout.addWidget(self.preview_label)
        
        stack.addWidget(processing_widget)
        
        # Page 2: Preview panel (during setup)
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        preview_layout.setSpacing(5)
        
        # Title row with minimal height
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)
        title_layout.addWidget(QLabel("<b>Original</b>"), 1)
        title_layout.addWidget(QLabel("<b>Level 1 Preview</b>"), 1)
        title_layout.addWidget(QLabel("<b>Level 2 Preview</b>"), 1)
        preview_layout.addLayout(title_layout)
        
        # Preview images row
        preview_images_layout = QHBoxLayout()
        preview_images_layout.setContentsMargins(0, 0, 0, 0)
        preview_images_layout.setSpacing(5)
        
        # Original image preview
        self.original_preview = QLabel()
        self.original_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_preview.setStyleSheet("border: 1px solid #ccc; background-color: black;")
        self.original_preview.setMinimumHeight(300)
        
        # Level 1 preview
        self.level1_preview = QLabel()
        self.level1_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.level1_preview.setStyleSheet("border: 1px solid #2196F3; background-color: black;")
        self.level1_preview.setMinimumHeight(300)
        
        # Level 2 preview
        self.level2_preview = QLabel()
        self.level2_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.level2_preview.setStyleSheet("border: 1px solid #F44336; background-color: black;")
        self.level2_preview.setMinimumHeight(300)
        
        preview_images_layout.addWidget(self.original_preview, 1)
        preview_images_layout.addWidget(self.level1_preview, 1)
        preview_images_layout.addWidget(self.level2_preview, 1)
        
        preview_layout.addLayout(preview_images_layout, 1)
        stack.addWidget(preview_widget)
        
        return right_panel, stack

    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = dialog.get_settings()
            save_settings(self.settings)

    def browse_source(self):
        """Open file dialog to select source directory."""
        path = QFileDialog.getExistingDirectory(self, "Quellverzeichnis auswählen")
        if path:
            self.source_path = path
            self.source_label.setText(f"Quellverzeichnis: {path}")
            self.update_expected_count()
            
            # Load sample image for preview
            load_sample_image(self)

    def browse_dest(self):
        """Open file dialog to select destination directory."""
        path = QFileDialog.getExistingDirectory(self, "Zielverzeichnis auswählen")
        if path:
            self.dest_path = path
            self.dest_label.setText(f"Zielverzeichnis: {path}")

    def update_expected_count(self):
        """Update expected augmentation count based on selected methods."""
        try:
            original_count, total_count = calculate_augmentation_count(self)
            
            self.count_info.setText(
                f"Aktuelle Anzahl Bilder: {original_count:,}\n"
                f"Erwartete Anzahl Bilder: {total_count:,}"
            )

            # Update preview when methods change
            self.update_preview()
        except Exception as e:
            logger.error(f"Error updating expected count: {e}")

    def update_preview(self):
        """Schedule an update to the preview images."""
        if not self.preview_checkbox.isChecked() or self.sample_image is None:
            return
            
        # Use a timer to debounce frequent updates
        self.preview_timer.start(100)

    def save_settings_to_project(self):
        """Speichert Augmentation-Settings ins Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            # Aktuelle Settings sammeln
            settings = {
                'methods': {},
                'flip_settings': {
                    'horizontal': self.horizontal_flip.isChecked(),
                    'vertical': self.vertical_flip.isChecked()
                }
            }
            
            # Method-Settings sammeln
            for method in self.methods:
                from gui.augmentation_common import get_method_key
                method_key = get_method_key(method)
                if method_key in self.method_levels:
                    checkbox, level1_spin, level2_spin = self.method_levels[method_key]
                    settings['methods'][method_key] = {
                        'enabled': checkbox.isChecked(),
                        'level1': level1_spin.value(),
                        'level2': level2_spin.value()
                    }
            
            self.project_manager.update_augmentation_settings(settings)
    
    def load_settings_from_project(self):
        """Lädt Augmentation-Settings aus dem Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            settings = self.project_manager.get_augmentation_settings()
            
            if 'methods' in settings:
                for method_key, method_settings in settings['methods'].items():
                    if method_key in self.method_levels:
                        checkbox, level1_spin, level2_spin = self.method_levels[method_key]
                        checkbox.setChecked(method_settings.get('enabled', False))
                        level1_spin.setValue(method_settings.get('level1', 2))
                        level2_spin.setValue(method_settings.get('level2', 10))
            
            if 'flip_settings' in settings:
                flip_settings = settings['flip_settings']
                self.horizontal_flip.setChecked(flip_settings.get('horizontal', False))
                self.vertical_flip.setChecked(flip_settings.get('vertical', False))
    
    def complete_augmentation_with_project_integration(self):
        """Erweiterte Augmentation-Completion mit Projekt-Integration"""
        # Settings speichern
        self.save_settings_to_project()
        
        # Workflow-Schritt markieren
        if hasattr(self, 'project_manager') and self.project_manager:
            self.project_manager.mark_step_completed(WorkflowStep.AUGMENTATION)


# ==================== AUGMENTATION APP INTEGRATION ====================

"""
Ergänzungen für gui/augmentation_app.py
"""

class AugmentationAppExtensions:
    """Erweiterungen für die Augmentation App"""
    
    def save_settings_to_project(self):
        """Speichert Augmentation-Settings ins Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            # Aktuelle Settings sammeln
            settings = {
                'methods': {},
                'flip_settings': {
                    'horizontal': self.horizontal_flip.isChecked(),
                    'vertical': self.vertical_flip.isChecked()
                }
            }
            
            # Method-Settings sammeln
            for method in self.methods:
                from gui.augmentation_common import get_method_key
                method_key = get_method_key(method)
                if method_key in self.method_levels:
                    checkbox, level1_spin, level2_spin = self.method_levels[method_key]
                    settings['methods'][method_key] = {
                        'enabled': checkbox.isChecked(),
                        'level1': level1_spin.value(),
                        'level2': level2_spin.value()
                    }
            
            self.project_manager.update_augmentation_settings(settings)
    
    def load_settings_from_project(self):
        """Lädt Augmentation-Settings aus dem Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            settings = self.project_manager.get_augmentation_settings()
            
            if 'methods' in settings:
                for method_key, method_settings in settings['methods'].items():
                    if method_key in self.method_levels:
                        checkbox, level1_spin, level2_spin = self.method_levels[method_key]
                        checkbox.setChecked(method_settings.get('enabled', False))
                        level1_spin.setValue(method_settings.get('level1', 2))
                        level2_spin.setValue(method_settings.get('level2', 10))
            
            if 'flip_settings' in settings:
                flip_settings = settings['flip_settings']
                self.horizontal_flip.setChecked(flip_settings.get('horizontal', False))
                self.vertical_flip.setChecked(flip_settings.get('vertical', False))
    
    def complete_augmentation_with_project_integration(self):
        """Erweiterte Augmentation-Completion mit Projekt-Integration"""
        # Settings speichern
        self.save_settings_to_project()
        
        # Workflow-Schritt markieren
        if hasattr(self, 'project_manager') and self.project_manager:
            self.project_manager.mark_step_completed(WorkflowStep.AUGMENTATION)
