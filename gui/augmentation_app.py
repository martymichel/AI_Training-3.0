"""Image augmentation application module."""

import cv2
import os
import json
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QGroupBox, QCheckBox,
    QFileDialog, QProgressBar, QMessageBox, QDoubleSpinBox,
    QDialog, QFormLayout
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import logging
from utils.augmentation_utils import augment_image_with_boxes
from itertools import product

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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

class ImageAugmentationApp(QMainWindow):
    """Main window for the image augmentation application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Augmentation Tool")
        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Zentral-Widget und Layout
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

        # Left panel for settings
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_panel.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-right: 1px solid #ddd;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                margin: 8px;
            }
            QLabel {
                padding: 4px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)

        # Right panel for preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)

        # Preview label
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: black;")
        right_layout.addWidget(self.preview_label)

        # Load settings
        self.settings = self.load_settings()

        # Source directory
        self.source_label = QLabel("Quellverzeichnis:")
        self.source_label.setStyleSheet("color: black;")
        self.source_path_button = QPushButton("Durchsuchen")
        self.source_path_button.clicked.connect(self.browse_source)
        self.source_layout = QHBoxLayout()
        self.source_layout.addWidget(self.source_label)
        self.source_layout.addWidget(self.source_path_button)
        left_layout.addLayout(self.source_layout)

        # Destination directory
        self.dest_label = QLabel("Zielverzeichnis:")
        self.dest_label.setStyleSheet("color: black;")
        self.dest_path_button = QPushButton("Durchsuchen")
        self.dest_path_button.clicked.connect(self.browse_dest)
        self.dest_layout = QHBoxLayout()
        self.dest_layout.addWidget(self.dest_label)
        self.dest_layout.addWidget(self.dest_path_button)
        left_layout.addLayout(self.dest_layout)

        # Combined count info
        self.count_info = QLabel("Aktuelle Anzahl Bilder: -\nErwartete Anzahl Bilder: -")
        self.count_info.setStyleSheet("""
            font-size: 14px;
            padding: 10px;
            background: rgba(33, 150, 243, 0.1);
            border-radius: 5px;
            margin: 5px;
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
        self.method_group.setLayout(self.method_layout)
        left_layout.addWidget(self.method_group)
        
        # Update checkbox style
        self.method_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QCheckBox {
                color: black;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #f8f8f8;
                margin: 2px;
            }
            QCheckBox:hover {
                background-color: #f0f0f0;
            }
        """)
        
        # Augmentation methods
        self.methods = ["Verschiebung", "Rotation", "Zoom", "Helligkeit", "Unschärfe"]
        self.method_checkboxes = []
        self.method_levels = {}

        for method in self.methods:
            checkbox = QCheckBox(method)
            checkbox.stateChanged.connect(self.update_expected_count)
            self.method_checkboxes.append(checkbox)
            self.method_layout.addWidget(checkbox)
            
            level_layout = QHBoxLayout()
            level_label1 = QLabel("Stufe 1:")
            level_label2 = QLabel("Stufe 2:")
            level_spinbox1 = QSpinBox()
            level_spinbox2 = QSpinBox()
            level_spinbox1.setRange(1, 100)
            level_spinbox2.setRange(1, 100)
            level_layout.addWidget(level_label1)
            level_layout.addWidget(level_spinbox1)
            level_layout.addWidget(level_label2)
            level_layout.addWidget(level_spinbox2)
            self.method_layout.addLayout(level_layout)

            method_key = self.get_method_key(method)
            self.method_levels[method_key] = (checkbox, level_spinbox1, level_spinbox2)

        # Add horizontal flip checkbox separately without levels
        self.flip_checkbox = QCheckBox("Horizontal Spiegeln")
        self.flip_checkbox.stateChanged.connect(self.update_expected_count)
        self.method_layout.addWidget(self.flip_checkbox)

        # Progress bar
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        # Start button
        self.start_button = QPushButton("Augmentierung starten")
        self.start_button.clicked.connect(self.start_augmentation)
        left_layout.addWidget(self.start_button)

        # Add panels to main layout
        self.layout.addWidget(left_panel)
        self.layout.addWidget(right_panel, stretch=1)

        # Paths
        self.source_path = None
        self.dest_path = None

    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = dialog.get_settings()
            self.save_settings()

    def load_settings(self):
        """Load settings from JSON file."""
        try:
            import json
            if os.path.exists('augmentation_settings.json'):
                with open('augmentation_settings.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
        return {}

    def save_settings(self):
        """Save settings to JSON file."""
        try:
            with open('augmentation_settings.json', 'w') as f:
                json.dump(self.settings, f)
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

    def get_method_key(self, method_name):
        """Convert German method name to English key."""
        method_map = {
            "Verschiebung": "Shift",
            "Rotation": "Rotate",
            "Zoom": "Zoom",
            "Helligkeit": "Brightness",
            "Unschärfe": "Blur"
        }
        return method_map.get(method_name, method_name)

    def browse_source(self):
        """Open file dialog to select source directory."""
        path = QFileDialog.getExistingDirectory(self, "Quellverzeichnis auswählen")
        if path:
            self.source_path = path
            self.source_label.setText(f"Quellverzeichnis: {path}")
            # Update counts
            image_files = list(Path(path).rglob("*.jpg")) + list(Path(path).rglob("*.png"))
            self.update_expected_count()

    def browse_dest(self):
        """Open file dialog to select destination directory."""
        path = QFileDialog.getExistingDirectory(self, "Zielverzeichnis auswählen")
        if path:
            self.dest_path = path
            self.dest_label.setText(f"Zielverzeichnis: {path}")

    def update_expected_count(self):
        """Update expected augmentation count based on selected methods."""
        try:
            if not self.source_path:
                self.count_info.setText("Aktuelle Anzahl Bilder: -\nErwartete Anzahl Bilder: -")
                return
                
            image_files = list(Path(self.source_path).rglob("*.jpg")) + \
                         list(Path(self.source_path).rglob("*.png"))
            if not image_files:
                self.count_info.setText("Aktuelle Anzahl Bilder: 0\nErwartete Anzahl Bilder: 0")
                return

            # Get selected methods
            selected_methods = []
            for method in self.methods:
                checkbox = next(cb for cb in self.method_checkboxes if cb.text() == method)
                if checkbox.isChecked():
                    selected_methods.append(method)

            if not selected_methods:
                self.count_info.setText(
                    f"Aktuelle Anzahl Bilder: {len(image_files)}\n"
                    f"Erwartete Anzahl Bilder: {len(image_files)}"
                )
                return

            # For each method, we have 3 possibilities: no change, level1, level2
            # Start with 1 combination (original image)
            combinations = 1
            
            # Each selected method adds 2 new variations (level1 and level2)
            for _ in selected_methods:
                combinations += 2

            # If horizontal flip is enabled, add 50% more combinations
            if self.flip_checkbox.isChecked():
                combinations = int(combinations * 1.5)  # 50% more combinations

            total_augmentations = len(image_files) * combinations 
            self.count_info.setText(
                f"Aktuelle Anzahl Bilder: {len(image_files):,}\n"
                f"Erwartete Anzahl Bilder: {total_augmentations:,}"
            )
        except Exception as e:
            logger.error(f"Error updating expected count: {e}")


    def start_augmentation(self):
        """Start the augmentation process."""
        try:
            # Initial validation
            if not self.source_path or not self.dest_path:
                QMessageBox.warning(self, "Fehler", 
                                  "Bitte wählen Sie Quell- und Zielverzeichnis aus.")
                return

            # Get selected methods
            selected_methods = []
            for method_name, (checkbox, level_spinbox1, level_spinbox2) in self.method_levels.items():
                if checkbox.isChecked():
                    level1 = level_spinbox1.value()
                    level2 = level_spinbox2.value()
                    if level1 >= level2:
                        QMessageBox.warning(self, "Fehler", 
                                          f"Für {method_name} muss Stufe 1 kleiner als Stufe 2 sein.")
                        return
                    selected_methods.append((method_name, level1, level2))

            if not selected_methods:
                QMessageBox.warning(self, "Fehler", 
                                  "Bitte wählen Sie mindestens eine Augmentierungsmethode aus.")
                return

            # Find all images and labels
            image_files = list(Path(self.source_path).rglob("*.jpg")) + \
                         list(Path(self.source_path).rglob("*.png"))
            label_files = {file.stem: file for file in Path(self.source_path).rglob("*.txt")}

            if not image_files:
                QMessageBox.warning(self, "Fehler", 
                                  "Keine Bilder im Quellverzeichnis gefunden.")
                return

            # Calculate total expected combinations
            total_combinations = list(product([0, 1, 2], repeat=len(selected_methods)))
            
            # Show expected augmentation count
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Augmentierung starten")
            msg.setText(
                f"Es werden {len(image_files)} Bilder mit {len(total_combinations)} "
                f"Kombinationen verarbeitet.\n\n"
                f"Maximal mögliche Augmentierungen: {len(image_files) * len(total_combinations)}\n"
                f"(Die tatsächliche Anzahl kann geringer sein, wenn Bilder die "
                f"Validierungskriterien nicht erfüllen)"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            if msg.exec() == QMessageBox.StandardButton.Cancel:
                return

            # Reset progress
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(image_files))
            
            valid_augmentations = 0
            invalid_augmentations = 0

            # Perform augmentation
            for i, image_file in enumerate(image_files):
                # Update progress for each source image
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()

                image = cv2.imread(str(image_file))
                label_file = label_files.get(image_file.stem)
                boxes = []

                if label_file:
                    with open(label_file, 'r') as f:
                        try:
                            boxes = [list(map(float, line.strip().split())) for line in f]
                        except ValueError as e:
                            logger.error(f"Fehler beim Parsen der Labels in {label_file}: {e}")
                            continue

                for combination in total_combinations:
                    augmented_image = image.copy()
                    augmented_boxes = boxes.copy()
                    valid_augmentation = True
                    augmented = False  # Track if any augmentation was applied
                    output_suffix = []

                    for method_idx, level in enumerate(combination):
                        method, level1, level2 = selected_methods[method_idx]
                        if level == 1:
                            augmented = True
                            augmented_image, augmented_boxes = augment_image_with_boxes(
                                augmented_image, augmented_boxes, method, 0, level1,
                                min_visibility=self.settings.get('min_visibility', 0.3),
                                min_size=self.settings.get('min_size', 20))
                            output_suffix.append(f"{method}_L1")
                        elif level == 2:
                            augmented = True
                            augmented_image, augmented_boxes = augment_image_with_boxes(
                                augmented_image, augmented_boxes, method, level1, level2,
                                min_visibility=self.settings.get('min_visibility', 0.3),
                                min_size=self.settings.get('min_size', 20))
                            output_suffix.append(f"{method}_L2")

                    # Apply horizontal flip if enabled (50% chance)
                    if self.flip_checkbox.isChecked() and np.random.random() < 0.5:
                        augmented = True
                        augmented_image, augmented_boxes = augment_image_with_boxes(
                            augmented_image, augmented_boxes, "HorizontalFlip", 0, 0,
                            min_visibility=self.settings.get('min_visibility', 0.3),
                            min_size=self.settings.get('min_size', 20))
                        output_suffix.append("Flipped")
                        # Skip if no valid boxes remain after augmentation
                        if not augmented_boxes and boxes:  # Only count as invalid if original had boxes
                            valid_augmentation = False
                            invalid_augmentations += 1
                            break

                        # Show preview of augmented image
                        if augmented_image is not None:
                            # Convert to RGB for Qt
                            preview_img = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                            
                            # Get preview label size
                            preview_width = min(800, self.preview_label.width())  # Limit max width
                            preview_height = min(600, self.preview_label.height()) # Limit max height
                            
                            # Calculate aspect ratio preserving scale
                            img_h, img_w = preview_img.shape[:2]
                            scale = min(preview_width/img_w, preview_height/img_h)
                            
                            # Calculate new size
                            new_w = int(img_w * scale)
                            new_h = int(img_h * scale)
                            
                            # Resize image
                            preview_img = cv2.resize(preview_img, (new_w, new_h), 
                                                   interpolation=cv2.INTER_AREA)
                            
                            # Convert to QImage and display
                            qimg = QImage(preview_img.data, new_w, new_h, 
                                        new_w * 3, QImage.Format.Format_RGB888)
                            self.preview_label.setPixmap(QPixmap.fromImage(qimg))
                            QApplication.processEvents()

                    # Only save if augmentation was applied and validation passed
                    if not valid_augmentation or not augmented:
                        continue

                    output_image_path = Path(self.dest_path) / \
                        f"{image_file.stem}_{'_'.join(output_suffix)}.jpg"
                    output_label_path = Path(self.dest_path) / \
                        f"{image_file.stem}_{'_'.join(output_suffix)}.txt"

                    # Save augmented image and labels
                    cv2.imwrite(str(output_image_path), augmented_image)
                    
                    with open(output_label_path, 'w') as f:
                        for box in augmented_boxes:
                            f.write(' '.join(map(str, box)) + '\n')

                    valid_augmentations += 1

            self.progress_bar.setValue(100)

            # Show final results
            QMessageBox.information(
                self,
                "Augmentierung abgeschlossen",
                f"Augmentierung erfolgreich abgeschlossen!\n\n"
                f"Verarbeitet: {len(image_files)} Bilder\n"
                f"Gültige Augmentierungen: {valid_augmentations}\n"
                f"Ungültige Augmentierungen: {invalid_augmentations}"
            )
            
        except Exception as e:
            logger.critical(f"Unbehandelter Fehler: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Fehler",
                f"Ein unerwarteter Fehler ist aufgetreten: {str(e)}"
            )