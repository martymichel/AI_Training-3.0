"""Image augmentation application module."""

import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QGroupBox, QCheckBox,
    QFileDialog, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt
import logging
from utils.augmentation_utils import augment_image_with_boxes
from itertools import product

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageAugmentationApp(QMainWindow):
    """Main window for the image augmentation application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Augmentation Tool")
        self.setGeometry(100, 100, 800, 600)

        # Zentral-Widget und Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Source directory
        self.source_label = QLabel("Quellverzeichnis:")
        self.source_path_button = QPushButton("Durchsuchen")
        self.source_path_button.clicked.connect(self.browse_source)
        self.source_layout = QHBoxLayout()
        self.source_layout.addWidget(self.source_label)
        self.source_layout.addWidget(self.source_path_button)

        # Destination directory
        self.dest_label = QLabel("Zielverzeichnis:")
        self.dest_path_button = QPushButton("Durchsuchen")
        self.dest_path_button.clicked.connect(self.browse_dest)
        self.dest_layout = QHBoxLayout()
        self.dest_layout.addWidget(self.dest_label)
        self.dest_layout.addWidget(self.dest_path_button)

        # Number of augmentations selection
        self.num_augmentations_label = QLabel("Anzahl Augmentierungen:")
        self.num_augmentations_spinbox = QSpinBox()
        self.num_augmentations_spinbox.setRange(3, 5)
        self.num_augmentations_spinbox.setValue(3)
        self.num_augmentations_spinbox.valueChanged.connect(self.update_method_activation)
        self.num_augmentations_layout = QHBoxLayout()
        self.num_augmentations_layout.addWidget(self.num_augmentations_label)
        self.num_augmentations_layout.addWidget(self.num_augmentations_spinbox)

        # Augmentation method selection
        self.method_group = QGroupBox("Augmentierungsmethoden auswählen")
        self.method_layout = QVBoxLayout()
        self.method_group.setLayout(self.method_layout)

        # Augmentation methods
        self.methods = ["Verschiebung", "Rotation", "Zoom", "Helligkeit", "Unschärfe"]
        self.method_checkboxes = []
        self.method_levels = {}

        for method in self.methods:
            checkbox = QCheckBox(method)
            checkbox.stateChanged.connect(self.update_method_activation)
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

        # Progress bar
        self.progress_bar = QProgressBar()

        # Start button
        self.start_button = QPushButton("Augmentierung starten")
        self.start_button.clicked.connect(self.start_augmentation)

        # Add widgets to layout
        self.layout.addLayout(self.source_layout)
        self.layout.addLayout(self.dest_layout)
        self.layout.addLayout(self.num_augmentations_layout)
        self.layout.addWidget(self.method_group)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.start_button)

        # Paths
        self.source_path = None
        self.dest_path = None

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

    def browse_dest(self):
        """Open file dialog to select destination directory."""
        path = QFileDialog.getExistingDirectory(self, "Zielverzeichnis auswählen")
        if path:
            self.dest_path = path
            self.dest_label.setText(f"Zielverzeichnis: {path}")

    def update_method_activation(self):
        """Update which methods can be selected based on max active methods."""
        max_active = self.num_augmentations_spinbox.value()
        active_methods = sum(1 for checkbox in self.method_checkboxes if checkbox.isChecked())

        for checkbox in self.method_checkboxes:
            if not checkbox.isChecked() and active_methods >= max_active:
                checkbox.setEnabled(False)
            else:
                checkbox.setEnabled(True)

    def start_augmentation(self):
        """Start the augmentation process."""
        try:
            if not self.source_path or not self.dest_path:
                QMessageBox.warning(self, "Fehler", 
                                  "Bitte wählen Sie Quell- und Zielverzeichnis aus.")
                return

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

            # Collect images and labels
            image_files = list(Path(self.source_path).rglob("*.jpg")) + \
                         list(Path(self.source_path).rglob("*.png"))
            label_files = {file.stem: file for file in Path(self.source_path).rglob("*.txt")}

            if not image_files:
                QMessageBox.warning(self, "Fehler", 
                                  "Keine Bilder im Quellverzeichnis gefunden.")
                return

            total_combinations = list(product([0, 1, 2], repeat=len(selected_methods)))
            self.progress_bar.setMaximum(len(image_files) * len(total_combinations))
            self.progress_bar.setValue(0)

            total_expected_files = len(image_files) * len(total_combinations)
            generated_files = 0

            # Perform augmentation
            for i, image_file in enumerate(image_files):
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
                    output_suffix = []

                    for method_idx, level in enumerate(combination):
                        method, level1, level2 = selected_methods[method_idx]
                        if level == 1:
                            augmented_image, augmented_boxes = augment_image_with_boxes(
                                augmented_image, augmented_boxes, method, 0, level1)
                            output_suffix.append(f"{method}_L1")
                        elif level == 2:
                            augmented_image, augmented_boxes = augment_image_with_boxes(
                                augmented_image, augmented_boxes, method, level1, level2)
                            output_suffix.append(f"{method}_L2")

                    output_image_path = Path(self.dest_path) / \
                        f"{image_file.stem}_{'_'.join(output_suffix)}.jpg"
                    output_label_path = Path(self.dest_path) / \
                        f"{image_file.stem}_{'_'.join(output_suffix)}.txt"

                    # Save augmented image and labels
                    cv2.imwrite(str(output_image_path), augmented_image)
                    
                    with open(output_label_path, 'w') as f:
                        for box in augmented_boxes:
                            f.write(' '.join(map(str, box)) + '\n')

                    generated_files += 1
                    self.progress_bar.setValue(self.progress_bar.value() + 1)

            if generated_files != total_expected_files:
                logger.warning(
                    f"Erwartet: {total_expected_files} Dateien, "
                    f"aber nur {generated_files} wurden generiert."
                )
                QMessageBox.warning(
                    self, "Unvollständige Augmentierung",
                    "Einige Dateien wurden nicht erfolgreich augmentiert. "
                    "Prüfen Sie die Logs für Details."
                )
            else:
                QMessageBox.information(
                    self, "Erfolg",
                    "Augmentierung erfolgreich abgeschlossen!"
                )

        except Exception as e:
            logger.critical(f"Unbehandelter Fehler: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Fehler",
                f"Ein unerwarteter Fehler ist aufgetreten: {str(e)}"
            )