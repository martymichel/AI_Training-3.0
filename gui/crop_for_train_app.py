"""
Crop for Train App
==================
Extracts individual part crops (BODEN, MANTEL, GEWINDE) from images using a trained detection model
to prepare training data for segmentation models.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QMessageBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QListWidget, QListWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QScrollArea, QCheckBox, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QFont

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from project_manager import ProjectManager, WorkflowStep


@dataclass
class CropConfig:
    """Configuration for cropping operations."""
    confidence_threshold: float = 0.5
    padding_pixels: int = 20
    min_crop_size: int = 50
    max_crops_per_class: int = 1000


@dataclass
class DetectedObject:
    """Represents a detected object with its bounding box and metadata."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    image_path: str
    crop_id: int


class CropWorkerThread(QThread):
    """Worker thread for processing images and extracting crops."""

    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    crop_found = pyqtSignal(DetectedObject, np.ndarray)  # detected_object, crop_image
    finished = pyqtSignal(bool, str)

    def __init__(self, image_dir: str, model_path: str, config: CropConfig, class_names: Dict[int, str]):
        super().__init__()
        self.image_dir = image_dir
        self.model_path = model_path
        self.config = config
        self.class_names = class_names
        self.should_stop = False

    def stop(self):
        """Stop the worker thread."""
        self.should_stop = True

    def run(self):
        """Main processing loop."""
        try:
            if not YOLO_AVAILABLE:
                self.finished.emit(False, "YOLO not available. Please install ultralytics.")
                return

            self.progress.emit("Loading detection model...")
            model = YOLO(self.model_path)

            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(self.image_dir).glob(f"*{ext}"))
                image_files.extend(Path(self.image_dir).glob(f"*{ext.upper()}"))

            if not image_files:
                self.finished.emit(False, "No image files found in source directory.")
                return

            total_files = len(image_files)
            self.progress.emit(f"Found {total_files} images to process")

            crops_per_class = {class_id: 0 for class_id in self.class_names.keys()}
            crop_counter = 0

            for i, image_path in enumerate(image_files):
                if self.should_stop:
                    break

                self.progress.emit(f"Processing {image_path.name}...")
                self.progress_value.emit(int((i / total_files) * 100))

                try:
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue

                    # Run detection
                    results = model(image, conf=self.config.confidence_threshold)

                    for result in results:
                        if result.boxes is None:
                            continue

                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)

                        for box, conf, class_id in zip(boxes, confidences, class_ids):
                            if class_id not in self.class_names:
                                continue

                            if crops_per_class[class_id] >= self.config.max_crops_per_class:
                                continue

                            x1, y1, x2, y2 = map(int, box)

                            # Add padding
                            h, w = image.shape[:2]
                            x1 = max(0, x1 - self.config.padding_pixels)
                            y1 = max(0, y1 - self.config.padding_pixels)
                            x2 = min(w, x2 + self.config.padding_pixels)
                            y2 = min(h, y2 + self.config.padding_pixels)

                            # Check minimum size
                            if (x2 - x1) < self.config.min_crop_size or (y2 - y1) < self.config.min_crop_size:
                                continue

                            # Extract crop
                            crop = image[y1:y2, x1:x2]

                            detected_obj = DetectedObject(
                                class_id=class_id,
                                class_name=self.class_names[class_id],
                                confidence=conf,
                                bbox=(x1, y1, x2, y2),
                                image_path=str(image_path),
                                crop_id=crop_counter
                            )

                            self.crop_found.emit(detected_obj, crop)
                            crops_per_class[class_id] += 1
                            crop_counter += 1

                except Exception as e:
                    self.progress.emit(f"Error processing {image_path.name}: {str(e)}")
                    continue

            self.progress_value.emit(100)

            total_crops = sum(crops_per_class.values())
            summary = f"Processing complete! Found {total_crops} crops:\n"
            for class_id, count in crops_per_class.items():
                summary += f"- {self.class_names[class_id]}: {count}\n"

            self.finished.emit(True, summary)

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class LivePreviewFrame(QWidget):
    """Widget to show live preview of a specific class during processing."""

    def __init__(self, class_name: str):
        super().__init__()
        self.class_name = class_name
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel(f"{self.class_name} Live Preview")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)

        # Preview area
        self.preview_scene = QGraphicsScene()
        self.preview_view = QGraphicsView(self.preview_scene)
        self.preview_view.setMinimumSize(200, 150)
        self.preview_view.setMaximumSize(300, 225)
        layout.addWidget(self.preview_view)

        # Status label
        self.status_label = QLabel("Waiting for crops...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def update_preview(self, crop_image: np.ndarray, detected_obj: DetectedObject):
        """Update the preview with a new crop image."""
        try:
            # Convert crop to QPixmap
            height, width, channel = crop_image.shape
            bytes_per_line = 3 * width

            # Convert numpy array to bytes for QImage
            crop_image_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            crop_bytes = crop_image_rgb.tobytes()

            q_image = QImage(crop_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # Update scene
            self.preview_scene.clear()
            self.preview_scene.addPixmap(pixmap)
            self.preview_view.fitInView(self.preview_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

            # Update status
            self.status_label.setText(f"Conf: {detected_obj.confidence:.3f} | Size: {width}x{height}")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def clear_preview(self):
        """Clear the preview."""
        self.preview_scene.clear()
        self.status_label.setText("Waiting for crops...")


class CropPreviewWidget(QWidget):
    """Widget to preview and manage detected crops."""

    crop_accepted = pyqtSignal(DetectedObject, np.ndarray)
    crop_rejected = pyqtSignal(DetectedObject)

    def __init__(self):
        super().__init__()
        self.pending_crops = []  # List of (DetectedObject, np.ndarray) tuples
        self.current_index = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Live preview frames section
        live_preview_group = QGroupBox("Live Preview")
        live_layout = QHBoxLayout()

        self.live_frames = {}
        class_names = ["BODEN", "MANTEL", "GEWINDE"]

        for class_name in class_names:
            frame = LivePreviewFrame(class_name)
            self.live_frames[class_name] = frame
            live_layout.addWidget(frame)

        live_preview_group.setLayout(live_layout)
        layout.addWidget(live_preview_group)

        # Info label
        self.info_label = QLabel("No crops to preview")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        # Preview area
        self.preview_scene = QGraphicsScene()
        self.preview_view = QGraphicsView(self.preview_scene)
        self.preview_view.setMinimumSize(400, 300)
        layout.addWidget(self.preview_view)

        # Navigation and action buttons
        nav_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_button)

        self.current_label = QLabel("0 / 0")
        self.current_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.current_label)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)

        # Action buttons
        action_layout = QHBoxLayout()

        self.accept_button = QPushButton("Accept Crop")
        self.accept_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.accept_button.clicked.connect(self.accept_current)
        action_layout.addWidget(self.accept_button)

        self.reject_button = QPushButton("Reject Crop")
        self.reject_button.setStyleSheet("background-color: #f44336; color: white;")
        self.reject_button.clicked.connect(self.reject_current)
        action_layout.addWidget(self.reject_button)

        self.accept_all_button = QPushButton("Accept All")
        self.accept_all_button.clicked.connect(self.accept_all)
        action_layout.addWidget(self.accept_all_button)

        layout.addLayout(action_layout)

        self.setLayout(layout)
        self.update_ui()

    def add_crop(self, detected_obj: DetectedObject, crop_image: np.ndarray):
        """Add a new crop for preview."""
        # Update live preview frame for this class
        if detected_obj.class_name in self.live_frames:
            self.live_frames[detected_obj.class_name].update_preview(crop_image, detected_obj)

        self.pending_crops.append((detected_obj, crop_image))
        self.update_ui()
        if len(self.pending_crops) == 1:
            self.show_current()

    def update_ui(self):
        """Update UI state based on pending crops."""
        has_crops = len(self.pending_crops) > 0

        self.prev_button.setEnabled(has_crops and self.current_index > 0)
        self.next_button.setEnabled(has_crops and self.current_index < len(self.pending_crops) - 1)
        self.accept_button.setEnabled(has_crops)
        self.reject_button.setEnabled(has_crops)
        self.accept_all_button.setEnabled(has_crops)

        if has_crops:
            self.current_label.setText(f"{self.current_index + 1} / {len(self.pending_crops)}")
            self.info_label.setText(f"Previewing crops ({len(self.pending_crops)} pending)")
        else:
            self.current_label.setText("0 / 0")
            self.info_label.setText("No crops to preview")

    def show_current(self):
        """Show the current crop in the preview."""
        if not self.pending_crops or self.current_index >= len(self.pending_crops):
            return

        detected_obj, crop_image = self.pending_crops[self.current_index]

        # Convert crop to QPixmap
        height, width, channel = crop_image.shape
        bytes_per_line = 3 * width

        # Convert numpy array to bytes for QImage
        crop_image_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        crop_bytes = crop_image_rgb.tobytes()

        q_image = QImage(crop_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Update scene
        self.preview_scene.clear()
        self.preview_scene.addPixmap(pixmap)
        self.preview_view.fitInView(self.preview_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Update info
        info_text = (f"Class: {detected_obj.class_name}\n"
                    f"Confidence: {detected_obj.confidence:.3f}\n"
                    f"Size: {crop_image.shape[1]}x{crop_image.shape[0]}\n"
                    f"Source: {Path(detected_obj.image_path).name}")
        self.info_label.setText(info_text)

    def show_previous(self):
        """Show previous crop."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current()
            self.update_ui()

    def show_next(self):
        """Show next crop."""
        if self.current_index < len(self.pending_crops) - 1:
            self.current_index += 1
            self.show_current()
            self.update_ui()

    def accept_current(self):
        """Accept the current crop."""
        if self.pending_crops:
            detected_obj, crop_image = self.pending_crops.pop(self.current_index)
            self.crop_accepted.emit(detected_obj, crop_image)

            # Adjust current index
            if self.current_index >= len(self.pending_crops):
                self.current_index = max(0, len(self.pending_crops) - 1)

            self.update_ui()
            if self.pending_crops:
                self.show_current()
            else:
                self.preview_scene.clear()

    def reject_current(self):
        """Reject the current crop."""
        if self.pending_crops:
            detected_obj, _ = self.pending_crops.pop(self.current_index)
            self.crop_rejected.emit(detected_obj)

            # Adjust current index
            if self.current_index >= len(self.pending_crops):
                self.current_index = max(0, len(self.pending_crops) - 1)

            self.update_ui()
            if self.pending_crops:
                self.show_current()
            else:
                self.preview_scene.clear()

    def accept_all(self):
        """Accept all pending crops."""
        while self.pending_crops:
            detected_obj, crop_image = self.pending_crops.pop(0)
            self.crop_accepted.emit(detected_obj, crop_image)

        self.current_index = 0
        self.update_ui()
        self.preview_scene.clear()


class CropForTrainApp(QMainWindow):
    """Main application for cropping training data from detected objects."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crop for Train - Segmentation Training Data Preparation")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Configuration
        self.config = CropConfig()
        self.class_names = {0: "BODEN", 1: "MANTEL", 2: "GEWINDE"}

        # Worker thread
        self.worker_thread = None

        # Accepted crops storage
        self.accepted_crops = {class_id: [] for class_id in self.class_names.keys()}

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Configuration and controls
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # Configuration group
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()

        # Directory selection
        dir_layout = QVBoxLayout()

        # Source directory
        source_layout = QHBoxLayout()
        self.source_label = QLabel("Source Directory:")
        self.source_path = QLabel("Not selected")
        self.source_path.setStyleSheet("border: 1px solid gray; padding: 4px;")
        source_button = QPushButton("Browse")
        source_button.clicked.connect(self.select_source_directory)
        source_layout.addWidget(self.source_label)
        source_layout.addWidget(self.source_path, 1)
        source_layout.addWidget(source_button)
        dir_layout.addLayout(source_layout)

        # Model path
        model_layout = QHBoxLayout()
        self.model_label = QLabel("Detection Model:")
        self.model_path = QLabel("Not selected")
        self.model_path.setStyleSheet("border: 1px solid gray; padding: 4px;")
        model_button = QPushButton("Browse")
        model_button.clicked.connect(self.select_model_file)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_path, 1)
        model_layout.addWidget(model_button)
        dir_layout.addLayout(model_layout)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_path = QLabel("Not selected")
        self.output_path.setStyleSheet("border: 1px solid gray; padding: 4px;")
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path, 1)
        output_layout.addWidget(output_button)
        dir_layout.addLayout(output_layout)

        config_layout.addLayout(dir_layout)

        # Parameters
        params_layout = QVBoxLayout()

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.1, 1.0)
        self.conf_spinbox.setValue(self.config.confidence_threshold)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.valueChanged.connect(self.update_config)
        conf_layout.addWidget(self.conf_spinbox)
        params_layout.addLayout(conf_layout)

        # Padding
        padding_layout = QHBoxLayout()
        padding_layout.addWidget(QLabel("Padding (pixels):"))
        self.padding_spinbox = QSpinBox()
        self.padding_spinbox.setRange(0, 100)
        self.padding_spinbox.setValue(self.config.padding_pixels)
        self.padding_spinbox.valueChanged.connect(self.update_config)
        padding_layout.addWidget(self.padding_spinbox)
        params_layout.addLayout(padding_layout)

        # Max crops per class
        max_crops_layout = QHBoxLayout()
        max_crops_layout.addWidget(QLabel("Max crops per class:"))
        self.max_crops_spinbox = QSpinBox()
        self.max_crops_spinbox.setRange(1, 10000)
        self.max_crops_spinbox.setValue(self.config.max_crops_per_class)
        self.max_crops_spinbox.valueChanged.connect(self.update_config)
        max_crops_layout.addWidget(self.max_crops_spinbox)
        params_layout.addLayout(max_crops_layout)

        config_layout.addLayout(params_layout)
        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        # Control buttons
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()

        self.start_button = QPushButton("Start Processing")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 10px;")
        self.start_button.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; font-size: 14px; padding: 10px;")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        # Save button removed - auto-save is now implemented

        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_label = QLabel("Ready to start")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        progress_group.setLayout(progress_layout)
        left_layout.addWidget(progress_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        self.update_statistics()
        stats_layout.addWidget(self.stats_text)

        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Right panel - Preview
        self.crop_preview = CropPreviewWidget()
        self.crop_preview.crop_accepted.connect(self.on_crop_accepted)
        self.crop_preview.crop_rejected.connect(self.on_crop_rejected)
        splitter.addWidget(self.crop_preview)

        # Set splitter proportions
        splitter.setSizes([350, 1050])

    def update_config(self):
        """Update configuration from UI controls."""
        self.config.confidence_threshold = self.conf_spinbox.value()
        self.config.padding_pixels = self.padding_spinbox.value()
        self.config.max_crops_per_class = self.max_crops_spinbox.value()

    def select_source_directory(self):
        """Select source directory containing images."""
        directory = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if directory:
            self.source_path.setText(directory)

    def select_model_file(self):
        """Select detection model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Detection Model", "", "Model Files (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            self.model_path.setText(file_path)

    def select_output_directory(self):
        """Select output directory for crops."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path.setText(directory)

    def start_processing(self):
        """Start the crop processing."""
        # Validate inputs
        if self.source_path.text() == "Not selected":
            QMessageBox.warning(self, "Error", "Please select a source directory.")
            return

        if self.model_path.text() == "Not selected":
            QMessageBox.warning(self, "Error", "Please select a detection model.")
            return

        if self.output_path.text() == "Not selected":
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return

        if not os.path.exists(self.source_path.text()):
            QMessageBox.warning(self, "Error", "Source directory does not exist.")
            return

        if not os.path.exists(self.model_path.text()):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return

        # Start worker thread
        self.worker_thread = CropWorkerThread(
            self.source_path.text(),
            self.model_path.text(),
            self.config,
            self.class_names
        )

        self.worker_thread.progress.connect(self.on_progress)
        self.worker_thread.progress_value.connect(self.progress_bar.setValue)
        self.worker_thread.crop_found.connect(self.on_crop_found)
        self.worker_thread.finished.connect(self.on_processing_finished)

        self.worker_thread.start()

        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_processing(self):
        """Stop the crop processing."""
        if self.worker_thread:
            self.worker_thread.stop()
            self.worker_thread.wait()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_label.setText("Processing stopped")

    def on_progress(self, message: str):
        """Handle progress updates."""
        self.progress_label.setText(message)

    def on_crop_found(self, detected_obj: DetectedObject, crop_image: np.ndarray):
        """Handle a new crop found by the worker."""
        self.crop_preview.add_crop(detected_obj, crop_image)

    def on_crop_accepted(self, detected_obj: DetectedObject, crop_image: np.ndarray):
        """Handle acceptance of a crop."""
        self.accepted_crops[detected_obj.class_id].append((detected_obj, crop_image))
        self.update_statistics()

        # Auto-save the accepted crop immediately
        self.auto_save_crop(detected_obj, crop_image)

    def on_crop_rejected(self, detected_obj: DetectedObject):
        """Handle rejection of a crop."""
        # Just log the rejection, nothing else needed
        pass

    def on_processing_finished(self, success: bool, message: str):
        """Handle completion of processing."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if success:
            self.progress_label.setText("Processing complete!")
            QMessageBox.information(self, "Complete", message)
        else:
            self.progress_label.setText("Processing failed!")
            QMessageBox.critical(self, "Error", message)

    def update_statistics(self):
        """Update the statistics display."""
        stats = []
        total_accepted = 0

        for class_id, class_name in self.class_names.items():
            count = len(self.accepted_crops[class_id])
            stats.append(f"{class_name}: {count} crops")
            total_accepted += count

        stats.insert(0, f"Total accepted: {total_accepted}")
        self.stats_text.setText("\n".join(stats))

    def auto_save_crop(self, detected_obj: DetectedObject, crop_image: np.ndarray):
        """Automatically save a single accepted crop."""
        try:
            output_dir = Path(self.output_path.text())

            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            class_name = detected_obj.class_name
            class_dir = output_dir / class_name.lower()
            class_dir.mkdir(exist_ok=True)

            # Generate unique filename
            source_name = Path(detected_obj.image_path).stem
            existing_files = list(class_dir.glob(f"{source_name}_{class_name.lower()}_*.png"))
            crop_number = len(existing_files)
            filename = f"{source_name}_{class_name.lower()}_{crop_number:03d}.png"
            filepath = class_dir / filename

            # Save crop as PNG
            cv2.imwrite(str(filepath), crop_image)

            # Update progress label to show auto-save
            self.progress_label.setText(f"Auto-saved: {filename}")

        except Exception as e:
            QMessageBox.warning(self, "Auto-Save Error", f"Failed to auto-save crop: {str(e)}")

    def save_crops(self):
        """Legacy method - now crops are auto-saved individually."""
        # This method is kept for compatibility but crops are now auto-saved
        QMessageBox.information(
            self, "Info",
            "Crops are now automatically saved when accepted. No manual save needed."
        )

    def closeEvent(self, event):
        """Handle application close."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    if not YOLO_AVAILABLE:
        QMessageBox.critical(None, "Error", "YOLO not available. Please install ultralytics:\npip install ultralytics")
        sys.exit(1)

    window = CropForTrainApp()
    window.show()

    sys.exit(app.exec())