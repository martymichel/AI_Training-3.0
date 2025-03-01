"""Dataset splitter application module."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QProgressBar, QMessageBox,
    QSpinBox, QGroupBox, QLineEdit, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap
import cv2
from utils.dataset_utils import DatasetSplitter, DatasetSplitterThread
import logging
import numpy as np
from pathlib import Path

class DatasetSplitterApp(QMainWindow):
    """Main window for the dataset splitter application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Splitter")
        self.setGeometry(100, 100, 800, 900)
        
        # Initialize splitter
        self.splitter = DatasetSplitter()
        
        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        
        # Class naming section
        self.class_group = QGroupBox("Class Names")
        self.class_layout = QVBoxLayout()
        self.class_group.setLayout(self.class_layout)
        
        # Preview image
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(800, 600)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.class_layout.addWidget(self.preview_label)
        
        # Scroll area for class inputs
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumSize(400, 100)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.class_layout.addWidget(self.scroll_area)
        
        self.class_inputs = {}  # Store class input widgets
        self.current_image = None
        self.current_boxes = None
        
        # Directory selection
        self.create_directory_group()
        
        # Add class naming section after directory selection
        self.layout.addWidget(self.class_group)
        self.class_group.setVisible(False)
        
        # Split ratios
        self.create_split_ratio_group()
        
        # Progress display
        self.progress_text = QLabel("Ready to split dataset")
        self.progress_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.progress_text)
        
        # Start button
        self.start_button = QPushButton("Start Splitting")
        self.start_button.setStyleSheet("background-color: blue; color: white; font-size: 16px")
        self.start_button.clicked.connect(self.start_splitting)
        self.layout.addWidget(self.start_button)
        
        # Add stretch at the bottom
        self.layout.addStretch()
        
        # Initialize thread
        self.split_thread = None

    def create_directory_group(self):
        """Create the directory selection group."""
        dir_group = QGroupBox("Directory Selection")
        dir_layout = QVBoxLayout()
        
        # Source directory
        source_layout = QHBoxLayout()
        self.source_label = QLabel("Source Directory:")
        self.source_path = QLabel("Not selected")
        self.source_button = QPushButton("Browse")
        self.source_button.clicked.connect(self.browse_source)
        source_layout.addWidget(self.source_label)
        source_layout.addWidget(self.source_path)
        source_layout.addWidget(self.source_button)
        dir_layout.addLayout(source_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_path = QLabel("Not selected")
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_button)
        dir_layout.addLayout(output_layout)
        
        dir_group.setLayout(dir_layout)
        self.layout.addWidget(dir_group)

    def create_split_ratio_group(self):
        """Create the split ratio selection group."""
        ratio_group = QGroupBox("Split Ratios")
        ratio_layout = QHBoxLayout()
        
        # Train split
        train_layout = QVBoxLayout()
        self.train_label = QLabel("Train %:")
        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(70)
        self.train_spin.valueChanged.connect(self.update_splits)
        train_layout.addWidget(self.train_label)
        train_layout.addWidget(self.train_spin)
        ratio_layout.addLayout(train_layout)
        
        # Validation split
        val_layout = QVBoxLayout()
        self.val_label = QLabel("Validation %:")
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(20)
        self.val_spin.valueChanged.connect(self.update_splits)
        val_layout.addWidget(self.val_label)
        val_layout.addWidget(self.val_spin)
        ratio_layout.addLayout(val_layout)
        
        # Test split (calculated)
        test_layout = QVBoxLayout()
        self.test_label = QLabel("Test %:")
        self.test_value = QLabel("10")
        test_layout.addWidget(self.test_label)
        test_layout.addWidget(self.test_value)
        ratio_layout.addLayout(test_layout)
        
        ratio_group.setLayout(ratio_layout)
        self.layout.addWidget(ratio_group)

    def update_splits(self):
        """Update split percentages ensuring they sum to 100."""
        train = self.train_spin.value()
        val = self.val_spin.value()
        test = 100 - train - val
        self.test_value.setText(str(test))
        
        # Validate total
        if test < 0:
            self.start_button.setEnabled(False)
            self.progress_text.setText("Error: Split percentages must sum to 100")
        else:
            self.start_button.setEnabled(True)
            self.progress_text.setText("Ready to split dataset")

    def browse_source(self):
        """Open file dialog to select source directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if dir_path:
            self.source_path.setText(dir_path)
            self.analyze_classes(dir_path)

    def analyze_classes(self, source_dir):
        """Analyze dataset to detect classes and show naming interface."""
        try:
            # Find image-label pairs
            valid_pairs = self.splitter.find_image_label_pairs(source_dir)
            if not valid_pairs:
                raise ValueError("No valid image-label pairs found")

            # Split pairs into images and labels
            image_files = [pair[0] for pair in valid_pairs]
            label_files = [pair[1] for pair in valid_pairs]

            # Detect classes
            classes = self.splitter.detect_classes(label_files)
            if not classes:
                raise ValueError("No classes found in label files")

            # Find sample image
            sample_image, sample_label = self.splitter.find_sample_image(image_files, label_files)
            if sample_image is None:
                raise ValueError("Could not find suitable sample image")

            # Load and display sample image
            self.current_image, self.current_boxes = self.splitter.load_image_with_boxes(
                sample_image, sample_label
            )
            self.update_preview()

            # Create class input fields
            self.create_class_inputs(classes)
            self.class_group.setVisible(True)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to analyze dataset: {str(e)}")

    def create_class_inputs(self, classes):
        """Create input fields for class names."""
        # Clear existing inputs
        for widget in self.class_inputs.values():
            widget.setParent(None)
        self.class_inputs.clear()

        # Create new inputs
        for class_id in sorted(classes):
            layout = QHBoxLayout()
            label = QLabel(f"Class {class_id}:")
            input_field = QLineEdit()
            input_field.setPlaceholderText(f"Enter name for class {class_id}")
            # Highlight on focus instead of text change
            input_field.focusInEvent = lambda e, cid=class_id: self.highlight_class(cid)
            input_field.focusOutEvent = lambda e: self.update_preview()
            
            layout.addWidget(label)
            layout.addWidget(input_field)
            self.scroll_layout.addLayout(layout)
            self.class_inputs[class_id] = input_field

    def highlight_class(self, class_id):
        """Highlight boxes of selected class."""
        if self.current_image is not None and self.current_boxes is not None:
            img_copy = self.current_image.copy()
            
            for box_class, x1, y1, x2, y2 in self.current_boxes:
                # Determine color and opacity based on whether this is the highlighted class
                if box_class == class_id:
                    color = (0, 255, 0)  # Green for highlighted class
                    alpha = 0.9
                else:
                    color = (64, 64, 64)  # Gray for other classes
                    alpha = 0.75
                
                # Draw semi-transparent box
                overlay = img_copy.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)
            
            self.update_preview(img_copy)

    def update_preview(self, image=None):
        """Update the preview image."""
        if image is None and self.current_image is not None:
            image = self.current_image.copy()
        
        if image is not None:
            height, width = image.shape[:2]
            scale = min(800 / width, 600 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            image = cv2.resize(image, (new_width, new_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            self.preview_label.setPixmap(QPixmap.fromImage(qt_image))

    def browse_output(self):
        """Open file dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path.setText(dir_path)

    def update_progress(self, message):
        """Update progress display."""
        self.progress_text.setText(message)

    def splitting_finished(self, success, message):
        """Handle completion of splitting operation."""
        self.start_button.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", f"Failed to split dataset: {message}")

    def start_splitting(self):
        """Start the dataset splitting process."""
        source_dir = self.source_path.text()
        output_dir = self.output_path.text()
        
        # Get class names from input fields
        class_names = {}
        for class_id, input_field in self.class_inputs.items():
            name = input_field.text().strip()
            if name:  # Only include named classes
                class_names[class_id] = name
            else:
                class_names[class_id] = f"Class_{class_id}"  # Default name if not specified
        
        if source_dir == "Not selected" or output_dir == "Not selected":
            QMessageBox.warning(self, "Error", "Please select both source and output directories")
            return
        
        train_split = self.train_spin.value()
        val_split = self.val_spin.value()
        test_split = 100 - train_split - val_split
        
        if test_split < 0:
            QMessageBox.warning(self, "Error", "Invalid split percentages")
            return
        
        self.start_button.setEnabled(False)
        self.progress_text.setText("Starting dataset split...")
        
        # Start splitting in thread
        self.split_thread = DatasetSplitterThread(
            self.splitter, source_dir, output_dir, train_split, val_split, class_names
        )
        self.split_thread.progress.connect(self.update_progress)
        self.split_thread.finished.connect(self.splitting_finished)
        self.split_thread.start()