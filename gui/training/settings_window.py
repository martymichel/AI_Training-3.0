"""Training settings window with project management integration."""

import sys
import os
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QProgressBar, QMessageBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit,
    QTextEdit, QTabWidget, QGroupBox, QFormLayout, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from gui.training.training_thread import TrainingSignals, start_training_thread, stop_training
from gui.training.dashboard_view import create_dashboard_tabs, update_dashboard_plots
from gui.training.training_utils import check_and_load_results_csv
from gui.training.parameter_info import ParameterInfoButton
from utils.validation import validate_yaml_for_model_type, check_gpu
from project_manager import WorkflowStep

# Configure logging
logger = logging.getLogger(__name__)

class TrainSettingsWindow(QMainWindow):
    """Main training settings window with integrated dashboard."""
    
    def __init__(self, project_manager=None):
        super().__init__()
        self.project_manager = project_manager
        self.setWindowTitle("YOLO Training Settings")
        self.setGeometry(100, 100, 1400, 800)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
        # Fixed independent styling - readable dark text on light background
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
                color: #2c3e50;
            }
            QWidget {
                background-color: #ffffff;
                color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QLabel {
                color: #2c3e50;
                background: transparent;
                padding: 2px;
                font-weight: 500;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #ffffff;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 6px;
                color: #2c3e50;
                font-size: 13px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #3498db;
                background-color: #f8f9fa;
            }
            QPushButton {
                background-color: #3498db;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                font-weight: 600;
                font-size: 13px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #ecf0f1;
            }
            QCheckBox {
                color: #2c3e50;
                font-weight: 500;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #bdc3c7;
                border-radius: 2px;
                background-color: #ffffff;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border-color: #3498db;
            }
            QProgressBar {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                text-align: center;
                color: #2c3e50;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                color: #2c3e50;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: #ffffff;
            }
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                color: #2c3e50;
                font-family: 'Consolas', monospace;
                font-size: 12px;
            }
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
                color: #2c3e50;
            }
        """)
        
        # Training state
        self.training_active = False
        self.training_thread = None
        self.signals = TrainingSignals()
        
        # Results monitoring
        self.last_check_time = 0
        self.results_timer = QTimer()
        self.results_timer.timeout.connect(self.check_for_results_update)
        self.results_timer.start(5000)  # Check every 5 seconds
        
        # Selected model path for continual training
        self.selected_model_path = None
        
        self.init_ui()
        self.connect_signals()
        self.check_gpu_status()
        
        # Load project-specific settings if available
        if self.project_manager:
            self.load_project_settings()

    def init_ui(self):
        """Initialize the compact user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(main_splitter)
        
        # Settings panel (left side) - more compact
        settings_panel = self.create_compact_settings_panel()
        main_splitter.addWidget(settings_panel)
        
        # Dashboard panel (right side)
        self.tabs, self.figure, self.canvas, self.log_text = create_dashboard_tabs(self)
        main_splitter.addWidget(self.tabs)
        
        # Set initial splitter sizes - more space for dashboard
        main_splitter.setSizes([380, 1020])
        
        # Initialize model options AFTER UI is fully created
        self.update_model_options()

    def create_compact_settings_panel(self):
        """Create compact settings panel without excessive frames."""
        panel = QWidget()
        panel.setFixedWidth(380)
        panel.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-right: 1px solid #bdc3c7;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Compact header
        header_label = QLabel("Training Configuration")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setWeight(QFont.Weight.Bold)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("color: #2c3e50; padding: 8px; background-color: #ffffff; border-radius: 4px;")
        layout.addWidget(header_label)
        
        # Basic Settings Group - compact
        basic_group = QGroupBox("Basic Settings")
        basic_layout = QFormLayout(basic_group)
        basic_layout.setContentsMargins(8, 12, 8, 8)
        basic_layout.setVerticalSpacing(6)
        basic_layout.setHorizontalSpacing(8)
        
        # Project directory
        project_layout = QHBoxLayout()
        project_layout.setSpacing(4)
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("Training will be saved here...")
        project_browse = QPushButton("...")
        project_browse.setFixedWidth(30)
        project_browse.clicked.connect(self.browse_project)
        project_layout.addWidget(self.project_input)
        project_layout.addWidget(project_browse)
        basic_layout.addRow("Project Dir:", project_layout)
        
        # Experiment name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., experiment_001")
        basic_layout.addRow("Experiment:", self.name_input)
        
        # Data YAML file
        data_layout = QHBoxLayout()
        data_layout.setSpacing(4)
        self.data_input = QLineEdit()
        self.data_input.setPlaceholderText("Path to data.yaml...")
        data_browse = QPushButton("...")
        data_browse.setFixedWidth(30)
        data_browse.clicked.connect(self.browse_data)
        data_layout.addWidget(self.data_input)
        data_layout.addWidget(data_browse)
        basic_layout.addRow("Dataset YAML:", data_layout)
        
        # Model Type
        self.model_type_input = QComboBox()
        self.model_type_input.addItems(["Detection", "Segmentation", "Nachtraining"])
        self.model_type_input.currentTextChanged.connect(self.update_model_options)
        basic_layout.addRow("Model Type:", self.model_type_input)
        
        # Model Selection Container
        self.model_container = QWidget()
        self.model_container_layout = QVBoxLayout(self.model_container)
        self.model_container_layout.setContentsMargins(0, 0, 0, 0)
        self.model_container_layout.setSpacing(0)
        
        # Model dropdown (default)
        self.model_dropdown_widget = QWidget()
        dropdown_layout = QHBoxLayout(self.model_dropdown_widget)
        dropdown_layout.setContentsMargins(0, 0, 0, 0)
        dropdown_layout.setSpacing(0)
        
        self.model_input = QComboBox()
        self.model_input.setMinimumWidth(250)
        dropdown_layout.addWidget(self.model_input)
        
        # Model file browser (for continual training)
        self.model_browse_widget = QWidget()
        browse_layout = QHBoxLayout(self.model_browse_widget)
        browse_layout.setContentsMargins(0, 0, 0, 0)
        browse_layout.setSpacing(4)
        
        self.model_path_display = QLineEdit()
        self.model_path_display.setPlaceholderText("No model selected...")
        self.model_path_display.setReadOnly(True)
        self.model_browse_button = QPushButton("Browse")
        self.model_browse_button.setFixedWidth(60)
        self.model_browse_button.clicked.connect(self.browse_model_file)
        
        browse_layout.addWidget(self.model_path_display)
        browse_layout.addWidget(self.model_browse_button)
        
        # Add both widgets to container
        self.model_container_layout.addWidget(self.model_dropdown_widget)
        self.model_container_layout.addWidget(self.model_browse_widget)
        
        # Initially hide browse widget
        self.model_browse_widget.hide()
        
        basic_layout.addRow("Model:", self.model_container)
        
        layout.addWidget(basic_group)
        
        # Training Parameters Group - compact
        training_group = QGroupBox("Training Parameters")
        training_layout = QFormLayout(training_group)
        training_layout.setContentsMargins(8, 12, 8, 8)
        training_layout.setVerticalSpacing(6)
        training_layout.setHorizontalSpacing(6)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.setSpacing(4)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        self.epochs_input.setFixedWidth(80)
        epochs_info = ParameterInfoButton(
            "Number of training iterations through the complete dataset.\n"
            "More epochs = longer training but potentially better results.\n"
            "Typical: 100-300 for new models, 10-50 for fine-tuning."
        )
        epochs_layout.addWidget(self.epochs_input)
        epochs_layout.addWidget(epochs_info)
        epochs_layout.addStretch()
        training_layout.addRow("Epochs:", epochs_layout)
        
        # Image Size
        imgsz_layout = QHBoxLayout()
        imgsz_layout.setSpacing(4)
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(320, 1280)
        self.imgsz_input.setValue(640)
        self.imgsz_input.setSingleStep(32)
        self.imgsz_input.setFixedWidth(80)
        imgsz_info = ParameterInfoButton(
            "Input image size (must be multiple of 32).\n"
            "Higher = better detail but slower training.\n"
            "Common: 640 (standard), 832 (detailed), 1024 (high detail)."
        )
        imgsz_layout.addWidget(self.imgsz_input)
        imgsz_layout.addWidget(imgsz_info)
        imgsz_layout.addStretch()
        training_layout.addRow("Image Size:", imgsz_layout)
        
        # Batch Size
        batch_layout = QHBoxLayout()
        batch_layout.setSpacing(4)
        self.batch_input = QDoubleSpinBox()
        self.batch_input.setRange(0.1, 128.0)
        self.batch_input.setValue(0.8)
        self.batch_input.setSingleStep(0.1)
        self.batch_input.setFixedWidth(80)
        batch_info = ParameterInfoButton(
            "Batch size affects training speed and memory usage.\n"
            "Auto-scaling: Values < 1.0 are percentage of available memory.\n"
            "Manual: Integer values for fixed batch size."
        )
        batch_layout.addWidget(self.batch_input)
        batch_layout.addWidget(batch_info)
        batch_layout.addStretch()
        training_layout.addRow("Batch:", batch_layout)
        
        # Learning Rate
        lr_layout = QHBoxLayout()
        lr_layout.setSpacing(4)
        self.lr0_input = QDoubleSpinBox()
        self.lr0_input.setRange(0.0001, 0.1)
        self.lr0_input.setValue(0.005)
        self.lr0_input.setDecimals(4)
        self.lr0_input.setSingleStep(0.001)
        self.lr0_input.setFixedWidth(80)
        lr_info = ParameterInfoButton(
            "Initial learning rate for optimizer.\n"
            "Higher = faster learning but risk of instability.\n"
            "Typical: 0.001-0.01 for AdamW optimizer."
        )
        lr_layout.addWidget(self.lr0_input)
        lr_layout.addWidget(lr_info)
        lr_layout.addStretch()
        training_layout.addRow("Learning Rate:", lr_layout)
        
        layout.addWidget(training_group)
        
        # Advanced Parameters Group - compact
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_layout = QFormLayout(advanced_group)
        advanced_layout.setContentsMargins(8, 12, 8, 8)
        advanced_layout.setVerticalSpacing(6)
        advanced_layout.setHorizontalSpacing(6)
        
        # Close Mosaic
        close_mosaic_layout = QHBoxLayout()
        close_mosaic_layout.setSpacing(4)
        self.close_mosaic_input = QSpinBox()
        self.close_mosaic_input.setRange(0, 50)
        self.close_mosaic_input.setValue(10)
        self.close_mosaic_input.setFixedWidth(60)
        close_mosaic_info = ParameterInfoButton(
            "Epoch to disable mosaic augmentation (recommended: 10).\n"
            "Mosaic augmentation is disabled for the last N epochs.\n"
            "This allows the model to learn on normal, non-augmented images."
        )
        close_mosaic_layout.addWidget(self.close_mosaic_input)
        close_mosaic_layout.addWidget(close_mosaic_info)
        close_mosaic_layout.addStretch()
        advanced_layout.addRow("Close Mosaic:", close_mosaic_layout)
        
        # Momentum
        momentum_layout = QHBoxLayout()
        momentum_layout.setSpacing(4)
        self.momentum_input = QDoubleSpinBox()
        self.momentum_input.setRange(0.0, 1.0)
        self.momentum_input.setValue(0.937)
        self.momentum_input.setDecimals(3)
        self.momentum_input.setSingleStep(0.01)
        self.momentum_input.setFixedWidth(80)
        momentum_info = ParameterInfoButton(
            "Momentum factor for SGD-based optimizers.\n"
            "Higher values provide more stability but slower convergence.\n"
            "Typical range: 0.9-0.99"
        )
        momentum_layout.addWidget(self.momentum_input)
        momentum_layout.addWidget(momentum_info)
        momentum_layout.addStretch()
        advanced_layout.addRow("Momentum:", momentum_layout)
        
        # Warmup Epochs
        warmup_epochs_layout = QHBoxLayout()
        warmup_epochs_layout.setSpacing(4)
        self.warmup_epochs_input = QSpinBox()
        self.warmup_epochs_input.setRange(0, 10)
        self.warmup_epochs_input.setValue(3)
        self.warmup_epochs_input.setFixedWidth(60)
        warmup_epochs_info = ParameterInfoButton(
            "Number of warmup epochs at the beginning of training.\n"
            "Gradually increases learning rate from 0 to lr0.\n"
            "Helps stabilize training start."
        )
        warmup_epochs_layout.addWidget(self.warmup_epochs_input)
        warmup_epochs_layout.addWidget(warmup_epochs_info)
        warmup_epochs_layout.addStretch()
        advanced_layout.addRow("Warmup Epochs:", warmup_epochs_layout)
        
        # Warmup Momentum
        warmup_momentum_layout = QHBoxLayout()
        warmup_momentum_layout.setSpacing(4)
        self.warmup_momentum_input = QDoubleSpinBox()
        self.warmup_momentum_input.setRange(0.0, 1.0)
        self.warmup_momentum_input.setValue(0.8)
        self.warmup_momentum_input.setDecimals(3)
        self.warmup_momentum_input.setSingleStep(0.01)
        self.warmup_momentum_input.setFixedWidth(80)
        warmup_momentum_info = ParameterInfoButton(
            "Initial momentum during warmup phase.\n"
            "Usually lower than final momentum.\n"
            "Typical: 0.5-0.9"
        )
        warmup_momentum_layout.addWidget(self.warmup_momentum_input)
        warmup_momentum_layout.addWidget(warmup_momentum_info)
        warmup_momentum_layout.addStretch()
        advanced_layout.addRow("Warmup Momentum:", warmup_momentum_layout)
        
        # Box Loss Weight
        box_layout = QHBoxLayout()
        box_layout.setSpacing(4)
        self.box_input = QDoubleSpinBox()
        self.box_input.setRange(1.0, 20.0)
        self.box_input.setValue(7.5)
        self.box_input.setDecimals(1)
        self.box_input.setSingleStep(0.5)
        self.box_input.setFixedWidth(80)
        box_info = ParameterInfoButton(
            "Weight for box regression loss.\n"
            "Higher values emphasize accurate bounding box prediction.\n"
            "Typical range: 5.0-10.0"
        )
        box_layout.addWidget(self.box_input)
        box_layout.addWidget(box_info)
        box_layout.addStretch()
        advanced_layout.addRow("Box Loss Weight:", box_layout)
        
        # Dropout
        dropout_layout = QHBoxLayout()
        dropout_layout.setSpacing(4)
        self.dropout_input = QDoubleSpinBox()
        self.dropout_input.setRange(0.0, 0.8)
        self.dropout_input.setValue(0.0)
        self.dropout_input.setDecimals(2)
        self.dropout_input.setSingleStep(0.1)
        self.dropout_input.setFixedWidth(80)
        dropout_info = ParameterInfoButton(
            "Dropout probability for regularization.\n"
            "Helps prevent overfitting on small datasets.\n"
            "0.0 = no dropout, 0.1-0.3 for overfitting prevention."
        )
        dropout_layout.addWidget(self.dropout_input)
        dropout_layout.addWidget(dropout_info)
        dropout_layout.addStretch()
        advanced_layout.addRow("Dropout:", dropout_layout)
        
        # Copy Paste (for segmentation)
        copy_paste_layout = QHBoxLayout()
        copy_paste_layout.setSpacing(4)
        self.copy_paste_input = QDoubleSpinBox()
        self.copy_paste_input.setRange(0.0, 1.0)
        self.copy_paste_input.setValue(0.0)
        self.copy_paste_input.setDecimals(2)
        self.copy_paste_input.setSingleStep(0.1)
        self.copy_paste_input.setFixedWidth(80)
        copy_paste_info = ParameterInfoButton(
            "Copy-paste augmentation probability (segmentation only).\n"
            "Copies instances from one image to another.\n"
            "Typical: 0.0-0.3 for segmentation models."
        )
        copy_paste_layout.addWidget(self.copy_paste_input)
        copy_paste_layout.addWidget(copy_paste_info)
        copy_paste_layout.addStretch()
        advanced_layout.addRow("Copy Paste:", copy_paste_layout)
        
        # Mask Ratio (for segmentation)
        mask_ratio_layout = QHBoxLayout()
        mask_ratio_layout.setSpacing(4)
        self.mask_ratio_input = QSpinBox()
        self.mask_ratio_input.setRange(1, 8)
        self.mask_ratio_input.setValue(4)
        self.mask_ratio_input.setFixedWidth(60)
        mask_ratio_info = ParameterInfoButton(
            "Mask downsampling ratio for segmentation.\n"
            "Higher values = faster training, lower accuracy.\n"
            "Typical: 4 (good balance), 1 (best quality), 8 (fastest)."
        )
        mask_ratio_layout.addWidget(self.mask_ratio_input)
        mask_ratio_layout.addWidget(mask_ratio_info)
        mask_ratio_layout.addStretch()
        advanced_layout.addRow("Mask Ratio:", mask_ratio_layout)
        
        layout.addWidget(advanced_group)
        
        # Training Options Group - compact checkboxes
        options_group = QGroupBox("Training Options")
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(8, 12, 8, 8)
        options_layout.setSpacing(4)
        
        self.resume_input = QCheckBox("Resume from last checkpoint")
        self.multi_scale_input = QCheckBox("Multi-scale training")
        self.cos_lr_input = QCheckBox("Cosine learning rate scheduler")
        self.cos_lr_input.setChecked(True)
        
        options_layout.addWidget(self.resume_input)
        options_layout.addWidget(self.multi_scale_input)
        options_layout.addWidget(self.cos_lr_input)
        
        layout.addWidget(options_group)
        
        # Control section - compact
        control_layout = QVBoxLayout()
        control_layout.setSpacing(6)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        
        self.start_button = QPushButton("Start Training")
        self.start_button.setMinimumHeight(35)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.start_button.clicked.connect(self.start_training)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setFixedWidth(60)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.stop_button.clicked.connect(self.stop_training)
        
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        control_layout.addLayout(buttons_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(20)
        control_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to start training")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #2c3e50; font-weight: bold; font-size: 12px;")
        control_layout.addWidget(self.status_label)
        
        layout.addLayout(control_layout)
        
        # GPU Status - very compact
        self.gpu_status_label = QLabel("Checking GPU...")
        self.gpu_status_label.setWordWrap(True)
        self.gpu_status_label.setStyleSheet("""
            color: #7f8c8d; 
            font-size: 11px; 
            padding: 6px; 
            background-color: #ffffff; 
            border-radius: 4px;
        """)
        self.gpu_status_label.setMaximumHeight(60)
        layout.addWidget(self.gpu_status_label)
        
        # Navigation button - compact
        self.verification_button = QPushButton("Continue to Verification")
        self.verification_button.setMinimumHeight(30)
        self.verification_button.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7d3c98;
            }
        """)
        self.verification_button.clicked.connect(self.open_verification_app)
        layout.addWidget(self.verification_button)
        
        layout.addStretch()
        return panel

    def update_model_options(self):
        """Update available model options based on selected type."""
        model_type = self.model_type_input.currentText().lower()
        
        if model_type == "nachtraining":
            # Hide dropdown, show browse interface
            self.model_dropdown_widget.hide()
            self.model_browse_widget.show()
            
            # Try to load current model from project
            if self.project_manager:
                try:
                    latest_model = self.project_manager.get_latest_model_path()
                    if latest_model and latest_model.exists():
                        self.model_path_display.setText(str(latest_model))
                        self.selected_model_path = str(latest_model)
                    else:
                        self.model_path_display.setText("No model found - please select one")
                        self.selected_model_path = None
                except Exception as e:
                    logger.warning(f"Error loading latest model: {e}")
                    self.model_path_display.setText("Please select a model file")
                    self.selected_model_path = None
            else:
                self.model_path_display.setText("Please select a model file")
                self.selected_model_path = None
        else:
            # Show dropdown, hide browse interface
            self.model_dropdown_widget.show()
            self.model_browse_widget.hide()
            
            # Clear and populate dropdown
            self.model_input.clear()
            
            if model_type == "segmentation":
                models = [
                    "yolo11n-seg.pt",
                    "yolo11s-seg.pt", 
                    "yolo11m-seg.pt",
                    "yolo11l-seg.pt",
                    "yolo11x-seg.pt",
                    "yolo8n-seg.pt",
                    "yolo8s-seg.pt",
                    "yolo8m-seg.pt",
                    "yolo8l-seg.pt",
                    "yolo8x-seg.pt"
                ]
            else:  # detection
                models = [
                    "yolo11n.pt",
                    "yolo11s.pt",
                    "yolo11m.pt", 
                    "yolo11l.pt",
                    "yolo11x.pt",
                    "yolo8n.pt",
                    "yolo8s.pt",
                    "yolo8m.pt",
                    "yolo8l.pt",
                    "yolo8x.pt"
                ]
            
            self.model_input.addItems(models)
            
            # Set default based on project manager if available
            if self.project_manager:
                try:
                    default_model = self.project_manager.get_default_model_path(model_type)
                    index = self.model_input.findText(default_model)
                    if index >= 0:
                        self.model_input.setCurrentIndex(index)
                except Exception:
                    pass

    def browse_project(self):
        """Browse for project directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.project_input.setText(directory)

    def browse_data(self):
        """Browse for dataset YAML file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset YAML", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.data_input.setText(file_path)

    def browse_model_file(self):
        """Browse for pre-trained model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-trained Model", "", "PyTorch Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.selected_model_path = file_path
            self.model_path_display.setText(file_path)

    def check_gpu_status(self):
        """Check and display GPU status."""
        try:
            gpu_available, gpu_message = check_gpu()
            if gpu_available:
                # Compact GPU status for small screens
                lines = gpu_message.split('\n')
                compact_msg = lines[0] if lines else "GPU Ready"
                self.gpu_status_label.setText(f"✅ {compact_msg}")
                self.gpu_status_label.setStyleSheet("""
                    color: #27ae60; 
                    font-size: 11px; 
                    padding: 6px; 
                    background-color: #ffffff; 
                    border-radius: 4px;
                """)
            else:
                self.gpu_status_label.setText(f"⚠️ CPU Mode")
                self.gpu_status_label.setStyleSheet("""
                    color: #f39c12; 
                    font-size: 11px; 
                    padding: 6px; 
                    background-color: #ffffff; 
                    border-radius: 4px;
                """)
        except Exception as e:
            self.gpu_status_label.setText(f"❌ GPU Check Failed")
            self.gpu_status_label.setStyleSheet("""
                color: #e74c3c; 
                font-size: 11px; 
                padding: 6px; 
                background-color: #ffffff; 
                border-radius: 4px;
            """)

    def connect_signals(self):
        """Connect training thread signals."""
        self.signals.progress_updated.connect(self.update_progress)
        self.signals.log_updated.connect(self.update_log)
        self.signals.results_updated.connect(self.update_dashboard)

    def start_training(self):
        """Start the training process."""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Get parameters
        data_path = self.data_input.text()
        epochs = self.epochs_input.value()
        imgsz = self.imgsz_input.value()
        batch = self.batch_input.value()
        lr0 = self.lr0_input.value()
        project = self.project_input.text()
        experiment = self.name_input.text()
        close_mosaic = self.close_mosaic_input.value()
        momentum = self.momentum_input.value()
        warmup_epochs = self.warmup_epochs_input.value()
        warmup_momentum = self.warmup_momentum_input.value()
        box = self.box_input.value()
        dropout = self.dropout_input.value()
        copy_paste = self.copy_paste_input.value()
        mask_ratio = self.mask_ratio_input.value()
        
        # Get model path
        model_type = self.model_type_input.currentText().lower()
        if model_type == "nachtraining":
            if not self.selected_model_path:
                QMessageBox.warning(self, "Error", "Please select a pre-trained model for continual training.")
                return
            model_path = self.selected_model_path
        else:
            model_path = self.model_input.currentText()
        
        # Disable UI during training
        self.training_active = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting training...")
        
        # Clear log
        self.log_text.setText("Training started...\n")
        
        # Save settings to project
        if self.project_manager:
            self.save_training_settings()
        
        # Start training in thread
        self.training_thread = start_training_thread(
            self.signals,
            data_path=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            resume=self.resume_input.isChecked(),
            multi_scale=self.multi_scale_input.isChecked(),
            cos_lr=self.cos_lr_input.isChecked(),
            close_mosaic=close_mosaic,
            momentum=momentum,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            box=box,
            dropout=dropout,
            copy_paste=copy_paste,
            mask_ratio=mask_ratio,
            project=project,
            experiment=experiment,
            model_path=model_path,
            model_type=model_type
        )

    def stop_training(self):
        """Stop the training process."""
        if self.training_active:
            stop_training()
            self.training_active = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Training stopped by user")

    def validate_inputs(self):
        """Validate training inputs."""
        if not self.data_input.text():
            QMessageBox.warning(self, "Error", "Please select a dataset YAML file.")
            return False
        
        if not self.project_input.text():
            QMessageBox.warning(self, "Error", "Please select a project directory.")
            return False
        
        if not self.name_input.text():
            QMessageBox.warning(self, "Error", "Please enter an experiment name.")
            return False
        
        # Validate YAML file
        yaml_path = self.data_input.text()
        if not os.path.exists(yaml_path):
            QMessageBox.warning(self, "Error", f"Dataset YAML file not found: {yaml_path}")
            return False
        
        # Check dataset format compatibility with selected model type
        model_type = self.model_type_input.currentText().lower()
        dataset_format = self.detect_dataset_format(yaml_path)
        
        # Validate dataset format for segmentation
        if model_type == "segmentation":
            if dataset_format == "bbox_only":
                reply = QMessageBox.critical(
                    self, "Dataset Format Error",
                    f"❌ Segmentation training selected but dataset contains only bounding boxes!\n\n"
                    f"Segmentation models require polygon annotations, not just bounding boxes.\n\n"
                    f"Solutions:\n"
                    f"1. Switch to 'Detection' mode (recommended for bounding box data)\n"
                    f"2. Re-annotate your data with polygon annotations\n\n"
                    f"Would you like me to automatically switch to Detection mode?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.model_type_input.setCurrentText("Detection")
                    self.update_model_options()
                    QMessageBox.information(self, "Mode Switched", "✅ Switched to Detection mode. Training will now work correctly.")
                    return True  # Continue with detection mode
                else:
                    return False  # User wants to fix dataset first
            elif dataset_format == "polygon":
                QMessageBox.information(self, "Format OK", "✅ Polygon dataset detected - perfect for segmentation training!")
        
        elif model_type == "detection" and dataset_format == "polygon":
            reply = QMessageBox.question(
                self, "Dataset Format Notice",
                f"ℹ️ Detection training selected but dataset contains polygon annotations.\n\n"
                f"Detection models can use polygon data but will only use bounding boxes.\n"
                f"Consider using 'Segmentation' mode for better polygon utilization.\n\n"
                f"Continue with Detection mode anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return False
        
        model_type = self.model_type_input.currentText().lower()
        is_valid, message = validate_yaml_for_model_type(yaml_path, model_type)
        if not is_valid:
            QMessageBox.critical(self, "Dataset Error", message)
            return False
        elif "Warning:" in message:
            reply = QMessageBox.question(
                self, "Dataset Warning", 
                f"{message}\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return False
        
        return True
    
    def detect_dataset_format(self, yaml_path):
        """Detect if dataset contains bounding boxes or polygons."""
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Get path to train labels
            train_path = data.get('train', '')
            if train_path:
                # Find labels directory
                train_labels_dir = Path(yaml_path).parent / "train" / "labels"
                if train_labels_dir.exists():
                    # Check first few label files
                    label_files = list(train_labels_dir.glob("*.txt"))[:5]
                    
                    has_bbox = False
                    has_polygon = False
                    
                    for label_file in label_files:
                        try:
                            with open(label_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) == 5:  # class x y w h = bounding box
                                        has_bbox = True
                                    elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:  # polygon
                                        has_polygon = True
                                    
                                    if has_bbox and has_polygon:
                                        return "mixed"
                        except:
                            continue
                    
                    if has_polygon:
                        return "polygon"
                    elif has_bbox:
                        return "bbox_only"
            
            return "unknown"
        except Exception as e:
            logger.warning(f"Could not detect dataset format: {e}")
            return "unknown"

    def update_progress(self, progress, message):
        """Update training progress."""
        self.progress_bar.setValue(progress)
        if message:
            self.status_label.setText(message)
        
        if progress >= 100 or progress == 0:
            self.training_active = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            if progress >= 100 and self.project_manager:
                self.project_manager.mark_step_completed(WorkflowStep.TRAINING)

    def update_log(self, message):
        """Update training log."""
        current_text = self.log_text.toPlainText()
        self.log_text.setPlainText(current_text + message + "\n")
        
        # Auto-scroll to bottom for QTextEdit
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_dashboard(self, df):
        """Update dashboard with new training data."""
        try:
            update_dashboard_plots(self, df)
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")

    def check_for_results_update(self):
        """Check for updates to results.csv file."""
        if not self.project_input.text() or not self.name_input.text():
            return
        
        project = self.project_input.text()
        experiment = self.name_input.text()
        
        df = check_and_load_results_csv(project, experiment, self.last_check_time)
        if df is not None:
            self.last_check_time = os.path.getmtime(df.filepath)
            self.update_dashboard(df)

    def load_project_settings(self):
        """Load saved training settings from project."""
        try:
            saved_settings = self.project_manager.get_training_settings()
            if saved_settings:
                # Apply saved settings to UI
                for key, value in saved_settings.items():
                    widget_name = f"{key}_input"
                    if hasattr(self, widget_name):
                        widget = getattr(self, widget_name)
                        try:
                            if hasattr(widget, 'setValue'):
                                widget.setValue(value)
                            elif hasattr(widget, 'setText'):
                                widget.setText(str(value))
                            elif hasattr(widget, 'setChecked'):
                                widget.setChecked(bool(value))
                        except Exception as e:
                            logger.warning(f"Could not set {key} to {value}: {e}")
        except Exception as e:
            logger.warning(f"Could not load project settings: {e}")

    def save_training_settings(self):
        """Save current training settings to project."""
        try:
            settings = {
                'epochs': self.epochs_input.value(),
                'imgsz': self.imgsz_input.value(),
                'batch': self.batch_input.value(),
                'lr0': self.lr0_input.value(),
                'close_mosaic': self.close_mosaic_input.value(),
                'momentum': self.momentum_input.value(),
                'warmup_epochs': self.warmup_epochs_input.value(),
                'warmup_momentum': self.warmup_momentum_input.value(),
                'box': self.box_input.value(),
                'dropout': self.dropout_input.value(),
                'copy_paste': self.copy_paste_input.value(),
                'mask_ratio': self.mask_ratio_input.value(),
                'resume': self.resume_input.isChecked(),
                'multi_scale': self.multi_scale_input.isChecked(),
                'cos_lr': self.cos_lr_input.isChecked(),
            }
            self.project_manager.update_training_settings(settings)
        except Exception as e:
            logger.warning(f"Could not save training settings: {e}")

    def load_last_training_results(self):
        """Load results from the last training experiment."""
        try:
            if not self.project_manager:
                return
            
            last_exp = self.project_manager.get_last_experiment_name()
            project_dir = self.project_manager.get_models_dir()
            
            # Look for results.csv in the last experiment
            df = check_and_load_results_csv(str(project_dir), last_exp)
            if df is not None:
                self.update_dashboard(df)
                self.last_check_time = os.path.getmtime(df.filepath)
        except Exception as e:
            logger.warning(f"Could not load last training results: {e}")

    def open_verification_app(self):
        """Open verification app and close training window."""
        try:
            from gui.verification_app import LiveAnnotationApp
            app = LiveAnnotationApp()
            app.project_manager = self.project_manager
            
            if self.project_manager:
                # Set model path to latest trained model
                model_path = self.project_manager.get_latest_model_path()
                if model_path and model_path.exists():
                    app.model_line_edit.setText(str(model_path))
                
                # Set test directory
                test_dir = self.project_manager.get_split_dir() / "test" / "images"
                if not test_dir.exists() or not list(test_dir.glob("*.jpg")):
                    test_dir = self.project_manager.get_labeled_dir()
                
                app.folder_line_edit.setText(str(test_dir))
            
            app.show()
            self.close()
        except Exception as e:
            logger.error(f"Failed to open verification app: {e}")
            QMessageBox.critical(self, "Error", f"Could not open verification app:\n{str(e)}")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.training_active:
            reply = QMessageBox.question(
                self, "Training Active",
                "Training is currently running. Stop training and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_training()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()