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
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
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
        
        # Initialize model options AFTER UI is fully created
        self.update_model_options()
        
        # Load project-specific settings if available
        if self.project_manager:
            self.load_project_settings()

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(main_splitter)
        
        # Settings panel (left side)
        settings_panel = self.create_settings_panel()
        main_splitter.addWidget(settings_panel)
        
        # Dashboard panel (right side)
        self.tabs, self.figure, self.canvas, self.log_text = create_dashboard_tabs(self)
        main_splitter.addWidget(self.tabs)
        
        # Set initial splitter sizes
        main_splitter.setSizes([400, 1000])

    def create_settings_panel(self):
        """Create the settings configuration panel."""
        panel = QWidget()
        panel.setFixedWidth(400)
        panel.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-right: 1px solid #dee2e6;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #495057;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #dee2e6;
            }
            QLabel {
                color: #495057;
                padding: 2px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: white;
                border: 2px solid #ced4da;
                border-radius: 4px;
                padding: 6px;
                color: #495057;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #007bff;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Basic Settings Group
        basic_group = QGroupBox("Basic Settings")
        basic_layout = QFormLayout()
        
        # Project directory
        project_layout = QHBoxLayout()
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("Training will be saved here...")
        project_browse = QPushButton("Browse")
        project_browse.clicked.connect(self.browse_project)
        project_layout.addWidget(self.project_input)
        project_layout.addWidget(project_browse)
        basic_layout.addRow("Project Directory:", project_layout)
        
        # Experiment name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., experiment_001")
        basic_layout.addRow("Experiment Name:", self.name_input)
        
        # Data YAML file
        data_layout = QHBoxLayout()
        self.data_input = QLineEdit()
        self.data_input.setPlaceholderText("Path to data.yaml file...")
        data_browse = QPushButton("Browse")
        data_browse.clicked.connect(self.browse_data)
        data_layout.addWidget(self.data_input)
        data_layout.addWidget(data_browse)
        basic_layout.addRow("Dataset YAML:", data_layout)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # Model Settings Group
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout()
        
        # Model Type
        self.model_type_input = QComboBox()
        self.model_type_input.addItems(["Detection", "Segmentation", "Nachtraining"])
        self.model_type_input.currentTextChanged.connect(self.update_model_options)
        model_layout.addRow("Model Type:", self.model_type_input)
        
        # Model Selection Container
        self.model_container = QWidget()
        self.model_container_layout = QVBoxLayout(self.model_container)
        self.model_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Model dropdown (default)
        self.model_dropdown_widget = QWidget()
        dropdown_layout = QHBoxLayout(self.model_dropdown_widget)
        dropdown_layout.setContentsMargins(0, 0, 0, 0)
        
        self.model_input = QComboBox()
        self.model_input.setMinimumWidth(200)
        dropdown_layout.addWidget(self.model_input)
        dropdown_layout.addStretch()
        
        # Model file browser (for continual training)
        self.model_browse_widget = QWidget()
        browse_layout = QHBoxLayout(self.model_browse_widget)
        browse_layout.setContentsMargins(0, 0, 0, 0)
        
        self.model_path_display = QLineEdit()
        self.model_path_display.setPlaceholderText("No model selected...")
        self.model_path_display.setReadOnly(True)
        self.model_browse_button = QPushButton("Browse Model...")
        self.model_browse_button.clicked.connect(self.browse_model_file)
        
        browse_layout.addWidget(self.model_path_display)
        browse_layout.addWidget(self.model_browse_button)
        
        # Add both widgets to container
        self.model_container_layout.addWidget(self.model_dropdown_widget)
        self.model_container_layout.addWidget(self.model_browse_widget)
        
        # Initially hide browse widget
        self.model_browse_widget.hide()
        
        model_layout.addRow("Model:", self.model_container)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training Parameters Group
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        # Epochs
        epochs_layout = QHBoxLayout()
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        epochs_info = ParameterInfoButton(
            "Number of complete passes through the training dataset.\n"
            "More epochs = longer training but potentially better results.\n"
            "Typical values: 100-300 for new models, 10-50 for fine-tuning."
        )
        epochs_layout.addWidget(self.epochs_input)
        epochs_layout.addWidget(epochs_info)
        epochs_layout.addStretch()
        params_layout.addRow("Epochs:", epochs_layout)
        
        # Image Size
        imgsz_layout = QHBoxLayout()
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(320, 1280)
        self.imgsz_input.setValue(640)
        self.imgsz_input.setSingleStep(32)
        imgsz_info = ParameterInfoButton(
            "Input image size for training (must be multiple of 32).\n"
            "Higher values = better detail detection but slower training.\n"
            "Common values: 640 (standard), 832 (detailed), 1024 (high detail)."
        )
        imgsz_layout.addWidget(self.imgsz_input)
        imgsz_layout.addWidget(imgsz_info)
        imgsz_layout.addStretch()
        params_layout.addRow("Image Size:", imgsz_layout)
        
        # Batch Size
        batch_layout = QHBoxLayout()
        self.batch_input = QDoubleSpinBox()
        self.batch_input.setRange(0.1, 128.0)
        self.batch_input.setValue(0.8)
        self.batch_input.setSingleStep(0.1)
        batch_info = ParameterInfoButton(
            "Batch size controls memory usage and training stability.\n"
            "Values < 1.0 = automatic sizing based on GPU memory.\n"
            "Values ≥ 1.0 = fixed batch size.\n"
            "Start with 0.8 for automatic, or 16 for fixed batch."
        )
        batch_layout.addWidget(self.batch_input)
        batch_layout.addWidget(batch_info)
        batch_layout.addStretch()
        params_layout.addRow("Batch Size:", batch_layout)
        
        # Learning Rate
        lr_layout = QHBoxLayout()
        self.lr0_input = QDoubleSpinBox()
        self.lr0_input.setRange(0.0001, 0.1)
        self.lr0_input.setValue(0.005)
        self.lr0_input.setDecimals(4)
        self.lr0_input.setSingleStep(0.001)
        lr_info = ParameterInfoButton(
            "Initial learning rate for optimization.\n"
            "Higher values = faster learning but less stable.\n"
            "Typical values: 0.001-0.01 for new training, 0.0001-0.001 for fine-tuning."
        )
        lr_layout.addWidget(self.lr0_input)
        lr_layout.addWidget(lr_info)
        lr_layout.addStretch()
        params_layout.addRow("Learning Rate:", lr_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Advanced Settings Group
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout()
        
        # Resume training
        self.resume_input = QCheckBox("Resume from last checkpoint")
        advanced_layout.addRow("Resume:", self.resume_input)
        
        # Multi-scale training
        self.multi_scale_input = QCheckBox("Multi-scale training")
        advanced_layout.addRow("Multi-scale:", self.multi_scale_input)
        
        # Cosine LR
        self.cos_lr_input = QCheckBox("Cosine learning rate scheduler")
        self.cos_lr_input.setChecked(True)
        advanced_layout.addRow("Cosine LR:", self.cos_lr_input)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Control Buttons
        control_group = QGroupBox("Training Control")
        control_layout = QVBoxLayout()
        
        # Start button
        self.start_button = QPushButton("Start Training")
        self.start_button.setMinimumHeight(50)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.start_button.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.stop_button.clicked.connect(self.stop_training)
        control_layout.addWidget(self.stop_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        control_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to start training")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #495057; font-weight: bold;")
        control_layout.addWidget(self.status_label)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # GPU Status
        self.gpu_status_label = QLabel("Checking GPU...")
        self.gpu_status_label.setWordWrap(True)
        self.gpu_status_label.setStyleSheet("color: #6c757d; font-size: 12px; padding: 10px;")
        layout.addWidget(self.gpu_status_label)
        
        # Navigation buttons
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()
        
        self.verification_button = QPushButton("Continue to Verification")
        self.verification_button.setMinimumHeight(40)
        self.verification_button.clicked.connect(self.open_verification_app)
        nav_layout.addWidget(self.verification_button)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        layout.addStretch()
        return panel

    def update_model_options(self):
        """Update available model options based on selected type."""
        model_type = self.model_type_input.currentText().lower()
        
        print(f"DEBUG: Updating model options for type: {model_type}")
        
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
                    print(f"Error loading latest model: {e}")
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
            else:  # detection (default)
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
            
            print(f"DEBUG: Adding {len(models)} models to dropdown")
            self.model_input.addItems(models)
            
            # Set default based on project manager if available
            if self.project_manager:
                try:
                    default_model = self.project_manager.get_default_model_path(model_type)
                    index = self.model_input.findText(default_model)
                    if index >= 0:
                        self.model_input.setCurrentIndex(index)
                        print(f"DEBUG: Set default model to {default_model}")
                except:
                    pass
            
            print(f"DEBUG: Model dropdown now has {self.model_input.count()} items")

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
                self.gpu_status_label.setText(f"✅ GPU Ready: {gpu_message}")
                self.gpu_status_label.setStyleSheet("color: #28a745; font-size: 12px; padding: 10px;")
            else:
                self.gpu_status_label.setText(f"⚠️ GPU Not Available: {gpu_message}")
                self.gpu_status_label.setStyleSheet("color: #ffc107; font-size: 12px; padding: 10px;")
        except Exception as e:
            self.gpu_status_label.setText(f"❌ GPU Check Failed: {str(e)}")
            self.gpu_status_label.setStyleSheet("color: #dc3545; font-size: 12px; padding: 10px;")

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
            close_mosaic=10,
            momentum=0.937,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            dropout=0.0,
            copy_paste=0.0,
            mask_ratio=4,
            project=project,
            experiment=experiment,
            model_path=model_path
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
        current_text = self.log_text.text()
        self.log_text.setText(current_text + message + "\n")
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.parent().verticalScrollBar()
        if scrollbar:
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