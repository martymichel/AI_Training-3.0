"""Main window for YOLO training with integrated dashboard."""

import os
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QFrame,
    QProgressBar,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QScrollArea,
    QGridLayout,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from config import Config
from utils.validation import validate_yaml, validate_yaml_for_model_type, check_gpu
from gui.training.dashboard_view import create_dashboard_tabs
from gui.training.training_thread import (
    TrainingSignals,
    start_training_thread,
    stop_training,
)
from gui.training.parameter_info import ParameterInfoButton
from project_manager import ProjectManager, WorkflowStep

# Configure logging
logger = logging.getLogger("training_gui")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)


class TrainSettingsWindow(QMainWindow):
    """Main window for YOLO training with integrated dashboard."""

    def __init__(self, project_manager=None):
        super().__init__()
        self.training_active = False
        self.project_manager = project_manager
        self.setWindowTitle("YOLO Training Advanced")
        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Initialize from config
        self.project = Config.training.project_dir
        self.experiment = Config.training.experiment_name

        # Initialize last checked time for results.csv
        self.last_results_check = 0
        self.results_check_timer = QTimer()
        self.results_check_timer.setInterval(2000)  # Check every 2 seconds
        self.results_check_timer.timeout.connect(self.check_results_csv)

        # Initialize UI elements to None (will be created in init_ui)
        self.model_type_input = None
        self.model_input = None
        self.multi_scale_label = None
        self.multi_scale_input = None
        self.multi_scale_info = None
        self.copy_paste_label = None
        self.copy_paste_input = None
        self.copy_paste_info = None
        self.mask_ratio_label = None
        self.mask_ratio_input = None
        self.mask_ratio_info = None

        # Initialize training signals
        self.signals = TrainingSignals()
        self.signals.progress_updated.connect(self.update_progress)
        self.signals.log_updated.connect(self.update_log)
        self.signals.results_updated.connect(self.update_dashboard)

        # Initialize UI
        self.init_ui()
        
        # Initialize model settings after UI is created
        self.initialize_model_settings()

    def init_ui(self):
        """Initialize the complete user interface."""
        # Create main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Left panel (settings)
        self.create_settings_panel()

        # Right panel (dashboard/log)
        self.create_dashboard_panel()

        # Add panels to splitter
        self.splitter.addWidget(self.settings_panel)
        self.splitter.addWidget(self.dashboard_panel)

        # Set initial sizes
        self.splitter.setSizes([400, 800])

    def create_settings_panel(self):
        """Create the settings panel with all controls."""
        self.settings_panel = QWidget()
        self.settings_panel.setStyleSheet("""
            QWidget {
                background-color: #f7f9fc;
                color: #2d3748;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #2d3748;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: bold;
                color: #2d3748;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #4a5568;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                border: 1px solid #cbd5e0;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
                color: #2d3748;
            }
            QCheckBox {
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #2d3748;
            }
        """)
        
        self.settings_layout = QVBoxLayout(self.settings_panel)

        # Create scroll area for settings
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f2f5;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #cbd5e0;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0aec0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        settings_content = QWidget()
        settings_content_layout = QVBoxLayout(settings_content)
        settings_content_layout.setSpacing(10)
        settings_scroll.setWidget(settings_content)
        
        # Add title to settings panel
        settings_title = QLabel("Training Configuration")
        settings_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        settings_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        settings_title.setMaximumHeight(30)
        settings_title.setStyleSheet("margin-bottom: 5px; color: #2d3748;")
        settings_content_layout.addWidget(settings_title)
        
        # Create project settings group
        self.create_project_settings_group(settings_content_layout)
        
        # Create basic training parameters group
        self.create_basic_training_group(settings_content_layout)
        
        # Create advanced parameters group
        self.create_advanced_training_group(settings_content_layout)
        
        # Create progress and control section
        self.create_progress_control_section(settings_content_layout)
        
        # Add scroll area to settings panel
        self.settings_layout.addWidget(settings_scroll)

    def create_project_settings_group(self, parent_layout):
        """Create project settings UI group."""
        group_frame = QGroupBox("Project Settings")
        group_layout = QGridLayout(group_frame)
        group_layout.setVerticalSpacing(6)
        
        row = 0
        # Project directory
        group_layout.addWidget(QLabel("Projekt-Verzeichnis:"), row, 0)
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("z.B. yolo_training_results")
        self.project_input.setText(Config.training.project_dir)
        group_layout.addWidget(self.project_input, row, 1)
        
        self.project_browse = QPushButton("...")
        self.project_browse.setMaximumWidth(40)
        self.project_browse.clicked.connect(self.browse_project)
        group_layout.addWidget(self.project_browse, row, 2)
        
        info_button = ParameterInfoButton(
            "Das Projekt-Verzeichnis bestimmt, wo die Trainingsergebnisse gespeichert werden."
        )
        group_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Experiment name
        group_layout.addWidget(QLabel("Experiment-Name:"), row, 0)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("z.B. experiment_v1")
        group_layout.addWidget(self.name_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Der Experiment-Name identifiziert das aktuelle Training."
        )
        group_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Data path
        group_layout.addWidget(QLabel("Datenpfad (YAML):"), row, 0)
        self.data_input = QLineEdit()
        group_layout.addWidget(self.data_input, row, 1)
        
        self.data_browse = QPushButton("...")
        self.data_browse.setMaximumWidth(40)
        self.data_browse.clicked.connect(self.browse_data)
        group_layout.addWidget(self.data_browse, row, 2)
        
        info_button = ParameterInfoButton(
            "YAML-Datei mit Datensatz-Konfiguration aus dem Dataset-Splitter."
        )
        group_layout.addWidget(info_button, row, 3)

        row += 1
        # Model Type Selection
        group_layout.addWidget(QLabel("Model-Typ:"), row, 0)
        self.model_type_input = QComboBox()
        self.model_type_input.addItems(["Detection", "Segmentation"])
        group_layout.addWidget(self.model_type_input, row, 1)

        info_button = ParameterInfoButton(
            "Detection: Für Bounding-Box Annotationen\nSegmentation: Für Polygon/Mask Annotationen"
        )
        group_layout.addWidget(info_button, row, 3)

        row += 1
        # Model Selection
        group_layout.addWidget(QLabel("Modell:"), row, 0)
        self.model_input = QComboBox()
        self.model_input.setEditable(True)
        group_layout.addWidget(self.model_input, row, 1)

        info_button = ParameterInfoButton(
            "Vorgefertigte YOLO-Modelle oder eigener Modellpfad."
        )
        group_layout.addWidget(info_button, row, 3)
        
        parent_layout.addWidget(group_frame)

    def create_basic_training_group(self, parent_layout):
        """Create basic training parameters group."""
        training_group = QGroupBox("Basic Training Parameters")
        training_layout = QGridLayout(training_group)
        training_layout.setVerticalSpacing(6)
        
        row = 0
        # Epochs
        training_layout.addWidget(QLabel("Anzahl Epochen:"), row, 0)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(5, 500)
        self.epochs_input.setValue(Config.training.epochs)
        training_layout.addWidget(self.epochs_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Anzahl der vollständigen Durchläufe durch den Trainingsdatensatz."
        )
        training_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Image size
        training_layout.addWidget(QLabel("Bildgröße:"), row, 0)
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(320, 1280)
        self.imgsz_input.setValue(Config.training.image_size)
        self.imgsz_input.setSingleStep(32)
        training_layout.addWidget(self.imgsz_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Die Größe, auf die alle Trainingsbilder skaliert werden (in Pixeln)."
        )
        training_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Batch
        training_layout.addWidget(QLabel("Batch:"), row, 0)
        self.batch_input = QDoubleSpinBox()
        self.batch_input.setRange(0.0, 1.0)
        self.batch_input.setDecimals(2)
        self.batch_input.setValue(Config.training.batch)
        self.batch_input.setSingleStep(0.05)
        training_layout.addWidget(self.batch_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Automatische Batch-Größe basierend auf verfügbarem GPU-Speicher."
        )
        training_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Learning rate
        training_layout.addWidget(QLabel("Lernrate (lr0):"), row, 0)
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.0001, 0.1)
        self.lr_input.setDecimals(4)
        self.lr_input.setValue(Config.training.lr0)
        self.lr_input.setSingleStep(0.0005)
        training_layout.addWidget(self.lr_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Die anfängliche Lernrate bestimmt die Schrittgröße beim Training."
        )
        training_layout.addWidget(info_button, row, 3)
        
        parent_layout.addWidget(training_group)

    def create_advanced_training_group(self, parent_layout):
        """Create advanced training parameters group."""
        advanced_frame = QGroupBox("Advanced Settings")
        self.advanced_layout = QGridLayout(advanced_frame)
        self.advanced_layout.setVerticalSpacing(6)
        
        row = 0
        # Resume training
        self.advanced_layout.addWidget(QLabel("Training fortsetzen:"), row, 0)
        self.resume_input = QCheckBox()
        self.resume_input.setStyleSheet("""
            QCheckBox {
                color: #333333;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #1976D2;
                border-radius: 4px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #1976D2;
                border-color: #1976D2;
            }
        """)
        self.resume_input.setChecked(Config.training.resume)
        self.advanced_layout.addWidget(self.resume_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Training vom letzten Checkpoint fortsetzen."
        )
        self.advanced_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Multi-scale training (DETECTION ONLY)
        self.multi_scale_label = QLabel("Multi-Scale Training:")
        self.advanced_layout.addWidget(self.multi_scale_label, row, 0)
        self.multi_scale_input = QCheckBox()
        self.multi_scale_input.setStyleSheet(self.resume_input.styleSheet())
        self.multi_scale_input.setChecked(Config.training.multi_scale)
        self.advanced_layout.addWidget(self.multi_scale_input, row, 1)
        
        self.multi_scale_info = ParameterInfoButton(
            "Training mit verschiedenen Bildgrößen (nur für Detection empfohlen)."
        )
        self.advanced_layout.addWidget(self.multi_scale_info, row, 3)
        
        row += 1
        # Copy-paste augmentation (SEGMENTATION ONLY)
        self.copy_paste_label = QLabel("Copy-Paste Augmentation:")
        self.advanced_layout.addWidget(self.copy_paste_label, row, 0)
        self.copy_paste_input = QCheckBox()
        self.copy_paste_input.setStyleSheet(self.resume_input.styleSheet())
        self.copy_paste_input.setChecked(False)
        self.advanced_layout.addWidget(self.copy_paste_input, row, 1)
        
        self.copy_paste_info = ParameterInfoButton(
            "Spezielle Augmentation für Segmentierung (kopiert Objekte zwischen Bildern)."
        )
        self.advanced_layout.addWidget(self.copy_paste_info, row, 3)
        
        row += 1
        # Mask ratio (SEGMENTATION ONLY)
        self.mask_ratio_label = QLabel("Mask Ratio:")
        self.advanced_layout.addWidget(self.mask_ratio_label, row, 0)
        self.mask_ratio_input = QSpinBox()
        self.mask_ratio_input.setRange(1, 8)
        self.mask_ratio_input.setValue(4)
        self.advanced_layout.addWidget(self.mask_ratio_input, row, 1)
        
        self.mask_ratio_info = ParameterInfoButton(
            "Verhältnis der Masken-Auflösung zur Bild-Auflösung (nur Segmentierung)."
        )
        self.advanced_layout.addWidget(self.mask_ratio_info, row, 3)
        
        row += 1
        # Cosine LR scheduling
        self.advanced_layout.addWidget(QLabel("Cosine Learning Rate:"), row, 0)
        self.cos_lr_input = QCheckBox()
        self.cos_lr_input.setStyleSheet(self.resume_input.styleSheet())
        self.cos_lr_input.setChecked(Config.training.cos_lr)
        self.advanced_layout.addWidget(self.cos_lr_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Lernrate nach Cosinus-Schema reduzieren."
        )
        self.advanced_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Close mosaic
        self.advanced_layout.addWidget(QLabel("Close Mosaic Epochs:"), row, 0)
        self.close_mosaic_input = QSpinBox()
        self.close_mosaic_input.setRange(0, 15)
        self.close_mosaic_input.setValue(Config.training.close_mosaic)
        self.advanced_layout.addWidget(self.close_mosaic_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Letzte Epochen ohne Mosaic-Augmentation für Feinabstimmung."
        )
        self.advanced_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Momentum
        self.advanced_layout.addWidget(QLabel("Momentum:"), row, 0)
        self.momentum_input = QDoubleSpinBox()
        self.momentum_input.setRange(0.8, 0.999)
        self.momentum_input.setDecimals(3)
        self.momentum_input.setValue(Config.training.momentum)
        self.momentum_input.setSingleStep(0.01)
        self.advanced_layout.addWidget(self.momentum_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Momentum-Parameter für den Optimizer."
        )
        self.advanced_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Warmup epochs
        self.advanced_layout.addWidget(QLabel("Warmup Epochs:"), row, 0)
        self.warmup_epochs_input = QSpinBox()
        self.warmup_epochs_input.setRange(0, 10)
        self.warmup_epochs_input.setValue(Config.training.warmup_epochs)
        self.advanced_layout.addWidget(self.warmup_epochs_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Epochen mit langsamer Lernraten-Erhöhung am Trainingsanfang."
        )
        self.advanced_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Warmup momentum
        self.advanced_layout.addWidget(QLabel("Warmup Momentum:"), row, 0)
        self.warmup_momentum_input = QDoubleSpinBox()
        self.warmup_momentum_input.setRange(0.0, 1.0)
        self.warmup_momentum_input.setDecimals(2)
        self.warmup_momentum_input.setValue(Config.training.warmup_momentum)
        self.advanced_layout.addWidget(self.warmup_momentum_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Anfangswert für Momentum während der Warmup-Phase."
        )
        self.advanced_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Box loss gain
        self.advanced_layout.addWidget(QLabel("Box Loss Gain:"), row, 0)
        self.box_input = QSpinBox()
        self.box_input.setRange(3, 10)
        self.box_input.setValue(Config.training.box)
        self.advanced_layout.addWidget(self.box_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Gewichtungsfaktor für den Bounding-Box-Lokalisierungs-Loss."
        )
        self.advanced_layout.addWidget(info_button, row, 3)
        
        row += 1
        # Dropout
        self.advanced_layout.addWidget(QLabel("Dropout:"), row, 0)
        self.dropout_input = QDoubleSpinBox()
        self.dropout_input.setRange(0.0, 0.5)
        self.dropout_input.setDecimals(2)
        self.dropout_input.setValue(Config.training.dropout)
        self.dropout_input.setSingleStep(0.05)
        self.advanced_layout.addWidget(self.dropout_input, row, 1)
        
        info_button = ParameterInfoButton(
            "Dropout-Rate zur Verhinderung von Überanpassung."
        )
        self.advanced_layout.addWidget(info_button, row, 3)
        
        parent_layout.addWidget(advanced_frame)

    def create_progress_control_section(self, parent_layout):
        """Create progress and control section."""
        progress_frame = QFrame()
        progress_frame.setFrameShape(QFrame.Shape.StyledPanel)
        progress_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
        """)
        progress_layout = QVBoxLayout(progress_frame)

        # Progress bar with label
        progress_label = QLabel("Training Progress:")
        progress_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        progress_layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (0/%m epochs)")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cbd5e0;
                border-radius: 4px;
                text-align: center;
                color: #2d3748;
                background-color: #edf2f7;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        # Progress detail label
        self.progress_detail = QLabel("Ready to start training")
        progress_layout.addWidget(self.progress_detail)

        # Control buttons
        buttons_layout = QHBoxLayout()

        # Start/Stop button
        self.start_button = QPushButton("Start Training")
        self.start_button.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.start_button.setMinimumHeight(40)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_button.clicked.connect(self.start_training)
        buttons_layout.addWidget(self.start_button)

        # Reset button
        self.reset_button = QPushButton("Reset Form")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.clicked.connect(self.reset_form)
        buttons_layout.addWidget(self.reset_button)

        progress_layout.addLayout(buttons_layout)
        parent_layout.addWidget(progress_frame)

    def create_dashboard_panel(self):
        """Create the dashboard panel."""
        self.dashboard_panel = QWidget()
        self.dashboard_layout = QVBoxLayout(self.dashboard_panel)

        # Create dashboard tabs
        self.tabs, self.figure, self.canvas, self.log_text = create_dashboard_tabs(self)
        self.dashboard_layout.addWidget(self.tabs)

    def initialize_model_settings(self):
        """Initialize model settings based on project manager."""
        try:
            if self.project_manager:
                # Detect annotation type and set appropriate default
                recommended_type = self.project_manager.get_recommended_model_type()
                if recommended_type == "segmentation":
                    self.model_type_input.setCurrentText("Segmentation")
                else:
                    self.model_type_input.setCurrentText("Detection")

            # Update model options and connect signals AFTER all UI elements are created
            self.update_model_options()
            
            # NOW connect the signal after everything is initialized
            self.model_type_input.currentTextChanged.connect(self.on_model_type_changed)

        except Exception as e:
            logger.error(f"Error initializing model settings: {e}")
            # Fallback to defaults
            self.update_model_options()
            self.model_type_input.currentTextChanged.connect(self.on_model_type_changed)

    def on_model_type_changed(self):
        """Handle model type change."""
        try:
            self.update_model_options()
            model_type = self.model_type_input.currentText().lower()
            self.update_ui_for_model_type(model_type)
        except Exception as e:
            logger.error(f"Error handling model type change: {e}")

    def update_model_options(self):
        """Update available model options based on selected type."""
        if not self.model_input:
            return
            
        model_type = self.model_type_input.currentText().lower()

        self.model_input.clear()

        if model_type == "segmentation":
            models = [
                "yolo11n-seg.pt",
                "yolo11s-seg.pt",
                "yolo11m-seg.pt",
                "yolo11l-seg.pt",
                "yolo11x-seg.pt"
            ]
        else:  # detection
            models = [
                "yolo11n.pt",
                "yolo11s.pt",
                "yolo11m.pt",
                "yolo11l.pt",
                "yolo11x.pt"
            ]

        self.model_input.addItems(models)

        # Set default based on project manager if available
        if self.project_manager:
            try:
                default_model = self.project_manager.get_default_model_path(model_type)
                index = self.model_input.findText(default_model)
                if index >= 0:
                    self.model_input.setCurrentIndex(index)
            except:
                pass  # Use first item as default

    def update_ui_for_model_type(self, model_type):
        """Update UI visibility based on selected model type."""
        # Safety check - only proceed if all UI elements exist
        required_elements = [
            'multi_scale_label', 'multi_scale_input', 'multi_scale_info',
            'copy_paste_label', 'copy_paste_input', 'copy_paste_info',
            'mask_ratio_label', 'mask_ratio_input', 'mask_ratio_info'
        ]
        
        for element in required_elements:
            if not hasattr(self, element) or getattr(self, element) is None:
                logger.warning(f"UI element {element} not ready for update")
                return
            
        is_segmentation = model_type == "segmentation"
        
        # Multi-scale training - hide for segmentation (not recommended)
        self.multi_scale_label.setVisible(not is_segmentation)
        self.multi_scale_input.setVisible(not is_segmentation)
        self.multi_scale_info.setVisible(not is_segmentation)
        
        # Segmentation-specific parameters - show only for segmentation
        self.copy_paste_label.setVisible(is_segmentation)
        self.copy_paste_input.setVisible(is_segmentation)
        self.copy_paste_info.setVisible(is_segmentation)
        
        self.mask_ratio_label.setVisible(is_segmentation)
        self.mask_ratio_input.setVisible(is_segmentation)
        self.mask_ratio_info.setVisible(is_segmentation)
        
        # Update batch size recommendations in tooltip
        if hasattr(self, 'batch_input'):
            if is_segmentation:
                self.batch_input.setToolTip(
                    "Segmentierung benötigt mehr GPU-Speicher als Detection.\n"
                    "Reduzieren Sie diesen Wert bei Out-of-Memory-Fehlern."
                )
            else:
                self.batch_input.setToolTip(
                    "Detection Training ist speicher-effizienter als Segmentierung.\n"
                    "Höhere Werte sind meist möglich."
                )

    def browse_project(self):
        """Open file dialog to select project directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Projektverzeichnis wählen", 
            self.project_input.text()
        )
        if directory:
            self.project_input.setText(directory)

    def browse_data(self):
        """Open file dialog to select YAML dataset file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Datenpfad auswählen", "", "YAML Dateien (*.yaml)"
        )
        if file_path:
            self.data_input.setText(file_path)

    def reset_form(self):
        """Reset form to default values."""
        if self.training_active:
            QMessageBox.warning(
                self,
                "Training aktiv",
                "Das Formular kann nicht zurückgesetzt werden, während ein Training läuft.",
            )
            return
            
        # Reset project and experiment fields
        if hasattr(self, "project_manager") and self.project_manager:
            self.project_input.setText(str(self.project_manager.get_models_dir()))
        else:
            self.project_input.setText(Config.training.project_dir)
        self.name_input.setText("")
        self.data_input.setText("")

        # Reset training parameters
        self.epochs_input.setValue(Config.training.epochs)
        self.imgsz_input.setValue(Config.training.image_size)
        self.batch_input.setValue(Config.training.batch)
        self.lr_input.setValue(Config.training.lr0)

        # Reset advanced parameters
        self.resume_input.setChecked(Config.training.resume)
        if hasattr(self, 'multi_scale_input'):
            self.multi_scale_input.setChecked(Config.training.multi_scale)
        self.cos_lr_input.setChecked(Config.training.cos_lr)
        self.close_mosaic_input.setValue(Config.training.close_mosaic)
        self.momentum_input.setValue(Config.training.momentum)
        self.warmup_epochs_input.setValue(Config.training.warmup_epochs)
        self.warmup_momentum_input.setValue(Config.training.warmup_momentum)
        self.box_input.setValue(Config.training.box)
        self.dropout_input.setValue(Config.training.dropout)
        
        # Reset segmentation-specific parameters
        if hasattr(self, 'copy_paste_input'):
            self.copy_paste_input.setChecked(False)
        if hasattr(self, 'mask_ratio_input'):
            self.mask_ratio_input.setValue(4)

        # Reset progress and logs
        self.progress_bar.setValue(0)
        self.progress_detail.setText("Ready to start training")
        self.log_text.setText("Training log will appear here...")

        # Reset dashboard
        self.setup_plots()

    def start_training(self):
        """Validate inputs and start the training process."""
        if self.training_active:
            # Currently training, stop it
            self.stop_training()
            return

        # Validate YAML file
        data_path = self.data_input.text().strip()
        if not data_path:
            QMessageBox.critical(
                self, "Validierungsfehler", "Bitte wählen Sie eine YAML-Datei aus."
            )
            return

        # Get current model type for validation
        model_type_str = self.model_type_input.currentText().lower()

        # Use enhanced validation that considers model type
        is_valid, message = validate_yaml_for_model_type(data_path, model_type_str)
        if not is_valid:
            QMessageBox.critical(
                self,
                "Validierungsfehler",
                f"Fehler in der YAML-Datei:\n{message}",
            )
            return
        elif "Warning:" in message:
            # Show warning but allow to continue
            response = QMessageBox.warning(
                self,
                "Validierungswarnung",
                f"{message}\n\nMöchten Sie trotzdem fortfahren?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if response == QMessageBox.StandardButton.No:
                return
        else:
            logger.info(f"YAML validation successful: {message}")

        # Check GPU availability
        gpu_available, gpu_message = check_gpu()
        if not gpu_available:
            response = QMessageBox.warning(
                self,
                "GPU-Warnung",
                f"{gpu_message}\nMöchten Sie trotzdem fortfahren?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if response == QMessageBox.StandardButton.No:
                return

        # Get training parameters
        model_type = self.model_type_input.currentText().lower()
        epochs = self.epochs_input.value()
        imgsz = self.imgsz_input.value()
        batch = float(self.batch_input.value())
        lr0 = self.lr_input.value()
        resume = self.resume_input.isChecked()
        multi_scale = self.multi_scale_input.isChecked() if hasattr(self, 'multi_scale_input') else False
        cos_lr = self.cos_lr_input.isChecked()
        close_mosaic = self.close_mosaic_input.value()
        momentum = self.momentum_input.value()
        warmup_epochs = self.warmup_epochs_input.value()
        warmup_momentum = self.warmup_momentum_input.value()
        box = self.box_input.value()
        dropout = self.dropout_input.value()
        
        # Get segmentation-specific parameters
        copy_paste = 0.0
        mask_ratio = 4
        if (model_type == "segmentation" and 
            hasattr(self, 'copy_paste_input') and 
            self.copy_paste_input.isVisible()):
            copy_paste = 0.3 if self.copy_paste_input.isChecked() else 0.0
        if (model_type == "segmentation" and 
            hasattr(self, 'mask_ratio_input') and 
            self.mask_ratio_input.isVisible()):
            mask_ratio = self.mask_ratio_input.value()

        # Get model path
        model_path = self.model_input.currentText()
        if not model_path:
            model_path = "yolo11n-seg.pt" if model_type == "segmentation" else "yolo11n.pt"

        # Get project settings
        self.project = self.project_input.text() or Config.training.project_dir
        self.experiment = self.name_input.text() or Config.training.experiment_name

        # Update UI to show training is active
        self.training_active = True
        self.start_button.setText("Stop Training")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        
        # Disable settings controls during training
        self.disable_settings(True)

        # Reset progress
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"%p% (0/{epochs} epochs)")
        self.progress_detail.setText("Initializing training...")

        # Start monitoring for results.csv
        self.results_check_timer.start()

        # Start training in a separate thread
        start_training_thread(
            self.signals,
            data_path,
            epochs,
            imgsz,
            batch,
            lr0,
            resume,
            multi_scale,
            cos_lr,
            close_mosaic,
            momentum,
            warmup_epochs,
            warmup_momentum,
            box,
            dropout,
            copy_paste,
            mask_ratio,
            self.project,
            self.experiment,
            model_path,
        )
        
        # Save settings to project
        self.save_training_settings_to_project()

    def stop_training(self):
        """Stop the current training process."""
        stop_training(self.project, self.experiment)
        self.signals.progress_updated.emit(0, "Training stopped by user")
        self.training_active = False
        self.start_button.setText("Start Training")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.disable_settings(False)
        self.results_check_timer.stop()

    def disable_settings(self, disabled):
        """Enable or disable settings controls."""
        # Project and experiment inputs
        if hasattr(self, 'project_input'):
            self.project_input.setReadOnly(disabled)
        if hasattr(self, 'project_browse'):
            self.project_browse.setEnabled(not disabled)
        if hasattr(self, 'name_input'):
            self.name_input.setReadOnly(disabled)
        if hasattr(self, 'data_input'):
            self.data_input.setReadOnly(disabled)
        if hasattr(self, 'data_browse'):
            self.data_browse.setEnabled(not disabled)

        # Basic training parameters
        if hasattr(self, 'epochs_input'):
            self.epochs_input.setReadOnly(disabled)
        if hasattr(self, 'imgsz_input'):
            self.imgsz_input.setReadOnly(disabled)
        if hasattr(self, 'batch_input'):
            self.batch_input.setReadOnly(disabled)
        if hasattr(self, 'lr_input'):
            self.lr_input.setReadOnly(disabled)

        # Advanced parameters
        if hasattr(self, 'resume_input'):
            self.resume_input.setEnabled(not disabled)
        if hasattr(self, 'multi_scale_input'):
            self.multi_scale_input.setEnabled(not disabled)
        if hasattr(self, 'cos_lr_input'):
            self.cos_lr_input.setEnabled(not disabled)
        if hasattr(self, 'close_mosaic_input'):
            self.close_mosaic_input.setReadOnly(disabled)
        if hasattr(self, 'momentum_input'):
            self.momentum_input.setReadOnly(disabled)
        if hasattr(self, 'warmup_epochs_input'):
            self.warmup_epochs_input.setReadOnly(disabled)
        if hasattr(self, 'warmup_momentum_input'):
            self.warmup_momentum_input.setReadOnly(disabled)
        if hasattr(self, 'box_input'):
            self.box_input.setReadOnly(disabled)
        if hasattr(self, 'dropout_input'):
            self.dropout_input.setReadOnly(disabled)
        
        # Segmentation-specific parameters
        if hasattr(self, 'copy_paste_input'):
            self.copy_paste_input.setEnabled(not disabled)
        if hasattr(self, 'mask_ratio_input'):
            self.mask_ratio_input.setReadOnly(disabled)

        # Reset button
        if hasattr(self, 'reset_button'):
            self.reset_button.setEnabled(not disabled)

    def update_progress(self, progress, message=""):
        """Update the progress bar and handle training completion/errors."""
        try:
            if progress >= 100:
                self.training_active = False
                self.start_button.setText("Start Training")
                self.start_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                self.progress_bar.setValue(100)
                self.progress_detail.setText("Training complete!")
                self.disable_settings(False)
                self.results_check_timer.stop()
                if not message:  # Only show success if no error
                    QMessageBox.information(
                        self, "Training", "Training erfolgreich abgeschlossen!"
                    )

                if hasattr(self, "project_manager") and self.project_manager:
                    # Trainiertes Modell registrieren
                    model_path = os.path.join(
                        self.project, self.experiment, "weights", "best.pt"
                    )
                    try:
                        self.project_manager.register_new_model(model_path)
                        self.project_manager.mark_step_completed(WorkflowStep.TRAINING)
                        self.notify_main_menu()
                    except Exception as e:
                        logger.error(f"Error registering model: {e}")

                # Ask to continue with verification step
                reply = QMessageBox.question(
                    self,
                    "Training abgeschlossen",
                    "Training erfolgreich abgeschlossen!\nWeiter zur Verifikation?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.open_verification_app()

            elif progress == 0 and message:
                self.training_active = False
                self.start_button.setText("Start Training")
                self.start_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                self.progress_bar.setValue(0)
                self.progress_detail.setText("Training failed")
                self.disable_settings(False)
                self.results_check_timer.stop()
                QMessageBox.critical(
                    self, "Fehler", f"Training fehlgeschlossen:\n{message}"
                )
            else:
                # Update progress bar value
                self.progress_bar.setValue(int(progress))

                # Update progress bar text to show current epoch
                epochs = self.epochs_input.value()
                current_epoch = int(progress * epochs / 100)
                self.progress_bar.setFormat(f"%p% ({current_epoch}/{epochs} epochs)")

                if message:
                    self.progress_detail.setText(message)
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
            self.training_active = False
            self.start_button.setEnabled(True)
            self.disable_settings(False)
            self.results_check_timer.stop()

    def update_log(self, log_message):
        """Update the log text area."""
        current_text = self.log_text.text()
        new_text = current_text + "\n" + log_message if current_text else log_message
        self.log_text.setText(new_text)

        # Make sure the tab is visible if important message
        if "error" in log_message.lower() or "exception" in log_message.lower():
            self.tabs.setCurrentIndex(1)  # Switch to log tab

    def check_results_csv(self):
        """Check if results.csv exists and has been updated."""
        from gui.training.training_utils import check_and_load_results_csv

        df = check_and_load_results_csv(
            self.project, self.experiment, self.last_results_check
        )
        if df is not None:
            self.last_results_check = os.path.getmtime(df.filepath)
            self.signals.results_updated.emit(df)

            # Update progress based on epochs
            try:
                current_epoch = int(df["epoch"].max())
                total_epochs = self.epochs_input.value()
                progress = int((current_epoch / total_epochs) * 100)
                self.progress_bar.setValue(progress)
                self.progress_bar.setFormat(
                    f"%p% ({current_epoch}/{total_epochs} epochs)"
                )
                self.progress_detail.setText(
                    f"Training epoch {current_epoch}/{total_epochs}"
                )
            except Exception as e:
                logger.error(f"Error updating progress from results.csv: {e}")

    def update_dashboard(self, df):
        """Update the dashboard with new data."""
        from gui.training.dashboard_view import update_dashboard_plots
        update_dashboard_plots(self, df)

    def setup_plots(self):
        """Initialize the matplotlib plots."""
        from gui.training.dashboard_view import setup_plots
        setup_plots(self.figure, self.canvas)

    def load_last_training_results(self):
        """Lädt Ergebnisse des letzten Trainingslaufs in das Dashboard."""
        if hasattr(self, "project_manager") and self.project_manager:
            last_exp = self.project_manager.get_last_experiment_name()
            if not last_exp:
                return
            from gui.training.training_utils import check_and_load_results_csv
            df = check_and_load_results_csv(
                str(self.project_manager.project_root), last_exp
            )
            if df is not None:
                self.last_results_check = os.path.getmtime(df.filepath)
                self.signals.results_updated.emit(df)        

    def closeEvent(self, event):
        """Handle window close event."""
        # Check if training is active
        if self.training_active:
            # Ask user if they want to stop training
            reply = QMessageBox.question(
                self,
                "Training Active",
                "Training is still in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Stop training
                self.stop_training()
            else:
                # Cancel close
                event.ignore()
                return

    def open_verification_app(self):
        """Open verification window and close training window."""
        try:
            from gui.verification_app import LiveAnnotationApp
            app = LiveAnnotationApp()
            app.project_manager = getattr(self, 'project_manager', None)
            if app.project_manager:
                model_path = app.project_manager.get_current_model_path()
                if not model_path:
                    model_path = app.project_manager.get_latest_model_path()
                if model_path and model_path.exists():
                    app.model_line_edit.setText(str(model_path))

                test_dir = app.project_manager.get_split_dir() / 'test' / 'images'
                if not test_dir.exists() or not list(test_dir.glob('*.jpg')):
                    test_dir = app.project_manager.get_labeled_dir()
                app.folder_line_edit.setText(str(test_dir))

            app.show()
            self.close()
        except Exception as e:
            logger.error(f"Failed to open verification app: {e}")

    def save_training_settings_to_project(self):
        """Speichert Training-Settings ins Projekt"""
        if hasattr(self, "project_manager") and self.project_manager:
            settings = {
                "epochs": self.epochs_input.value(),
                "imgsz": self.imgsz_input.value(),
                "batch": self.batch_input.value(),
                "lr0": self.lr_input.value(),
                "resume": self.resume_input.isChecked(),
                "multi_scale": getattr(self.multi_scale_input, 'isChecked', lambda: False)(),
                "cos_lr": self.cos_lr_input.isChecked(),
                "close_mosaic": self.close_mosaic_input.value(),
                "momentum": self.momentum_input.value(),
                "warmup_epochs": self.warmup_epochs_input.value(),
                "warmup_momentum": self.warmup_momentum_input.value(),
                "box": self.box_input.value(),
                "dropout": self.dropout_input.value(),
                "model_type": self.model_type_input.currentText(),
                "model_path": self.model_input.currentText(),
            }
            
            # Add segmentation-specific settings if available
            if hasattr(self, 'copy_paste_input'):
                settings["copy_paste"] = self.copy_paste_input.isChecked()
            if hasattr(self, 'mask_ratio_input'):
                settings["mask_ratio"] = self.mask_ratio_input.value()

            self.project_manager.update_training_settings(settings)

    def register_trained_model_to_project(self, model_path: str, accuracy: float = None):
        """Registriert trainiertes Modell im Projekt"""
        if hasattr(self, "project_manager") and self.project_manager:
            training_params = {
                "epochs": self.epochs_input.value(),
                "lr0": self.lr_input.value(),
                "batch": self.batch_input.value(),
                "imgsz": self.imgsz_input.value(),
                "model_type": self.model_type_input.currentText(),
            }

            timestamp = self.project_manager.register_new_model(
                model_path, accuracy, training_params
            )

            # Workflow-Schritt markieren
            self.project_manager.mark_step_completed(WorkflowStep.TRAINING)
            self.notify_main_menu()

            return timestamp

    def notify_main_menu(self):
        """Informiert das Hauptmenü, den Workflow-Status zu aktualisieren."""
        try:
            from PyQt6.QtWidgets import QApplication
            from gui.main_menu import MainMenu

            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, MainMenu):
                    widget.update_workflow_status()
                    break
        except Exception as e:
            logger.error(f"Failed to notify main menu: {e}")