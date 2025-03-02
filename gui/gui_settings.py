"""Settings window for YOLO training configuration."""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QMessageBox, QProgressBar,
    QHBoxLayout, QTextEdit
)
from PyQt6.QtCore import QTimer
from config import Config
from utils.validation import validate_yaml, check_gpu
from yolo.yolo_train import start_training

class TrainSettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.training_active = False
        self.setWindowTitle("YOLO Training Settings")
        self.setGeometry(150, 150, Config.ui.window_width, Config.ui.window_height)

        # Initialize from config
        self.project = Config.training.project_dir
        self.experiment = Config.training.experiment_name

        layout = QVBoxLayout()

        # Project directory with tooltip
        self.project_label = QLabel("Objekt/Artikel (Projektverzeichnis):")
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("Beispiel: yolo_training_results")
        self.project_input.setText(Config.training.project_dir)
        self.project_input.setToolTip("Verzeichnis für die Trainingsergebnisse")
        self.project_browse = QPushButton("Durchsuchen")
        self.project_browse.clicked.connect(self.browse_project)
        layout.addWidget(self.project_label)
        project_layout = QHBoxLayout()
        project_layout.addWidget(self.project_input)
        project_layout.addWidget(self.project_browse)
        layout.addLayout(project_layout)

        # Experiment name with tooltip
        self.name_label = QLabel("Trainingsbezeichnung:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Beispiel: experiment_v1")
        self.name_input.setToolTip("Eindeutiger Name für dieses Training")
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)

        # Data path with validation
        self.data_label = QLabel("Datenpfad zur .yaml-Datei des Bilder-Dataset:")
        self.data_input = QLineEdit()
        self.data_input.setToolTip("YAML-Datei mit Trainingsdaten-Konfiguration")
        self.data_browse = QPushButton("Durchsuchen")
        self.data_browse.clicked.connect(self.browse_data)
        layout.addWidget(self.data_label)
        layout.addWidget(self.data_input)
        layout.addWidget(self.data_browse)

        # Training parameters with tooltips
        self.epochs_label = QLabel("Anzahl Epochen:")
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(5, 300)
        self.epochs_input.setValue(Config.training.epochs)
        self.epochs_input.setToolTip("Anzahl der Trainingsdurchläufe")
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs_input)

        self.imgsz_label = QLabel("Bildgrösse:")
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(640, 1280)
        self.imgsz_input.setValue(Config.training.image_size)
        self.imgsz_input.setToolTip("Größe der Trainingsbilder in Pixeln")
        layout.addWidget(self.imgsz_label)
        layout.addWidget(self.imgsz_input)

        self.batch_label = QLabel("Batch:")
        self.batch_input = QDoubleSpinBox()
        self.batch_input.setRange(0.7, 0.99)
        self.batch_input.setDecimals(2)
        self.batch_input.setValue(Config.training.batch)
        self.batch_input.setToolTip("Batch-Größe als Anteil des verfügbaren Speichers")
        layout.addWidget(self.batch_label)
        layout.addWidget(self.batch_input)

        self.lr_label = QLabel("Anfangs-Lernrate (lr0):")
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.001, 0.01)
        self.lr_input.setDecimals(4)
        self.lr_input.setValue(Config.training.lr0)
        self.lr_input.setToolTip("Initiale Lernrate für den Optimizer")
        layout.addWidget(self.lr_label)
        layout.addWidget(self.lr_input)

        # New parameters
        self.resume_input = QCheckBox("Training fortsetzen")
        self.resume_input.setChecked(Config.training.resume)
        self.resume_input.setToolTip("Training vom letzten Checkpoint fortsetzen")
        layout.addWidget(self.resume_input)

        self.multi_scale_input = QCheckBox("Multi-Scale Training")
        self.multi_scale_input.setChecked(Config.training.multi_scale)
        self.multi_scale_input.setToolTip("Training mit verschiedenen Bildgrößen")
        layout.addWidget(self.multi_scale_input)

        self.cos_lr_input = QCheckBox("Cosine Learning Rate")
        self.cos_lr_input.setChecked(Config.training.cos_lr)
        self.cos_lr_input.setToolTip("Cosinus-basierte Lernratenanpassung")
        layout.addWidget(self.cos_lr_input)

        self.close_mosaic_label = QLabel("Close Mosaic Epochs:")
        self.close_mosaic_input = QSpinBox()
        self.close_mosaic_input.setRange(0, 15)
        self.close_mosaic_input.setValue(Config.training.close_mosaic)
        self.close_mosaic_input.setToolTip("Epochen bis Mosaic Augmentation beendet wird")
        layout.addWidget(self.close_mosaic_label)
        layout.addWidget(self.close_mosaic_input)

        self.momentum_label = QLabel("Momentum:")
        self.momentum_input = QDoubleSpinBox()
        self.momentum_input.setRange(0.9, 0.95)
        self.momentum_input.setDecimals(2)
        self.momentum_input.setValue(Config.training.momentum)
        self.momentum_input.setToolTip("SGD Momentum/Adam Beta1")
        layout.addWidget(self.momentum_label)
        layout.addWidget(self.momentum_input)

        self.warmup_epochs_label = QLabel("Warmup Epochs:")
        self.warmup_epochs_input = QSpinBox()
        self.warmup_epochs_input.setRange(2, 5)
        self.warmup_epochs_input.setValue(Config.training.warmup_epochs)
        self.warmup_epochs_input.setToolTip("Anzahl der Warmup-Epochen")
        layout.addWidget(self.warmup_epochs_label)
        layout.addWidget(self.warmup_epochs_input)

        self.warmup_momentum_label = QLabel("Warmup Momentum:")
        self.warmup_momentum_input = QDoubleSpinBox()
        self.warmup_momentum_input.setRange(0.8, 0.9)
        self.warmup_momentum_input.setDecimals(2)
        self.warmup_momentum_input.setValue(Config.training.warmup_momentum)
        self.warmup_momentum_input.setToolTip("Momentum während Warmup")
        layout.addWidget(self.warmup_momentum_label)
        layout.addWidget(self.warmup_momentum_input)

        self.box_label = QLabel("Box Loss Gain:")
        self.box_input = QSpinBox()
        self.box_input.setRange(7, 8)
        self.box_input.setValue(Config.training.box)
        self.box_input.setToolTip("Box Loss Gewichtung")
        layout.addWidget(self.box_label)
        layout.addWidget(self.box_input)

        self.dropout_label = QLabel("Dropout:")
        self.dropout_input = QDoubleSpinBox()
        self.dropout_input.setRange(0, 0.1)
        self.dropout_input.setDecimals(2)
        self.dropout_input.setValue(Config.training.dropout)
        self.dropout_input.setToolTip("Dropout Regularisierung")
        layout.addWidget(self.dropout_label)
        layout.addWidget(self.dropout_input)

        # Training starten
        self.start_button = QPushButton("Training starten")
        self.start_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_button)

        # Fortschrittsanzeige
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

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

    def start_training(self):
        """Validate inputs and start the training process."""
        if self.training_active:
            QMessageBox.warning(self, "Training aktiv", 
                              "Ein Training läuft bereits.")
            return

        # Validate YAML file
        data_path = self.data_input.text()
        is_valid, error_message = validate_yaml(data_path)
        if not is_valid:
            QMessageBox.critical(self, "Validierungsfehler", 
                               f"Fehler in der YAML-Datei:\n{error_message}")
            return

        # Check GPU availability
        gpu_available, gpu_message = check_gpu()
        if not gpu_available:
            response = QMessageBox.warning(
                self, "GPU-Warnung",
                f"{gpu_message}\nMöchten Sie trotzdem fortfahren?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if response == QMessageBox.StandardButton.No:
                return

        # Get training parameters
        data_path = self.data_input.text()
        epochs = self.epochs_input.value()
        imgsz = self.imgsz_input.value()
        batch = float(self.batch_input.value())
        lr0 = self.lr_input.value()
        resume = self.resume_input.isChecked()
        multi_scale = self.multi_scale_input.isChecked()
        cos_lr = self.cos_lr_input.isChecked()
        close_mosaic = self.close_mosaic_input.value()
        momentum = self.momentum_input.value()
        warmup_epochs = self.warmup_epochs_input.value()
        warmup_momentum = self.warmup_momentum_input.value()
        box = self.box_input.value()
        dropout = self.dropout_input.value()

        # Get project settings
        self.project = self.project_input.text() or Config.training.project_dir
        self.experiment = self.name_input.text() or Config.training.experiment_name

        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        self.training_active = True
        
        try:
            start_training(
                data_path, epochs, imgsz, batch, lr0, resume, multi_scale,
                cos_lr, close_mosaic, momentum, warmup_epochs, warmup_momentum,
                box, dropout,
                self.project, self.experiment,
                progress_callback=self.update_progress,
                log_callback=None
            )
        except Exception as e:
            self.update_progress(0, str(e))
            self.training_active = False
            self.start_button.setEnabled(True)

    def update_progress(self, progress, error_message=""):
        """Update the progress bar and handle training completion/errors."""
        try:
            if progress >= 100:
                self.start_button.setEnabled(True)
                self.progress_bar.setValue(100)
                self.training_active = False
                if not error_message:  # Only show success if no error
                    QMessageBox.information(self, "Training", "Training erfolgreich abgeschlossen!")
            elif progress == 0 and error_message:
                self.start_button.setEnabled(True)
                self.progress_bar.setValue(0)
                self.training_active = False
                QMessageBox.critical(self, "Fehler", f"Training fehlgeschlagen:\n{error_message}")
            else:
                self.progress_bar.setValue(int(progress))
        except Exception as e:
            print(f"Fehler beim Update des Fortschritts: {e}")
            self.training_active = False
            self.start_button.setEnabled(True)

    def closeEvent(self, event):
        """Handle window close event."""
        event.accept()