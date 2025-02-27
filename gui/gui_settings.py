"""Settings window for YOLO training configuration."""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QMessageBox, QProgressBar
)
from PyQt6.QtCore import QTimer
from config import Config
from utils.validation import validate_yaml, check_gpu
from yolo.yolo_train import start_training_threaded
from gui.gui_dashboard import DashboardWindow

class TrainSettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
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
        self.project_input.setToolTip("Verzeichnis für die Trainingsergebnisse")
        layout.addWidget(self.project_label)
        layout.addWidget(self.project_input)

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
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(Config.training.epochs)
        self.epochs_input.setToolTip("Anzahl der Trainingsdurchläufe")
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs_input)

        self.imgsz_label = QLabel("Bildgrösse:")
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(32, 2048)
        self.imgsz_input.setValue(Config.training.image_size)
        self.imgsz_input.setToolTip("Größe der Trainingsbilder in Pixeln")
        layout.addWidget(self.imgsz_label)
        layout.addWidget(self.imgsz_input)

        self.batch_label = QLabel("Batch-Grösse:")
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 128)
        self.batch_input.setValue(Config.training.batch_size)
        self.batch_input.setToolTip("Anzahl der Bilder pro Trainingsschritt")
        layout.addWidget(self.batch_label)
        layout.addWidget(self.batch_input)

        self.lr_label = QLabel("Anfangs-Lernrate (lr0):")
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.0001, 1.0)
        self.lr_input.setDecimals(4)
        self.lr_input.setValue(Config.training.learning_rate)
        self.lr_input.setToolTip("Initiale Lernrate für den Optimizer")
        layout.addWidget(self.lr_label)
        layout.addWidget(self.lr_input)

        self.optimizer_label = QLabel("Optimizer:")
        self.optimizer_input = QComboBox()
        self.optimizer_input.addItems(["AdamW", "Adam", "SGD"])
        self.optimizer_input.setCurrentText(Config.training.optimizer)
        self.optimizer_input.setToolTip("Optimierungsalgorithmus für das Training")
        layout.addWidget(self.optimizer_label)
        layout.addWidget(self.optimizer_input)

        self.augment_label = QLabel("Daten-Augmentierung:")
        self.augment_input = QCheckBox("Aktivieren")
        self.augment_input.setChecked(Config.training.augmentation)
        self.augment_input.setToolTip("Automatische Datenerweiterung durch Transformationen")
        layout.addWidget(self.augment_label)
        layout.addWidget(self.augment_input)

        # Training starten
        self.start_button = QPushButton("Training starten")
        self.start_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_button)

        # Fortschrittsanzeige
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Dashboard Button (deaktiviert)
        self.dashboard_button = QPushButton("Dashboard anzeigen...")
        self.dashboard_button.setEnabled(False)
        self.dashboard_button.clicked.connect(self.open_dashboard)
        layout.addWidget(self.dashboard_button)

        self.setLayout(layout)

        # Hintergrundprüfung auf results.csv
        self.check_timer = QTimer()
        self.check_timer.setInterval(Config.ui.update_interval)
        self.check_timer.timeout.connect(self.check_results_csv)
        self.check_timer.start()

    def browse_data(self):
        """Open file dialog to select YAML dataset file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Datenpfad auswählen", "", "YAML Dateien (*.yaml)"
        )
        if file_path:
            self.data_input.setText(file_path)

    def start_training(self):
        """Validate inputs and start the training process."""
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
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.No:
                return

        # Get training parameters
        data_path = self.data_input.text()
        epochs = self.epochs_input.value()
        imgsz = self.imgsz_input.value()
        batch = self.batch_input.value()
        lr0 = self.lr_input.value()
        optimizer = self.optimizer_input.currentText()
        augment = self.augment_input.isChecked()

        # Get project settings
        self.project = self.project_input.text() or Config.training.project_dir
        self.experiment = self.name_input.text() or Config.training.experiment_name

        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)

        start_training_threaded(
            data_path, epochs, imgsz, batch, lr0, optimizer, augment,
            self.project, self.experiment,
            callback=self.update_progress
        )

    def check_results_csv(self):
        """Check if results.csv exists and enable dashboard button."""
        base_path = os.path.join(self.project, self.experiment)
        for root, dirs, files in os.walk(base_path):
            if "results.csv" in files:
                self.dashboard_button.setEnabled(True)
                return

    def open_dashboard(self):
        """Open the training dashboard window."""
        self.dashboard = DashboardWindow(
            self.project,
            self.experiment,
            total_epochs=self.epochs_input.value()
        )
        self.dashboard.show()

    def update_progress(self, progress, error_message=""):
        """Update the progress bar and handle training completion/errors."""
        try:
            if progress >= 100:
                self.start_button.setEnabled(True)
                self.progress_bar.setValue(100)
                QMessageBox.information(self, "Training", "Training erfolgreich abgeschlossen!")
            elif progress == 0 and error_message:
                self.start_button.setEnabled(True)
                self.progress_bar.setValue(0)
                QMessageBox.critical(self, "Fehler", f"Training fehlgeschlagen:\n{error_message}")
            else:
                self.progress_bar.setValue(int(progress))
        except Exception as e:
            print(f"Fehler beim Update des Fortschritts: {e}")