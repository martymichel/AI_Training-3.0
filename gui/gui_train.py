from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QMessageBox, QProgressBar, QPlainTextEdit
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from yolo.yolo_train import start_training_threaded
from gui.gui_dashboard import DashboardWindow
from project_manager import ProjectManager
import os
class TrainSettingsWindow(QWidget):
    def __init__(self, project_manager=None):
        super().__init__()
        self.setWindowTitle("Trainings-Einstellungen")
        self.setGeometry(150, 150, 600, 700)

        # Project Manager Integration
        self.project_manager = project_manager

        # Standardwerte setzen, damit check_results_csv() keinen Fehler wirft
        self.project = "yolo_training_results"
        self.experiment = "experiment"

        layout = QVBoxLayout()

        # Objekt/Artikel (Projektverzeichnis)
        self.project_label = QLabel("Objekt/Artikel (Projektverzeichnis):")
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("Beispiel: yolo_training_results")
        layout.addWidget(self.project_label)
        layout.addWidget(self.project_input)

        # Trainingsbezeichnung (Name)
        self.name_label = QLabel("Trainingsbezeichnung:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Beispiel: experiment_v1")
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)        

        # Datenpfad
        self.data_label = QLabel("Datenpfad zur .yaml-Datei des Bilder-Dataset:")
        self.data_input = QLineEdit()
        self.data_browse = QPushButton("Durchsuchen")
        self.data_browse.clicked.connect(self.browse_data)
        layout.addWidget(self.data_label)
        layout.addWidget(self.data_input)
        layout.addWidget(self.data_browse)

        # Epochen
        self.epochs_label = QLabel("Anzahl Epochen:")
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs_input)

        # Bildgrösse
        self.imgsz_label = QLabel("Bildgrösse:")
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(320, 1280)  # Updated range for segmentation support
        self.imgsz_input.setValue(640)
        layout.addWidget(self.imgsz_label)
        layout.addWidget(self.imgsz_input)

        # Batch-Grösse
        self.batch_label = QLabel("Batch-Grösse:")
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 128)
        self.batch_input.setValue(16)
        layout.addWidget(self.batch_label)
        layout.addWidget(self.batch_input)

        # Lernrate (lr0)
        self.lr_label = QLabel("Anfangs-Lernrate (lr0):")
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.0001, 1.0)
        self.lr_input.setDecimals(4)
        self.lr_input.setValue(0.01)
        layout.addWidget(self.lr_label)
        layout.addWidget(self.lr_input)

        # Model Type and Model Selection
        self.model_type_label = QLabel("Model-Typ:")
        self.model_type_input = QComboBox()
        self.model_type_input.addItems(["Detection", "Segmentation", "Nachtraining"])
        self.model_type_input.currentTextChanged.connect(self.update_model_options)
        layout.addWidget(self.model_type_label)
        layout.addWidget(self.model_type_input)

        # Model Selection
        self.model_label = QLabel("Modell:")
        self.model_input = QComboBox()
        self.model_input.setEditable(False)  # No custom model paths by default
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_input)
        
        # Browse button for custom models (hidden by default)
        self.model_browse_button = QPushButton("Modell-Datei auswählen...")
        self.model_browse_button.clicked.connect(self.browse_model_file)
        self.model_browse_button.hide()
        layout.addWidget(self.model_browse_button)
        
        # Custom model path display (hidden by default)
        self.custom_model_path = QLabel()
        self.custom_model_path.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        self.custom_model_path.hide()
        layout.addWidget(self.custom_model_path)

        # Optimizer Auswahl mit AdamW als Standard
        self.optimizer_label = QLabel("Optimizer:")
        self.optimizer_input = QComboBox()
        self.optimizer_input.addItems(["AdamW", "Adam", "SGD"])
        self.optimizer_input.setCurrentText("AdamW")  # Standard auf AdamW
        layout.addWidget(self.optimizer_label)
        layout.addWidget(self.optimizer_input)

        # Augmentierung
        self.augment_label = QLabel("Daten-Augmentierung:")
        self.augment_input = QCheckBox("Aktivieren")
        self.augment_input.setChecked(True)
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

        # Neues Ausgabefeld als eingebautes Terminal (kleine Schrift)
        # self.console_output = QPlainTextEdit()
        # self.console_output.setReadOnly(True)
        # font = QFont("Courier", 8)
        # self.console_output.setFont(font)
        # layout.addWidget(self.console_output)

        self.setLayout(layout)

        # Initialize model options based on project manager
        self.initialize_model_settings()

        # Initialize custom model path
        self.selected_model_path = None

        # Hintergrundprüfung auf results.csv
        self.check_timer = QTimer()
        self.check_timer.setInterval(5000)  # Alle 5 Sekunden
        self.check_timer.timeout.connect(self.check_results_csv)
        self.check_timer.start()

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

                # Set default model path
                default_model = self.project_manager.get_default_model_path()
            else:
                default_model = "yolo11n.pt"

            # Update model options
            self.update_model_options()

        except Exception as e:
            print(f"Error initializing model settings: {e}")
            # Fallback to defaults
            self.update_model_options()

    def browse_model_file(self):
        """Browse for pre-trained model file for continual training."""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Vortrainiertes Modell auswählen", 
            "", 
            "PyTorch Modelle (*.pt);;Alle Dateien (*)"
        )
        if file_path:
            self.selected_model_path = file_path
            model_name = os.path.basename(file_path)
            self.custom_model_path.setText(f"Ausgewählt: {model_name}")
    def update_model_options(self):
        """Update available model options based on selected type."""
        model_type = self.model_type_input.currentText().lower()

        self.model_input.clear()
        
        # Show/hide browse controls based on model type
        if model_type == "nachtraining":
            # Hide dropdown, show browse button
            self.model_input.hide()
            self.model_browse_button.show()
            self.custom_model_path.show()
            self.model_label.setText("Vortrainiertes Modell:")
            
            # Load current model if available from project
            if self.project_manager:
                current_model = self.project_manager.get_latest_model_path()
                if current_model and current_model.exists():
                    self.custom_model_path.setText(f"Ausgewählt: {current_model.name}")
                    self.selected_model_path = str(current_model)
                else:
                    self.custom_model_path.setText("Kein Modell ausgewählt - Bitte wählen Sie eine .pt Datei")
                    self.selected_model_path = None
            return
        else:
            # Show dropdown, hide browse button
            self.model_input.show()
            self.model_browse_button.hide()
            self.custom_model_path.hide()
            self.model_label.setText("Modell:")

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

    def browse_data(self):
        from PyQt6.QtWidgets import QFileDialog
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Datenpfad auswählen", "", "YAML Dateien (*.yaml)")
        if file_path:
            self.data_input.setText(file_path)

    def start_training(self):
        data_path = self.data_input.text()
        epochs = self.epochs_input.value()
        imgsz = self.imgsz_input.value()
        batch = self.batch_input.value()
        lr0 = self.lr_input.value()
        optimizer = self.optimizer_input.currentText()
        
        # Get model path based on mode
        model_type = self.model_type_input.currentText().lower()
        if model_type == "nachtraining":
            if not hasattr(self, 'selected_model_path') or not self.selected_model_path:
                QMessageBox.warning(self, "Fehler", "Bitte wählen Sie ein vortrainiertes Modell für das Nachtraining aus.")
                return
            model_path = self.selected_model_path
        else:
            model_path = self.model_input.currentText()

        # Bereinige Eingaben (falls nötig)
        self.project = self.project_input.text() or "yolo_training_results"
        self.experiment = self.name_input.text() or "experiment"

        # Validation
        if not data_path:
            QMessageBox.warning(self, "Fehler", "Bitte wählen Sie eine YAML-Datei aus.")
            return

        if not model_path:
            QMessageBox.warning(self, "Fehler", "Bitte wählen Sie ein Modell aus oder wählen Sie eine Modell-Datei.")
            return

        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)

        start_training_threaded(
            data_path, epochs, imgsz, batch, lr0, optimizer, False,
            self.project, self.experiment, model_path,
            callback=self.update_progress
        )

    def check_results_csv(self):
        base_path = os.path.join(self.project, self.experiment)
        for root, dirs, files in os.walk(base_path):
            if "results.csv" in files:
                self.dashboard_button.setEnabled(True)
                return

    def open_dashboard(self):
        # Übergibt zusätzlich die festgelegte Anzahl der Epochen
        self.dashboard = DashboardWindow(self.project, self.experiment, total_epochs=self.epochs_input.value())
        self.dashboard.show()

    def update_progress(self, progress, error_message=""):
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
            print(e)
