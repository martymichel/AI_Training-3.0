from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QMessageBox, QProgressBar, QPlainTextEdit
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from yolo.yolo_train import start_training_threaded
from gui.gui_dashboard import DashboardWindow
import os
class TrainSettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trainings-Einstellungen")
        self.setGeometry(150, 150, 600, 700)

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
        self.imgsz_input.setRange(32, 2048)
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

        # Hintergrundprüfung auf results.csv
        self.check_timer = QTimer()
        self.check_timer.setInterval(5000)  # Alle 5 Sekunden
        self.check_timer.timeout.connect(self.check_results_csv)
        self.check_timer.start()

    def browse_data(self):
        from PyQt6.QtWidgets import QFileDialog
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Datenpfad auswählen", "", "YAML Dateien (*.yaml)")
        if file_path:
            self.data_input.setText(file_path)

    def start_training(self):
        data_path = self.data_input.text()
        epochs = self.epochs_input.value()
        lr0 = self.lr_input.value()
        optimizer = self.optimizer_input.currentText()
        # Bereinige Eingaben (falls nötig)
        self.project = self.project_input.text() or "yolo_training_results"
        self.experiment = self.name_input.text() or "experiment"

        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)

        start_training_threaded(
            data_path, epochs, 640, 16, lr0, optimizer, False, 
            self.project, self.experiment, 
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
