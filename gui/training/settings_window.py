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
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from config import Config
from utils.validation import validate_yaml, check_gpu
from gui.training.settings_ui import create_settings_ui
from gui.training.dashboard_view import create_dashboard_tabs
from gui.training.training_thread import (
    TrainingSignals,
    start_training_thread,
    stop_training,
)
from project_manager import ProjectManager, WorkflowStep

# Configure logging
logger = logging.getLogger("training_gui")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)
class TrainSettingsWindow(QMainWindow):
    """Main window for YOLO training with integrated dashboard."""

    def __init__(self):
        super().__init__()
        self.training_active = False
        self.setWindowTitle("YOLO Training Advanced")
        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Initialize training signals
        self.signals = TrainingSignals()
        self.signals.progress_updated.connect(self.update_progress)
        self.signals.log_updated.connect(self.update_log)
        self.signals.results_updated.connect(self.update_dashboard)

        # Initialize from config
        self.project = Config.training.project_dir
        self.experiment = Config.training.experiment_name

        # Initialize last checked time for results.csv
        self.last_results_check = 0
        self.results_check_timer = QTimer()
        self.results_check_timer.setInterval(2000)  # Check every 2 seconds
        self.results_check_timer.timeout.connect(self.check_results_csv)

        # Create main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Left panel (settings)
        self.settings_panel = QWidget()
        self.settings_panel.setStyleSheet(
            """
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
        """
        )
        self.settings_layout = QVBoxLayout(self.settings_panel)

        # Create all settings UI components
        self.settings_controls = create_settings_ui(self)

        # Progress and control section
        progress_frame = QFrame()
        progress_frame.setFrameShape(QFrame.Shape.StyledPanel)
        progress_frame.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
        """
        )
        progress_layout = QVBoxLayout(progress_frame)

        # Progress bar with label
        progress_label = QLabel("Training Progress:")
        progress_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        progress_layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        # Note: We'll update this format dynamically with the actual epochs value
        self.progress_bar.setFormat("%p% (0/%m epochs)")
        self.progress_bar.setStyleSheet(
            """
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
        """
        )
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
        self.start_button.setStyleSheet(
            """
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
        """
        )
        self.start_button.clicked.connect(self.start_training)
        buttons_layout.addWidget(self.start_button)

        # Reset button
        self.reset_button = QPushButton("Reset Form")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.clicked.connect(self.reset_form)
        buttons_layout.addWidget(self.reset_button)

        progress_layout.addLayout(buttons_layout)
        self.settings_layout.addWidget(progress_frame)

        # Right panel (dashboard/log)
        self.dashboard_panel = QWidget()
        self.dashboard_layout = QVBoxLayout(self.dashboard_panel)

        # Create dashboard tabs
        self.tabs, self.figure, self.canvas, self.log_text = create_dashboard_tabs(self)
        self.dashboard_layout.addWidget(self.tabs)

        # Add panels to splitter
        self.splitter.addWidget(self.settings_panel)
        self.splitter.addWidget(self.dashboard_panel)

        # Set initial sizes
        self.splitter.setSizes([400, 800])

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
        self.multi_scale_input.setChecked(Config.training.multi_scale)
        self.cos_lr_input.setChecked(Config.training.cos_lr)
        self.close_mosaic_input.setValue(Config.training.close_mosaic)
        self.momentum_input.setValue(Config.training.momentum)
        self.warmup_epochs_input.setValue(Config.training.warmup_epochs)
        self.warmup_momentum_input.setValue(Config.training.warmup_momentum)
        self.box_input.setValue(Config.training.box)
        self.dropout_input.setValue(Config.training.dropout)

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

        is_valid, error_message = validate_yaml(data_path)
        if not is_valid:
            QMessageBox.critical(
                self,
                "Validierungsfehler",
                f"Fehler in der YAML-Datei:\n{error_message}",
            )
            return

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

        # Update UI to show training is active
        self.training_active = True
        self.start_button.setText("Stop Training")
        self.start_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """
        )
        
        # Disable settings controls during training
        self.disable_settings(True)

        # Reset progress
        self.progress_bar.setValue(0)

        # Update progress bar format with dynamic epochs value
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
            self.project,
            self.experiment,
        )

    def stop_training(self):
        """Stop the current training process."""
        stop_training(self.project, self.experiment)
        # In a real implementation, you would need to signal the training process to stop
        self.signals.progress_updated.emit(0, "Training stopped by user")
        self.training_active = False
        self.start_button.setText("Start Training")
        self.start_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )
        self.disable_settings(False)
        self.results_check_timer.stop()

    def disable_settings(self, disabled):
        """Enable or disable settings controls."""
        # Project and experiment inputs
        self.project_input.setReadOnly(disabled)
        self.project_browse.setEnabled(not disabled)
        self.name_input.setReadOnly(disabled)
        self.data_input.setReadOnly(disabled)
        self.data_browse.setEnabled(not disabled)

        # Basic training parameters
        self.epochs_input.setReadOnly(disabled)
        self.imgsz_input.setReadOnly(disabled)
        self.batch_input.setReadOnly(disabled)
        self.lr_input.setReadOnly(disabled)

        # Advanced parameters
        self.resume_input.setEnabled(not disabled)
        self.multi_scale_input.setEnabled(not disabled)
        self.cos_lr_input.setEnabled(not disabled)
        self.close_mosaic_input.setReadOnly(disabled)
        self.momentum_input.setReadOnly(disabled)
        self.warmup_epochs_input.setReadOnly(disabled)
        self.warmup_momentum_input.setReadOnly(disabled)
        self.box_input.setReadOnly(disabled)
        self.dropout_input.setReadOnly(disabled)

        # Reset button
        self.reset_button.setEnabled(not disabled)


    def update_progress(self, progress, message=""):
        """Update the progress bar and handle training completion/errors."""
        try:
            if progress >= 100:
                self.training_active = False
                self.start_button.setText("Start Training")
                self.start_button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """
                )
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
                self.start_button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """
                )
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

        # Stop timers
        self.results_check_timer.stop()

        # Accept close event
        event.accept()

    def save_training_settings_to_project(self):
        """Speichert Training-Settings ins Projekt"""
        if hasattr(self, "project_manager") and self.project_manager:
            settings = {
                "epochs": self.epochs_input.value(),
                "imgsz": self.imgsz_input.value(),
                "batch": self.batch_input.value(),
                "lr0": self.lr_input.value(),
                "resume": self.resume_input.isChecked(),
                "multi_scale": self.multi_scale_input.isChecked(),
                "cos_lr": self.cos_lr_input.isChecked(),
                "close_mosaic": self.close_mosaic_input.value(),
                "momentum": self.momentum_input.value(),
                "warmup_epochs": self.warmup_epochs_input.value(),
                "warmup_momentum": self.warmup_momentum_input.value(),
                "box": self.box_input.value(),
                "dropout": self.dropout_input.value(),
            }

            self.project_manager.update_training_settings(settings)

    def register_trained_model_to_project(
        self, model_path: str, accuracy: float = None
    ):
        """Registriert trainiertes Modell im Projekt"""
        if hasattr(self, "project_manager") and self.project_manager:
            training_params = {
                "epochs": self.epochs_input.value(),
                "lr0": self.lr_input.value(),
                "batch": self.batch_input.value(),
                "imgsz": self.imgsz_input.value(),
            }

            timestamp = self.project_manager.register_new_model(
                model_path, accuracy, training_params
            )

            # Workflow-Schritt markieren
            self.project_manager.mark_step_completed(WorkflowStep.TRAINING)

            return timestamp

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

"""
Ergänzungen für gui/training/settings_window.py
"""
class TrainingWindowExtensions:
    """Erweiterungen für das Training Window"""

    def save_training_settings_to_project(self):
        """Speichert Training-Settings ins Projekt"""
        if hasattr(self, "project_manager") and self.project_manager:
            settings = {
                "epochs": self.epochs_input.value(),
                "imgsz": self.imgsz_input.value(),
                "batch": self.batch_input.value(),
                "lr0": self.lr_input.value(),
                "resume": self.resume_input.isChecked(),
                "multi_scale": self.multi_scale_input.isChecked(),
                "cos_lr": self.cos_lr_input.isChecked(),
                "close_mosaic": self.close_mosaic_input.value(),
                "momentum": self.momentum_input.value(),
                "warmup_epochs": self.warmup_epochs_input.value(),
                "warmup_momentum": self.warmup_momentum_input.value(),
                "box": self.box_input.value(),
                "dropout": self.dropout_input.value(),
            }

            self.project_manager.update_training_settings(settings)

    def register_trained_model_to_project(
        self, model_path: str, accuracy: float = None
    ):
        """Registriert trainiertes Modell im Projekt"""
        if hasattr(self, "project_manager") and self.project_manager:
            training_params = {
                "epochs": self.epochs_input.value(),
                "lr0": self.lr_input.value(),
                "batch": self.batch_input.value(),
                "imgsz": self.imgsz_input.value(),
            }

            timestamp = self.project_manager.register_new_model(
                model_path, accuracy, training_params
            )

            # Workflow-Schritt markieren
            self.project_manager.mark_step_completed(WorkflowStep.TRAINING)

            return timestamp    