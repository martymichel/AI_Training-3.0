#!/usr/bin/env python3
"""
Live Annotation App – Ungebremste Verarbeitung mit Logging und Umbenennung falsch annotierter Bilder

- Das Modell wird auf den Originalbildern angewendet.
- Für die Anzeige im Mosaik werden die annotierten Bilder auf eine kleinere Größe skaliert.
- Falsch annotierte Bilder werden im Original (ohne rotes Overlay) gespeichert und umbenannt in
  false_img1, false_img2, ….
- Im Log wird für jedes falsch annotierte Bild festgehalten, welche Klassen (mit Häufigkeiten)
  erwartet und vorhergesagt wurden sowie welche Klassen fehlen bzw. zusätzlich vorhanden sind.
- Der Threshold für die Inferenz ist definierbar über einen Schieberegler.
- Nach Abschluss erscheint eine Zusammenfassung.
"""

import os
import glob
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import json
import yaml


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog,
    QProgressBar, QSlider, QPlainTextEdit, QMessageBox, QSpinBox, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from .verification_core import OptimizeThresholdsWorker, AnnotationWorker
from .verification_utils import (
    validate_model_path, validate_test_folder, get_model_status,
    open_directory, format_summary, setup_logging
)
class LiveAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_worker = None
        self.setWindowTitle("Modell-Verifikation")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
        # Main layout with sidebar
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # Sidebar
        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(300)
        sidebar_container.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-left: 1px solid #ccc;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                margin: 8px;
                color: black;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border-color: #ccc;
            }
            QLabel {
                color: black;
                padding: 4px;
                margin: 2px;
            }
            QLineEdit {
                color: black;
                background: white;
                padding: 6px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        
        # Auswahl YOLO-Modell (best.pt)
        model_layout = QHBoxLayout()
        self.model_line_edit = QLineEdit()
        self.model_button = QPushButton("Modell auswählen")
        self.model_button.clicked.connect(self.browse_model)
        sidebar_layout.addWidget(QLabel("YOLO Modell (best.pt):"))
        model_layout.addWidget(self.model_line_edit)
        model_layout.addWidget(self.model_button)
        sidebar_layout.addLayout(model_layout)
        
        # Auswahl Testdatenverzeichnis
        folder_layout = QHBoxLayout()
        self.folder_line_edit = QLineEdit()
        self.folder_button = QPushButton("Testordner auswählen")
        self.folder_button.clicked.connect(self.browse_folder)
        sidebar_layout.addWidget(QLabel("Testdatenverzeichnis:"))
        folder_layout.addWidget(self.folder_line_edit)
        folder_layout.addWidget(self.folder_button)
        sidebar_layout.addLayout(folder_layout)
        
        # Threshold-Schieberegler
        threshold_layout = QHBoxLayout()
        sidebar_layout.addWidget(QLabel("Schwellenwert:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(25)  # Standard = 0.25
        self.threshold_value_label = QLabel("0.25")
        self.threshold_value_label.setStyleSheet("color: black;")
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        sidebar_layout.addLayout(threshold_layout)
        
        # IoU-Threshold-Schieberegler
        iou_layout = QHBoxLayout()
        sidebar_layout.addWidget(QLabel("IoU Threshold:"))
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)  # Standard = 0.45
        self.iou_value_label = QLabel("0.45")
        self.iou_value_label.setStyleSheet("color: black;")
        self.iou_slider.valueChanged.connect(self.update_iou_label)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_value_label)
        sidebar_layout.addLayout(iou_layout)
        
        # Optimize Thresholds section
        optimize_group = QWidget()
        optimize_layout = QVBoxLayout(optimize_group)
        
        # Step size for optimization
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Schrittweite:"))
        self.step_size_spin = QSpinBox()
        self.step_size_spin.setRange(1, 10)
        self.step_size_spin.setStyleSheet("""
            QSpinBox {
                color: black;
                background: white;
                padding: 4px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        self.step_size_spin.setValue(1)
        self.step_size_spin.setSuffix("%")
        step_layout.addWidget(self.step_size_spin)
        optimize_layout.addLayout(step_layout)
        
        # Optimize button
        self.optimize_button = QPushButton("1. Optimalwerte suchen")
        self.optimize_button.setMinimumHeight(40)
        self.optimize_button.setFont(QFont("", 11))
        self.optimize_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 8px;
                padding: 10px;
                margin: 10px 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.optimize_button.clicked.connect(self.optimize_thresholds)
        optimize_layout.addWidget(self.optimize_button)
        
        sidebar_layout.addWidget(optimize_group)
        
        # Start-Button
        self.start_button = QPushButton("2. Bilder-Annotation starten")
        self.start_button.setMinimumHeight(60)
        self.start_button.setFont(QFont("", 12, QFont.Weight.Bold))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 15px 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_button.clicked.connect(self.start_live_annotation)
        sidebar_layout.addWidget(self.start_button)
        
        # Fortschrittsbalken
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        sidebar_layout.addWidget(self.progress_bar)
        # Schwarze schrift für den Fortschrittsbalken
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                /* allgemein */
                text-align: center;
                font-size: 18px;      /* Grösser */
                font-weight: bold;    /* Fett */
                min-height: 30px;     /* Erhöht die Mindesthöhe, so wird er „dicker“ */
                color: black;
                /* Du kannst hier auch 'border: 1px solid #999;' angeben oder Hintergrundfarbe etc. */
            }
            QProgressBar::chunk {
                background-color: #2196F3; /* Blau gefüllter Teil */
                width: 1px;               /* Default: schmal, 
                                            wird dynamisch abhängig vom Wert */
                margin: 0.5px;            /* Kleiner Abstand pro chunk, 
                                            kann man anpassen */
            }
        """)
        
        # Add stretch to push everything to the top
        sidebar_layout.addStretch()
        
        # Anzeige des Mosaiks
        self.mosaic_label = QLabel()
        self.mosaic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Navigation buttons for mosaic
        nav_layout = QHBoxLayout()
        self.prev_mosaic_btn = QPushButton("← Vorheriges Mosaik")
        self.next_mosaic_btn = QPushButton("Nächstes Mosaik →")

        # Anfangs: verbergen
        self.prev_mosaic_btn.hide()
        self.next_mosaic_btn.hide()

        self.prev_mosaic_btn.setEnabled(False)
        self.next_mosaic_btn.setEnabled(False)
        self.prev_mosaic_btn.clicked.connect(self.show_previous_mosaic)
        self.next_mosaic_btn.clicked.connect(self.show_next_mosaic)
        
        # Style navigation buttons
        nav_button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """
        self.prev_mosaic_btn.setStyleSheet(nav_button_style)
        self.next_mosaic_btn.setStyleSheet(nav_button_style)
        
        nav_layout.addWidget(self.prev_mosaic_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_mosaic_btn)
        
        mosaic_container = QWidget()
        mosaic_layout = QVBoxLayout(mosaic_container)
        mosaic_layout.addWidget(self.mosaic_label)
        mosaic_layout.addLayout(nav_layout)
        content_layout.addWidget(mosaic_container)
        
        # Andon-Anzeige
        self.andon_label = QLabel()
        self.andon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.andon_label.setMinimumHeight(80)
        content_layout.addWidget(self.andon_label)
        
        # Separate label for misannotated directory link
        self.misannotated_link = QLabel()
        self.misannotated_link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.misannotated_link.setMinimumHeight(40)
        content_layout.addWidget(self.misannotated_link)
        
        # Zusammenfassungsausgabefeld (wird nach Beendigung angezeigt)
        self.summary_output = QPlainTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.hide()
        self.summary_output.setStyleSheet("font-size: 16px;") # Grössere Schrift
        content_layout.addWidget(self.summary_output)
        
        # Add main content and sidebar to main layout
        main_layout.addWidget(content_widget, stretch=4)
        main_layout.addWidget(sidebar_container)
        
        self.worker = None
        self.image_list = []
        self.optimize_worker = None

        self.open_plot_dir_button = QPushButton("Bilder des Tunings")
        self.open_plot_dir_button.setEnabled(False)
        self.open_plot_dir_button.clicked.connect(self.open_optimization_plot_dir)
        sidebar_layout.addWidget(self.open_plot_dir_button)

        # Button to jump directly to Live Detection
        self.open_detection_button = QPushButton("→ Live Detection")
        self.open_detection_button.setVisible(False)
        self.open_detection_button.clicked.connect(self.open_live_detection)
        sidebar_layout.addWidget(self.open_detection_button)

        self.session_output_dir = None

    def open_optimization_plot_dir(self):
        if self.optimize_worker and hasattr(self.optimize_worker, "plot_dir"):
            plot_dir = self.optimize_worker.plot_dir
        elif self.session_output_dir:
            plot_dir = self.session_output_dir / "optimization_plots"
        else:
            plot_dir = None
        if plot_dir and os.path.isdir(str(plot_dir)):
            os.startfile(str(plot_dir)) if os.name == 'nt' else os.system(f'xdg-open "{plot_dir}"')

    def open_live_detection(self):
        """Close this app and launch the live detection app."""
        if hasattr(self, "project_manager") and self.project_manager:
            settings_dir = str(self.project_manager.project_root)
            subprocess.Popen([
                sys.executable,
                "-m",
                "gui.camera_app",
                settings_dir,
                "--show-detection",
            ])
        self.close()

    def update_threshold_label(self, value):
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")

    def update_iou_label(self, value):
        iou = value / 100.0
        self.iou_value_label.setText(f"{iou:.2f}")

    def open_misannotated_dir(self, dir_path):
        """Open the directory containing misannotated images."""
        os.startfile(dir_path) if os.name == 'nt' else os.system(f'xdg-open "{dir_path}"')

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "KI-Modell (.pt) auswählen", "", "PT Dateien (*.pt)")
        if file_path:
            self.model_line_edit.setText(file_path)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Test-Dataset auswählen")
        if folder:
            self.folder_line_edit.setText(folder)

    def optimize_thresholds(self):
        """Start threshold optimization process."""
        model_path = self.model_line_edit.text().strip()
        test_folder = self.folder_line_edit.text().strip()
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Bitte wählen Sie eine gültige Modell-Datei aus")
            return
            
        if not os.path.isdir(test_folder):
            QMessageBox.warning(self, "Error", "Bitte wählen Sie ein gültiges Testverzeichnis aus")
            return

        # Collect all images in test folder
        self.image_list = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.image_list.extend(glob.glob(os.path.join(test_folder, ext)))
        self.image_list.sort()
        if not self.image_list:
            QMessageBox.warning(self, "Error", "Keine Bilder im Testverzeichnis gefunden")
            return

        # Prepare output directory
        if hasattr(self, "project_manager") and self.project_manager:
            base_dir = self.project_manager.get_verification_dir() / "verification_app"
        else:
            base_dir = Path("verification_app")
        base_dir.mkdir(parents=True, exist_ok=True)
        if not self.session_output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_output_dir = base_dir / timestamp
            self.session_output_dir.mkdir(parents=True, exist_ok=True)

        # Disable UI during optimization
        self.optimize_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Start optimization worker
        self.optimize_worker = OptimizeThresholdsWorker(
            model_path=model_path,
            image_list=self.image_list,
            step_size=self.step_size_spin.value(),
            output_dir=str(self.session_output_dir)
        )
        self.optimize_worker.progress_updated.connect(self.progress_bar.setValue)
        self.optimize_worker.stage_updated.connect(lambda msg: self.summary_output.setPlainText(msg))
        self.optimize_worker.optimization_finished.connect(self.optimization_complete)
        self.optimize_worker.start()
        
        # Update status
        self.summary_output.setPlainText("Starting Tuning...")
        self.summary_output.show()

    def optimization_complete(self, results):
        """Handle completion of threshold optimization."""
        self.optimize_button.setEnabled(True)
        self.start_button.setEnabled(True)
        
        if not results:
            QMessageBox.warning(self, "Error", "Schwellenwert Tuning fehlgeschlagen")
            return

        # Button aktivieren
        self.open_plot_dir_button.setEnabled(True)    
        
        # Zusammenbauen einer kurzen Summary mit Link
        conf_value = int(results['conf'] * 100)
        iou_value = int(results['iou'] * 100)

        # Sliderpositionen updaten
        self.threshold_slider.setValue(conf_value)
        self.iou_slider.setValue(iou_value)

        # Persist optimal values to project settings
        if hasattr(self, "project_manager") and self.project_manager:
            class_ids = self.project_manager.get_classes().keys()
            class_thresholds = {str(cid): results['conf'] for cid in class_ids}
            update = {
                'iou_threshold': results['iou'],
                'class_thresholds': class_thresholds,
            }
            self.project_manager.update_live_detection_settings(update)
            det_file = Path(self.project_manager.project_root) / "detection_settings.json"
            try:
                current = {}
                if det_file.exists():
                    with open(det_file, 'r', encoding='utf-8') as f:
                        current = json.load(f)
                current.update(update)
                with open(det_file, 'w', encoding='utf-8') as f:
                    json.dump(current, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save detection settings: {e}")

        # Labels dazu aktualisieren
        self.threshold_value_label.setText(f"{conf_value/100:.2f}")
        self.iou_value_label.setText(f"{iou_value/100:.2f}")        

        summary = (
        f"Schwellenwert-Tuning abgeschlossen\n\n"
            f"Beste Konfiguration:\n"
            f"- Konfidenz Schwellenwert: {results['conf']:.2f}\n"
            f"- IoU Schwellenwert: {results['iou']:.2f}\n"
            f"- Genauigkeit: {results['accuracy']:.1f}%\n\n"
            f"Die Slider wurden automatisch auf diese Einstellung eingestellt.\n"
            f"Du kannst jetzt auf 'Bilder-Annotation starten' klicken, um die Verifikation zu starten."
        )
        
        self.summary_output.setPlainText(summary)
        self.summary_output.show()
        
        QMessageBox.information(
            self,
            "Tuning abgeschlossen",
            f"Optimale Schwellenwerte gefunden:\nKonfidenz: {results['conf']:.2f}\nIoU: {results['iou']:.2f}\n"
            f"Erwartete Modellgenauigkeit: {results['accuracy']:.1f}%"
        )

    def start_live_annotation(self):
        model_path = self.model_line_edit.text().strip()
        test_folder = self.folder_line_edit.text().strip()
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Bitte wählen Sie eine gültige Modell-Datei aus")
            return
            
        if not os.path.isdir(test_folder):
            QMessageBox.warning(self, "Error", "Bitte wählen Sie ein gültiges Testverzeichnis aus")
            return

        # Collect all images in test folder
        self.image_list = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.image_list.extend(glob.glob(os.path.join(test_folder, ext)))
        self.image_list.sort()
        if not self.image_list:
            QMessageBox.warning(self, "Error", "Keine Bilder im Testverzeichnis gefunden")
            return

        # Prepare output directory
        if hasattr(self, "project_manager") and self.project_manager:
            base_dir = self.project_manager.get_verification_dir() / "verification_app"
        else:
            base_dir = Path("verification_app")
        base_dir.mkdir(parents=True, exist_ok=True)
        if not self.session_output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_output_dir = base_dir / timestamp
            self.session_output_dir.mkdir(parents=True, exist_ok=True)

        # Get threshold values from sliders
        threshold = self.threshold_slider.value() / 100.0
        iou_threshold = self.iou_slider.value() / 100.0

        # Start worker thread
        self.worker = AnnotationWorker(
            model_path=model_path,
            image_list=self.image_list,
            output_dir=str(self.session_output_dir),
            threshold=threshold,
            iou_threshold=iou_threshold,
            tile_size=256
        )
        self.worker.mosaic_updated.connect(self.update_mosaic)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.summary_signal.connect(self.show_summary)
        self.worker.finished.connect(self.annotation_finished)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.summary_output.hide()
        self.andon_label.clear()

    def update_mosaic(self, pixmap):
        self.mosaic_label.setPixmap(pixmap)
        # Update navigation buttons
        if self.worker:
            # Anzahl Mosaic-Einträge
            total_mosaics = len(self.worker.mosaic_history)
            # Button nur anzeigen, wenn mehr als 1 Mosaik existiert
            if total_mosaics > 1:
                self.prev_mosaic_btn.show()
                self.next_mosaic_btn.show()
            else:
                self.prev_mosaic_btn.hide()
                self.next_mosaic_btn.hide()
            
            # Prev-Button nur aktiv, wenn wir nicht beim Index=0 sind
            self.prev_mosaic_btn.setEnabled(self.worker.current_mosaic_index > 0)
            # Next-Button nur aktiv, wenn wir nicht beim letzten Index sind
            self.next_mosaic_btn.setEnabled(
                self.worker.current_mosaic_index < total_mosaics - 1
            )

    def show_previous_mosaic(self):
        """Show the previous mosaic from history."""
        if self.worker and self.worker.current_mosaic_index > 0:
            self.worker.current_mosaic_index -= 1
            self.mosaic_label.setPixmap(self.worker.mosaic_history[self.worker.current_mosaic_index])
            self.prev_mosaic_btn.setEnabled(self.worker.current_mosaic_index > 0)
            self.next_mosaic_btn.setEnabled(True)

    def show_next_mosaic(self):
        """Show the next mosaic from history."""
        if self.worker and self.worker.current_mosaic_index < len(self.worker.mosaic_history) - 1:
            self.worker.current_mosaic_index += 1
            self.mosaic_label.setPixmap(self.worker.mosaic_history[self.worker.current_mosaic_index])
            self.next_mosaic_btn.setEnabled(
                self.worker.current_mosaic_index < len(self.worker.mosaic_history) - 1
            )
            self.prev_mosaic_btn.setEnabled(True)

    def show_summary(self, summary_text, correct_percentage, misannotated_dir):
        # Add FileHandler to logger now that we have the misannotated_dir
        log_file = os.path.join(misannotated_dir, "live_annotation.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Update summary text
        self.summary_output.setPlainText(summary_text)
        self.summary_output.show()
        
        # Update Andon display
        if correct_percentage >= 98:
            color = "#4CAF50"  # Green
            status = "KI-MODELL SEHR GUT"
        elif correct_percentage >= 95:
            color = "#FF9800"  # Orange
            status = "KI-MODELL AKZEPTABEL"
        else:
            color = "#F44336"  # Red
            status = "KI-MODELL UNGENÜGEND"
            
        # Update Andon display (status only)
        self.andon_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: white;
            background-color: {color};
            border-radius: 5px;
            border-radius: 10px;
            padding: 10px;
            margin: 10px;
        """)
        self.andon_label.setText(f"Status: {status} ({correct_percentage:.1f}%)")
        
        # Update misannotated directory link (separate)
        self.misannotated_link.setStyleSheet("""
            font-size: 14px;
            color: #2196F3;
            padding: 10px;
        """)
        self.misannotated_link.setText(f"<a href='{misannotated_dir}' style='color: #2196F3;'>Klicken Sie hier um den Ordner mit falsch annotierten Bildern zu öffnen</a>")
        self.misannotated_link.setOpenExternalLinks(False)
        self.misannotated_link.linkActivated.connect(self.open_misannotated_dir)

        # Show button to jump to live detection
        self.open_detection_button.setVisible(True)

    def annotation_finished(self):
        self.start_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        logger.info("Test-Annotation & Modellverifikation abgeschlossen.")
        self.open_detection_button.setVisible(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveAnnotationApp()
    window.show()
    sys.exit(app.exec())
