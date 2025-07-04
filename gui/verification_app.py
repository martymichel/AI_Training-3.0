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
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog,
    QProgressBar, QSlider, QPlainTextEdit, QMessageBox, QSpinBox, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Import der Module - verschiedene Varianten versuchen
OptimizeThresholdsWorker = None
AnnotationWorker = None
validation_functions_available = False

# Versuch 1: Relative Imports (wenn als Package verwendet)
try:
    from .verification_core import OptimizeThresholdsWorker, AnnotationWorker
    from .verification_utils import (
        validate_model_path, validate_test_folder, get_model_status,
        open_directory, format_summary, setup_logging
    )
    validation_functions_available = True
    print("Successfully imported modules with relative imports")
except ImportError:
    # Versuch 2: Absolute Imports (wenn direkt ausgeführt)
    try:
        from verification_core import OptimizeThresholdsWorker, AnnotationWorker
        from verification_utils import (
            validate_model_path, validate_test_folder, get_model_status,
            open_directory, format_summary, setup_logging
        )
        validation_functions_available = True
        print("Successfully imported modules with absolute imports")
    except ImportError:
        # Versuch 3: Inline Definition der benötigten Klassen
        print("Warning: Could not import verification modules. Using inline definitions.")
        
        # Hier definieren wir die Klassen inline falls Import fehlschlägt
        import cv2
        import numpy as np
        from ultralytics import YOLO
        from datetime import datetime
        from collections import Counter
        import torch
        from PyQt6.QtCore import QThread, pyqtSignal
        from PyQt6.QtGui import QImage, QPixmap
        
        class OptimizeThresholdsWorker(QThread):
            progress_updated = pyqtSignal(int)
            stage_updated = pyqtSignal(str)
            optimization_finished = pyqtSignal(dict)
            
            def __init__(self, model_path, image_list, step_size=5, log_file=None):
                super().__init__()
                self.model_path = model_path
                self.image_list = image_list
                self.step_size = step_size
                self.plot_dir = None
                
            def run(self):
                try:
                    model = YOLO(self.model_path)
                    
                    # Setup directories
                    model_dir = os.path.dirname(os.path.abspath(self.model_path))
                    parent_dir = os.path.dirname(model_dir)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.plot_dir = os.path.join(parent_dir, f"optimization_plots_{timestamp}")
                    os.makedirs(self.plot_dir, exist_ok=True)
                    
                    best_result = {'conf': 0.25, 'iou': 0.45, 'accuracy': 0.0}
                    
                    # Simple grid search
                    conf_range = np.arange(0.1, 1.0, self.step_size/100.0)
                    iou_range = np.arange(0.3, 0.8, self.step_size/100.0)
                    
                    total_combinations = len(conf_range) * len(iou_range)
                    current_combination = 0
                    
                    for conf in conf_range:
                        for iou in iou_range:
                            if self.isInterruptionRequested():
                                break
                                
                            good_count = 0
                            total_images = len(self.image_list)
                            
                            # Test this combination
                            for img_path in self.image_list:
                                if not os.path.exists(img_path):
                                    continue
                                    
                                # Load ground truth
                                gt_counter = Counter()
                                annot_file = os.path.splitext(img_path)[0] + ".txt"
                                if os.path.exists(annot_file):
                                    with open(annot_file, 'r') as f:
                                        for line in f:
                                            parts = line.strip().split()
                                            if len(parts) >= 5:
                                                cls = int(float(parts[0]))
                                                gt_counter[cls] += 1
                                
                                # Predict
                                pred_counter = Counter()
                                results = model.predict(source=img_path, conf=conf, iou=iou, show=False, verbose=False)
                                if results and len(results) > 0:
                                    result = results[0]
                                    if hasattr(result, "boxes") and result.boxes is not None:
                                        for box in result.boxes:
                                            cls_pred = int(box.cls[0].cpu().numpy())
                                            pred_counter[cls_pred] += 1
                                
                                if gt_counter == pred_counter:
                                    good_count += 1
                            
                            accuracy = (good_count / total_images) * 100 if total_images > 0 else 0
                            
                            if accuracy > best_result['accuracy']:
                                best_result = {'conf': conf, 'iou': iou, 'accuracy': accuracy}
                            
                            current_combination += 1
                            progress = int((current_combination / total_combinations) * 100)
                            self.progress_updated.emit(progress)
                            
                            if accuracy >= 99:
                                break
                        if best_result['accuracy'] >= 99:
                            break
                    
                    self.optimization_finished.emit(best_result)
                    
                except Exception as e:
                    logger.error(f"Error during optimization: {e}")
                    self.optimization_finished.emit({})
        
        class AnnotationWorker(QThread):
            mosaic_updated = pyqtSignal(QPixmap)
            progress_updated = pyqtSignal(int)
            summary_signal = pyqtSignal(str, float, str)
            finished = pyqtSignal()
            
            def __init__(self, model_path, image_list, misannotated_dir, threshold, iou_threshold, tile_size=200):
                super().__init__()
                self.model_path = model_path
                self.image_list = image_list
                self.misannotated_dir = misannotated_dir
                self.threshold = threshold
                self.iou_threshold = iou_threshold
                self.tile_size = tile_size
                self.mosaic_history = []
                self.current_mosaic_index = -1
                
            def run(self):
                try:
                    model = YOLO(self.model_path)
                    total_images = len(self.image_list)
                    
                    # Setup output directory
                    model_dir = os.path.dirname(os.path.abspath(self.model_path))
                    parent_dir = os.path.dirname(model_dir)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.misannotated_dir = os.path.join(parent_dir, f"misannotated_{timestamp}")
                    os.makedirs(self.misannotated_dir, exist_ok=True)
                    
                    current_index = 0
                    batch_size = 9
                    bad_count = 0
                    good_count = 0
                    false_index = 1
                    
                    while current_index < total_images:
                        if self.isInterruptionRequested():
                            break
                            
                        batch = self.image_list[current_index:current_index+batch_size]
                        if len(batch) < batch_size:
                            batch += [""] * (batch_size - len(batch))
                            
                        valid_images = [img for img in batch if img and os.path.exists(img)]
                        results = None
                        if valid_images:
                            results = model.predict(source=valid_images, conf=self.threshold, iou=self.iou_threshold, show=False, verbose=False)
                        
                        # Create mosaic
                        ts = self.tile_size
                        mosaic = np.zeros((3*ts, 3*ts, 3), dtype=np.uint8)
                        
                        for i, img_path in enumerate(batch):
                            if not img_path or not os.path.exists(img_path):
                                continue
                                
                            orig_img = cv2.imread(img_path)
                            if orig_img is None:
                                continue
                                
                            annotated_img = orig_img.copy()
                            h, w = annotated_img.shape[:2]
                            
                            # Load ground truth
                            gt_counter = Counter()
                            annot_file = os.path.splitext(img_path)[0] + ".txt"
                            if os.path.exists(annot_file):
                                with open(annot_file, 'r') as f:
                                    for line in f:
                                        parts = line.strip().split()
                                        if len(parts) >= 5:
                                            cls = int(float(parts[0]))
                                            gt_counter[cls] += 1
                                            x_center = float(parts[1]) * w
                                            y_center = float(parts[2]) * h
                                            bw = float(parts[3]) * w
                                            bh = float(parts[4]) * h
                                            x1 = int(x_center - bw/2)
                                            y1 = int(y_center - bh/2)
                                            x2 = int(x_center + bw/2)
                                            y2 = int(y_center + bh/2)
                                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.putText(annotated_img, str(cls), (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                            
                            # Load predictions
                            pred_counter = Counter()
                            if results is not None and img_path in valid_images:
                                idx = valid_images.index(img_path)
                                r = results[idx]
                                if hasattr(r, "boxes") and r.boxes is not None:
                                    for box in r.boxes:
                                        coords = box.xyxy[0].cpu().numpy().astype(int)
                                        cls_pred = int(box.cls[0].cpu().numpy())
                                        pred_counter[cls_pred] += 1
                                        cv2.rectangle(annotated_img, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
                                        cv2.putText(annotated_img, str(cls_pred), (coords[0], max(coords[1]-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                            
                            # Check if correct
                            if gt_counter != pred_counter:
                                bad_flag = True
                                bad_count += 1
                                ext = os.path.splitext(img_path)[1]
                                new_filename = f"false_img{false_index}{ext}"
                                false_index += 1
                                dest_file = os.path.join(self.misannotated_dir, new_filename)
                                cv2.imwrite(dest_file, orig_img)
                            else:
                                bad_flag = False
                                good_count += 1
                            
                            # Create display image
                            display_img = cv2.resize(annotated_img, (ts, ts))
                            if bad_flag:
                                overlay = display_img.copy()
                                overlay[:] = (0, 0, 255)
                                alpha = 0.25
                                cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
                            
                            row = i // 3
                            col = i % 3
                            mosaic[row*ts:(row+1)*ts, col*ts:(col+1)*ts] = display_img
                        
                        # Convert to QPixmap
                        rgb_mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
                        height, width, channel = rgb_mosaic.shape
                        bytesPerLine = 3 * width
                        qImg = QImage(rgb_mosaic.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qImg)
                        self.mosaic_updated.emit(pixmap)
                        self.mosaic_history.append(pixmap)
                        self.current_mosaic_index = len(self.mosaic_history) - 1
                        
                        progress = int(min(100, (current_index + batch_size) / total_images * 100))
                        self.progress_updated.emit(progress)
                        current_index += batch_size
                    
                    summary = (f"Live Annotation abgeschlossen.\n"
                              f"Gesamtbilder: {total_images}\n"
                              f"Korrekt annotiert: {good_count}\n"
                              f"Falsch annotiert: {bad_count}\n"
                              f"Falsch annotierte Bilder im Ordner: {self.misannotated_dir}")
                    
                    correct_percentage = (good_count / total_images) * 100 if total_images > 0 else 0
                    self.summary_signal.emit(summary, correct_percentage, self.misannotated_dir)
                    self.finished.emit()
                    
                except Exception as e:
                    logger.error(f"Error during annotation: {e}")
                    self.finished.emit()
        
        # Dummy validation functions
        def validate_model_path(path): return os.path.exists(path)
        def validate_test_folder(path): return os.path.isdir(path)
        def get_model_status(acc): return ("#4CAF50", "GUT") if acc >= 95 else ("#F44336", "SCHLECHT")
        def open_directory(path): os.startfile(path) if os.name == 'nt' else os.system(f'xdg-open "{path}"')
        def format_summary(total, good, bad, dir_path): return f"Total: {total}, Good: {good}, Bad: {bad}"
        def setup_logging(dir_path): pass

class LiveAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_worker = None
        self.optimize_worker = None
        self.worker = None
        self.image_list = []
        
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
        
        # Plot directory button
        self.open_plot_dir_button = QPushButton("Bilder des Tunings")
        self.open_plot_dir_button.setEnabled(False)
        self.open_plot_dir_button.clicked.connect(self.open_optimization_plot_dir)
        sidebar_layout.addWidget(self.open_plot_dir_button)
        
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
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                min-height: 30px;
                color: black;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 1px;
                margin: 0.5px;
            }
        """)
        
        # Add stretch to push everything to the top
        sidebar_layout.addStretch()
        
        # Anzeige des Mosaiks
        self.mosaic_label = QLabel()
        self.mosaic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mosaic_label.setText("Kein Mosaik verfügbar - Starten Sie die Annotation")
        self.mosaic_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9; padding: 20px;")
        
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
        self.andon_label.setText("Status wird nach der Annotation angezeigt")
        self.andon_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0; padding: 10px;")
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
        self.summary_output.setStyleSheet("font-size: 16px;")
        content_layout.addWidget(self.summary_output)
        
        # Add main content and sidebar to main layout
        main_layout.addWidget(content_widget, stretch=4)
        main_layout.addWidget(sidebar_container)

    def open_optimization_plot_dir(self):
        """Open directory containing optimization plots."""
        if self.optimize_worker and hasattr(self.optimize_worker, "plot_dir"):
            plot_dir = self.optimize_worker.plot_dir
            if os.path.isdir(plot_dir):
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(plot_dir)
                    else:  # Linux/Mac
                        os.system(f'xdg-open "{plot_dir}"')
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Fehler beim Öffnen des Verzeichnisses: {e}")

    def update_threshold_label(self, value):
        """Update threshold label when slider changes."""
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")

    def update_iou_label(self, value):
        """Update IoU label when slider changes."""
        iou = value / 100.0
        self.iou_value_label.setText(f"{iou:.2f}")

    def open_misannotated_dir(self, dir_path):
        """Open the directory containing misannotated images."""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(dir_path)
            else:  # Linux/Mac
                os.system(f'xdg-open "{dir_path}"')
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Fehler beim Öffnen des Verzeichnisses: {e}")

    def browse_model(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "KI-Modell (.pt) auswählen", 
            "", 
            "PT Dateien (*.pt);;Alle Dateien (*)"
        )
        if file_path:
            self.model_line_edit.setText(file_path)

    def browse_folder(self):
        """Browse for test dataset folder."""
        folder = QFileDialog.getExistingDirectory(self, "Test-Dataset auswählen")
        if folder:
            self.folder_line_edit.setText(folder)

    def validate_inputs(self):
        """Validate user inputs before starting processes."""
        model_path = self.model_line_edit.text().strip()
        test_folder = self.folder_line_edit.text().strip()
        
        if not model_path:
            QMessageBox.warning(self, "Error", "Bitte wählen Sie eine Modell-Datei aus")
            return False
            
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Die gewählte Modell-Datei existiert nicht")
            return False
            
        if not test_folder:
            QMessageBox.warning(self, "Error", "Bitte wählen Sie ein Testverzeichnis aus")
            return False
            
        if not os.path.isdir(test_folder):
            QMessageBox.warning(self, "Error", "Das gewählte Testverzeichnis existiert nicht")
            return False

        # Check for images in test folder
        self.image_list = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            self.image_list.extend(glob.glob(os.path.join(test_folder, ext)))
        
        if not self.image_list:
            QMessageBox.warning(self, "Error", "Keine Bilder im Testverzeichnis gefunden")
            return False
            
        self.image_list.sort()
        return True

    def optimize_thresholds(self):
        """Start threshold optimization process."""
        if not self.validate_inputs():
            return
            
        # Check if optimization module is available
        if OptimizeThresholdsWorker is None:
            QMessageBox.warning(self, "Error", "Threshold-Optimierung Modul nicht verfügbar. Stellen Sie sicher, dass verification_core.py im gleichen Verzeichnis liegt.")
            return

        model_path = self.model_line_edit.text().strip()
        
        # Disable UI during optimization
        self.optimize_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Start optimization worker
        self.optimize_worker = OptimizeThresholdsWorker(
            model_path=model_path,
            image_list=self.image_list,
            step_size=self.step_size_spin.value()
        )
        self.optimize_worker.progress_updated.connect(self.progress_bar.setValue)
        self.optimize_worker.stage_updated.connect(self.update_stage_display)
        self.optimize_worker.optimization_finished.connect(self.optimization_complete)
        self.optimize_worker.start()
        
        # Update status
        self.summary_output.setPlainText("Schwellenwert-Tuning wird gestartet...")
        self.summary_output.show()

    def update_stage_display(self, message):
        """Update display with current optimization stage."""
        self.summary_output.setPlainText(message)

    def optimization_complete(self, results):
        """Handle completion of threshold optimization."""
        self.optimize_button.setEnabled(True)
        self.start_button.setEnabled(True)
        
        if not results:
            QMessageBox.warning(self, "Error", "Schwellenwert-Tuning fehlgeschlagen")
            self.summary_output.setPlainText("Tuning fehlgeschlagen - bitte versuchen Sie es erneut.")
            return

        # Enable plot button
        self.open_plot_dir_button.setEnabled(True)    
        
        # Update slider positions
        conf_value = max(0, min(100, int(results['conf'] * 100)))
        iou_value = max(0, min(100, int(results['iou'] * 100)))

        self.threshold_slider.setValue(conf_value)
        self.iou_slider.setValue(iou_value)

        # Update labels
        self.threshold_value_label.setText(f"{conf_value/100:.2f}")
        self.iou_value_label.setText(f"{iou_value/100:.2f}")        

        summary = (
            f"Schwellenwert-Tuning abgeschlossen\n\n"
            f"Beste Konfiguration:\n"
            f"- Konfidenz Schwellenwert: {results['conf']:.2f}\n"
            f"- IoU Schwellenwert: {results['iou']:.2f}\n"
            f"- Genauigkeit: {results['accuracy']:.1f}%\n\n"
            f"Die Slider wurden automatisch auf diese Einstellung eingestellt.\n"
            f"Sie können jetzt auf 'Bilder-Annotation starten' klicken, um die Verifikation zu starten."
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
        """Start live annotation process."""
        if not self.validate_inputs():
            return
            
        # Check if annotation module is available
        if AnnotationWorker is None:
            QMessageBox.warning(self, "Error", "Annotation Modul nicht verfügbar. Stellen Sie sicher, dass verification_core.py im gleichen Verzeichnis liegt.")
            return

        model_path = self.model_line_edit.text().strip()
        
        # Get threshold values from sliders
        threshold = self.threshold_slider.value() / 100.0
        iou_threshold = self.iou_slider.value() / 100.0
        
        # Start worker thread
        self.worker = AnnotationWorker(
            model_path=model_path,
            image_list=self.image_list,
            misannotated_dir="",  # Will be created by worker
            threshold=threshold,
            iou_threshold=iou_threshold,
            tile_size=256
        )
        self.worker.mosaic_updated.connect(self.update_mosaic)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.summary_signal.connect(self.show_summary)
        self.worker.finished.connect(self.annotation_finished)
        self.worker.start()
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.summary_output.hide()
        self.andon_label.setText("Annotation läuft...")
        self.andon_label.setStyleSheet("border: 1px solid #ccc; background-color: #fff3cd; padding: 10px;")

    def update_mosaic(self, pixmap):
        """Update mosaic display."""
        self.mosaic_label.setPixmap(pixmap)
        
        # Update navigation buttons
        if self.worker and hasattr(self.worker, 'mosaic_history'):
            total_mosaics = len(self.worker.mosaic_history)
            
            # Show buttons only if more than 1 mosaic exists
            if total_mosaics > 1:
                self.prev_mosaic_btn.show()
                self.next_mosaic_btn.show()
            else:
                self.prev_mosaic_btn.hide()
                self.next_mosaic_btn.hide()
            
            # Enable/disable buttons based on current position
            if hasattr(self.worker, 'current_mosaic_index'):
                self.prev_mosaic_btn.setEnabled(self.worker.current_mosaic_index > 0)
                self.next_mosaic_btn.setEnabled(
                    self.worker.current_mosaic_index < total_mosaics - 1
                )

    def show_previous_mosaic(self):
        """Show the previous mosaic from history."""
        if (self.worker and 
            hasattr(self.worker, 'mosaic_history') and 
            hasattr(self.worker, 'current_mosaic_index') and
            self.worker.current_mosaic_index > 0):
            
            self.worker.current_mosaic_index -= 1
            self.mosaic_label.setPixmap(self.worker.mosaic_history[self.worker.current_mosaic_index])
            self.prev_mosaic_btn.setEnabled(self.worker.current_mosaic_index > 0)
            self.next_mosaic_btn.setEnabled(True)

    def show_next_mosaic(self):
        """Show the next mosaic from history."""
        if (self.worker and 
            hasattr(self.worker, 'mosaic_history') and 
            hasattr(self.worker, 'current_mosaic_index') and
            self.worker.current_mosaic_index < len(self.worker.mosaic_history) - 1):
            
            self.worker.current_mosaic_index += 1
            self.mosaic_label.setPixmap(self.worker.mosaic_history[self.worker.current_mosaic_index])
            self.next_mosaic_btn.setEnabled(
                self.worker.current_mosaic_index < len(self.worker.mosaic_history) - 1
            )
            self.prev_mosaic_btn.setEnabled(True)

    def show_summary(self, summary_text, correct_percentage, misannotated_dir):
        """Display final summary and results."""
        # Add FileHandler to logger now that we have the misannotated_dir
        try:
            log_file = os.path.join(misannotated_dir, "live_annotation.log")
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
        
        # Update summary text
        self.summary_output.setPlainText(summary_text)
        self.summary_output.show()
        
        # Update Andon display based on performance
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
            border-radius: 10px;
            padding: 10px;
            margin: 10px;
        """)
        self.andon_label.setText(f"Status: {status} ({correct_percentage:.1f}%)")
        
        # Update misannotated directory link (separate)
        if os.path.isdir(misannotated_dir):
            self.misannotated_link.setStyleSheet("""
                font-size: 14px;
                color: #2196F3;
                padding: 10px;
            """)
            self.misannotated_link.setText(
                f"<a href='{misannotated_dir}' style='color: #2196F3;'>"
                f"Klicken Sie hier um den Ordner mit falsch annotierten Bildern zu öffnen</a>"
            )
            self.misannotated_link.setOpenExternalLinks(False)
            self.misannotated_link.linkActivated.connect(self.open_misannotated_dir)
        else:
            self.misannotated_link.setText("Kein Ordner für falsch annotierte Bilder verfügbar")

    def annotation_finished(self):
        """Handle completion of annotation process."""
        self.start_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        logger.info("Test-Annotation & Modellverifikation abgeschlossen.")
        
        # Show completion message
        QMessageBox.information(
            self,
            "Annotation abgeschlossen",
            "Die Modellverifikation wurde erfolgreich abgeschlossen. "
            "Überprüfen Sie die Zusammenfassung für Details."
        )

    def closeEvent(self, event):
        """Handle application close event."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, 
                'Beenden', 
                'Annotation läuft noch. Wirklich beenden?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.terminate()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveAnnotationApp()
    window.show()
    sys.exit(app.exec())