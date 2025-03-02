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
import cv2
import numpy as np
from ultralytics import YOLO
import sys
from datetime import datetime
import logging
from collections import Counter
import torch

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog,
    QApplication, QProgressBar, QSlider, QPlainTextEdit, QMessageBox, QSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

# Globaler Logger konfigurieren (Ausgabe auf Konsole; später wird FileHandler hinzugefügt)
logger = logging.getLogger("LiveAnnotation")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class OptimizeThresholdsWorker(QThread):
    """Worker thread for threshold optimization."""
    progress_updated = pyqtSignal(int)
    stage_updated = pyqtSignal(str)
    optimization_finished = pyqtSignal(dict)
    
    def __init__(self, model_path, image_list, step_size=5):
        super().__init__()
        self.model_path = model_path
        self.image_list = image_list
        self.step_size = step_size
        self.total_progress = 0
        
    def run(self):
        try:
            model = YOLO(self.model_path)
            self.model = model  # Store model as instance variable
            best_result = {
                'conf': 0.25,  # Default values
                'iou': 0.45,
                'accuracy': 0.0
            }
            
            # Stage 1: Coarse search with 20% steps
            self.stage_updated.emit("Stage 1: Coarse search")
            coarse_step = 20
            coarse_best = self.search_grid(
                conf_range=(1, 100),
                iou_range=(1, 100),
                step=coarse_step,
                progress_weight=0.3,
                model=model
            )
            
            # Stage 2: Medium search around best coarse result
            self.stage_updated.emit("Stage 2: Medium search")
            conf_best = coarse_best['conf'] * 100
            iou_best = coarse_best['iou'] * 100
            medium_step = 10
            
            # Define search ranges around best coarse result
            conf_min = max(1, conf_best - coarse_step)
            conf_max = min(100, conf_best + coarse_step)
            iou_min = max(1, iou_best - coarse_step)
            iou_max = min(100, iou_best + coarse_step)
            
            medium_best = self.search_grid(
                conf_range=(conf_min, conf_max),
                iou_range=(iou_min, iou_max),
                step=medium_step,
                progress_weight=0.3,
                base_progress=30,
                model=model
            )
            
            # Stage 3: Fine search with user-defined step size
            self.stage_updated.emit("Stage 3: Fine search")
            conf_best = medium_best['conf'] * 100
            iou_best = medium_best['iou'] * 100
            
            # Define final search ranges
            conf_min = max(1, conf_best - medium_step)
            conf_max = min(100, conf_best + medium_step)
            iou_min = max(1, iou_best - medium_step)
            iou_max = min(100, iou_best + medium_step)
            
            best_result = self.search_grid(
                conf_range=(conf_min, conf_max),
                iou_range=(iou_min, iou_max),
                step=self.step_size,
                progress_weight=0.4,
                base_progress=60,
                model=model
            )
            
            self.optimization_finished.emit(best_result)
            
        except Exception as e:
            logger.error(f"Error during threshold optimization: {e}")
            self.optimization_finished.emit({})

    def search_grid(self, conf_range, iou_range, step, progress_weight=1.0, base_progress=0, model=None):
        """Search for optimal thresholds in a given range with specified step size."""
        if model is None:
            raise ValueError("Model must be provided")
            
        # Ensure ranges are within valid bounds (0-100)
        start_conf = max(1, min(100, float(conf_range[0])))
        end_conf = max(1, min(100, float(conf_range[1])))
        start_iou = max(1, min(100, float(iou_range[0])))
        end_iou = max(1, min(100, float(iou_range[1])))
        step = float(step)
        
        best_result = {
            'conf': 0.25,
            'iou': 0.45,
            'accuracy': 0.0
        }
        
        # Calculate steps ensuring we don't exceed bounds
        conf_steps = max(1, int((end_conf - start_conf) / step) + 1)
        iou_steps = max(1, int((end_iou - start_iou) / step) + 1)
        total_combinations = conf_steps * iou_steps
        current_combination = 0
        
        # Process images in larger batches for speed
        batch_size = min(16, len(self.image_list))
        
        # Use numpy for more efficient range generation
        conf_range = np.clip(np.arange(start_conf, end_conf + step/2, step), 1, 100)
        iou_range = np.clip(np.arange(start_iou, end_iou + step/2, step), 1, 100)
        
        for conf in conf_range:
            for iou in iou_range:
                # Ensure thresholds are within valid ranges
                conf_threshold = np.clip(conf / 100.0, 0.01, 0.99)
                iou_threshold = np.clip(iou / 100.0, 0.01, 0.99)
                
                good_count = 0
                total_images = len(self.image_list)
                
                # Process images in batches
                for i in range(0, total_images, batch_size):
                    batch = self.image_list[i:i+batch_size]
                    results = model.predict(
                        source=batch,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        show=False,
                        verbose=False,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    
                    # Compare predictions with ground truth
                    for img_path, result in zip(batch, results):
                        # Get ground truth
                        gt_counter = Counter()
                        annot_file = os.path.splitext(img_path)[0] + ".txt"
                        if os.path.exists(annot_file):
                            with open(annot_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        cls = int(float(parts[0]))
                                        gt_counter[cls] += 1
                        
                        # Get predictions
                        pred_counter = Counter()
                        if hasattr(result, "boxes") and result.boxes is not None:
                            for box in result.boxes:
                                cls_pred = int(box.cls[0].cpu().numpy())
                                pred_counter[cls_pred] += 1
                        
                        # Compare counters
                        if gt_counter == pred_counter:
                            good_count += 1
                
                # Calculate accuracy
                accuracy = (good_count / total_images) * 100
                if accuracy > best_result['accuracy']:
                    best_result = {
                            'conf': conf_threshold,
                            'iou': iou_threshold,
                            'accuracy': accuracy
                        }
                
                current_combination += 1
                progress = base_progress + int((current_combination / total_combinations) * 100 * progress_weight)
                self.progress_updated.emit(progress)
                
                # Early stopping if we find very good results
                if best_result['accuracy'] >= 98:
                    return best_result
                
                # Check if we should continue
                if self.isInterruptionRequested():
                    return best_result
        
        return best_result

class AnnotationWorker(QThread):
    mosaic_updated = pyqtSignal(QPixmap)
    progress_updated = pyqtSignal(int)
    summary_signal = pyqtSignal(str, float, str)  # text, percentage, misannotated_dir
    finished = pyqtSignal()
    
    def __init__(self, model_path, image_list, misannotated_dir, threshold, iou_threshold, tile_size=200):
        super().__init__()
        self.model_path = model_path
        self.image_list = image_list
        self.misannotated_dir = misannotated_dir
        self.threshold = threshold  # z. B. 0.25
        self.iou_threshold = iou_threshold
        self.tile_size = tile_size
        self.mosaic_history = []  # Store mosaic history
        self.current_mosaic_index = -1

    def run(self):
        # Im Worker-Thread werden alle nötigen Module importiert
        import cv2, numpy as np, os
        from ultralytics import YOLO
        from PyQt6.QtGui import QImage, QPixmap
        from collections import Counter

        # Lade das YOLO-Modell (Originalauflösung)
        model = YOLO(self.model_path)
        total_images = len(self.image_list)
        
        # Create misannotated directory one level up from model
        model_dir = os.path.dirname(os.path.abspath(self.model_path))
        parent_dir = os.path.dirname(model_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.misannotated_dir = os.path.join(parent_dir, f"misannotated_{timestamp}")
        os.makedirs(self.misannotated_dir, exist_ok=True)
        
        current_index = 0
        batch_size = 9
        bad_count = 0
        good_count = 0
        false_index = 1  # Zähler für falsch annotierte Bilder

        while current_index < total_images:
            batch = self.image_list[current_index:current_index+batch_size]
            if len(batch) < batch_size:
                batch += [""] * (batch_size - len(batch))
            valid_images = [img for img in batch if img]
            results = None
            if valid_images:
                results = model.predict(source=valid_images, conf=self.threshold, iou=self.iou_threshold, show=False, verbose=False)
            ts = self.tile_size
            mosaic = np.zeros((3*ts, 3*ts, 3), dtype=np.uint8)
            
            for i, img_path in enumerate(batch):
                row = i // 3
                col = i % 3
                if not img_path:
                    continue
                # Lese Originalbild (volle Auflösung)
                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    continue
                # Erstelle eine annotierte Kopie, in die sowohl GT als auch Vorhersagen gezeichnet werden
                annotated_img = orig_img.copy()
                h, w = annotated_img.shape[:2]
                
                # Ground‑Truth aus der Annotation (verwende Counter für Frequenzen)
                gt_counter = Counter()
                base = os.path.splitext(img_path)[0]
                annot_file = base + ".txt"
                if os.path.exists(annot_file):
                    with open(annot_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls = int(float(parts[0]))
                                gt_counter[cls] += 1
                                # Zeichne GT-Box in Originalauflösung
                                x_center = float(parts[1]) * w
                                y_center = float(parts[2]) * h
                                bw = float(parts[3]) * w
                                bh = float(parts[4]) * h
                                x1 = int(x_center - bw/2)
                                y1 = int(y_center - bh/2)
                                x2 = int(x_center + bw/2)
                                y2 = int(y_center + bh/2)
                                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_img, str(cls), (x1, max(y1-5, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                # Vorhersagen des Modells (verwende Counter für Frequenzen)
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
                            cv2.putText(annotated_img, str(cls_pred), (coords[0], max(coords[1]-5,0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                
                # Vergleiche die erwarteten und vorhergesagten Klassen (als Counter)
                if gt_counter != pred_counter:
                    bad_flag = True
                    bad_count += 1
                    # Berechne fehlende bzw. extra Klassen
                    missing = gt_counter - pred_counter
                    extra = pred_counter - gt_counter
                    # Neuer Dateiname: false_imgX.ext
                    ext = os.path.splitext(img_path)[1]
                    new_filename = f"false_img{false_index}{ext}"
                    false_index += 1
                    dest_file = os.path.join(self.misannotated_dir, new_filename)
                    # Speichere das annotierte Originalbild ohne rotes Overlay
                    cv2.imwrite(dest_file, annotated_img)
                    # Erstelle einen Log-Eintrag
                    log_line = (f"{new_filename}: Erwartet: {dict(gt_counter)}, Vorhergesagt: {dict(pred_counter)}. "
                                f"Fehlend: {dict(missing)}, Überschuss: {dict(extra)}")
                    logger.info("Falsch annotiert: " + log_line)
                else:
                    bad_flag = False
                    good_count += 1
                
                # Für das Mosaik: Erstelle eine verkleinerte Version
                display_img = cv2.resize(annotated_img, (ts, ts))
                if bad_flag:
                    # Rotes Overlay mit Alpha 0.25 für die Anzeige
                    overlay = display_img.copy()
                    overlay[:] = (0, 0, 255)
                    alpha = 0.25
                    cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
                mosaic[row*ts:(row+1)*ts, col*ts:(col+1)*ts] = display_img
            
            # Konvertiere das Mosaik zur Anzeige in Qt
            rgb_mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_mosaic.shape
            bytesPerLine = 3 * width
            qImg = QImage(rgb_mosaic.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.mosaic_updated.emit(pixmap)
            # Store mosaic in history
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
        
        # Calculate percentage of correctly annotated images
        correct_percentage = (good_count / total_images) * 100

        self.summary_signal.emit(summary, correct_percentage, self.misannotated_dir)
        self.finished.emit()


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
        sidebar_layout.addWidget(QLabel("Threshold:"))
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
        self.step_size_spin.setValue(5)
        self.step_size_spin.setSuffix("%")
        step_layout.addWidget(self.step_size_spin)
        optimize_layout.addLayout(step_layout)
        
        # Optimize button
        self.optimize_button = QPushButton("Optimize Thresholds")
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
        self.start_button = QPushButton("Bilder-Annotation starten")
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
                color: black;
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
        content_layout.addWidget(self.summary_output)
        
        # Add main content and sidebar to main layout
        main_layout.addWidget(content_widget, stretch=4)
        main_layout.addWidget(sidebar_container)
        
        self.worker = None
        self.image_list = []
        self.optimize_worker = None

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Modell auswählen", "", "PT Dateien (*.pt)")
        if file_path:
            self.model_line_edit.setText(file_path)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Testordner auswählen")
        if folder:
            self.folder_line_edit.setText(folder)

    def optimize_thresholds(self):
        """Start threshold optimization process."""
        model_path = self.model_line_edit.text().strip()
        test_folder = self.folder_line_edit.text().strip()
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Please select a valid model file")
            return
            
        if not os.path.isdir(test_folder):
            QMessageBox.warning(self, "Error", "Please select a valid test folder")
            return

        # Collect all images in test folder
        self.image_list = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.image_list.extend(glob.glob(os.path.join(test_folder, ext)))
        self.image_list.sort()
        if not self.image_list:
            QMessageBox.warning(self, "Error", "No images found in test folder")
            return
        
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
        self.optimize_worker.stage_updated.connect(lambda msg: self.summary_output.setPlainText(msg))
        self.optimize_worker.optimization_finished.connect(self.optimization_complete)
        self.optimize_worker.start()
        
        # Update status
        self.summary_output.setPlainText("Starting optimization...")
        self.summary_output.show()

    def optimization_complete(self, results):
        """Handle completion of threshold optimization."""
        self.optimize_button.setEnabled(True)
        self.start_button.setEnabled(True)
        
        if not results:
            QMessageBox.warning(self, "Error", "Threshold optimization failed")
            return
        
        # Update threshold sliders with optimal values
        conf_value = int(results['conf'] * 100)
        iou_value = int(results['iou'] * 100)
        self.threshold_slider.setValue(conf_value)
        self.iou_slider.setValue(iou_value)
        
        # Show optimization results
        summary = (
            f"Threshold Optimization Complete\n\n"
            f"Best Configuration:\n"
            f"- Confidence Threshold: {results['conf']:.2f}\n"
            f"- IoU Threshold: {results['iou']:.2f}\n"
            f"- Accuracy: {results['accuracy']:.1f}%\n\n"
            f"The sliders have been automatically set to these optimal values.\n"
            f"You can now click 'Bilder-Annotation starten' to run the verification\n"
            f"with these optimized settings."
        )
        self.summary_output.setPlainText(summary)
        
        QMessageBox.information(
            self,
            "Optimization Complete",
            f"Optimal thresholds found:\nConfidence: {results['conf']:.2f}\nIoU: {results['iou']:.2f}\n"
            f"Expected accuracy: {results['accuracy']:.1f}%"
        )

    def start_live_annotation(self):
        model_path = self.model_line_edit.text().strip()
        test_folder = self.folder_line_edit.text().strip()
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Please select a valid model file")
            return
            
        if not os.path.isdir(test_folder):
            QMessageBox.warning(self, "Error", "Please select a valid test folder")
            return

        # Collect all images in test folder
        self.image_list = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.image_list.extend(glob.glob(os.path.join(test_folder, ext)))
        self.image_list.sort()
        if not self.image_list:
            QMessageBox.warning(self, "Error", "No images found in test folder")
            return
        
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
            tile_size=200
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
            self.prev_mosaic_btn.setEnabled(self.worker.current_mosaic_index > 0)
            self.next_mosaic_btn.setEnabled(
                self.worker.current_mosaic_index < len(self.worker.mosaic_history) - 1
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
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Update summary text
        self.summary_output.setPlainText(summary_text)
        self.summary_output.show()
        
        # Update Andon display
        if correct_percentage >= 98:
            color = "#4CAF50"  # Green
            status = "MODELL SEHR GUT"
        elif correct_percentage >= 95:
            color = "#FF9800"  # Orange
            status = "MODELL AKZEPTABEL"
        else:
            color = "#F44336"  # Red
            status = "MODELL UNGENÜGEND"
            
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

    def annotation_finished(self):
        self.start_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        logger.info("Live Annotation abgeschlossen.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveAnnotationApp()
    window.show()
    sys.exit(app.exec())
