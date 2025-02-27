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

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog,
    QApplication, QProgressBar, QSlider, QPlainTextEdit
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

# Globaler Logger konfigurieren (Ausgabe auf Konsole; später wird FileHandler hinzugefügt)
logger = logging.getLogger("LiveAnnotation")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class AnnotationWorker(QThread):
    mosaic_updated = pyqtSignal(QPixmap)
    progress_updated = pyqtSignal(int)
    summary_signal = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, model_path, image_list, misannotated_dir, threshold, tile_size=200):
        super().__init__()
        self.model_path = model_path
        self.image_list = image_list
        self.misannotated_dir = misannotated_dir
        self.threshold = threshold  # z. B. 0.25
        self.tile_size = tile_size

    def run(self):
        # Im Worker-Thread werden alle nötigen Module importiert
        import cv2, numpy as np, os
        from ultralytics import YOLO
        from PyQt6.QtGui import QImage, QPixmap
        from collections import Counter

        # Lade das YOLO-Modell (Originalauflösung)
        model = YOLO(self.model_path)
        total_images = len(self.image_list)
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
                results = model.predict(source=valid_images, conf=self.threshold, show=False, verbose=False)
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
            progress = int(min(100, (current_index + batch_size) / total_images * 100))
            self.progress_updated.emit(progress)
            current_index += batch_size
        
        summary = (f"Live Annotation abgeschlossen.\n"
                   f"Gesamtbilder: {total_images}\n"
                   f"Korrekt annotiert: {good_count}\n"
                   f"Falsch annotiert: {bad_count}\n"
                   f"Falsch annotierte Bilder im Ordner: {self.misannotated_dir}")

        self.summary_signal.emit(summary)
        self.finished.emit()


class LiveAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Annotation")
        self.setGeometry(100, 100, 950, 950)
        layout = QVBoxLayout(self)
        
        # Auswahl YOLO-Modell (best.pt)
        model_layout = QHBoxLayout()
        self.model_line_edit = QLineEdit()
        self.model_button = QPushButton("Modell auswählen")
        self.model_button.clicked.connect(self.browse_model)
        model_layout.addWidget(QLabel("YOLO Modell (best.pt):"))
        model_layout.addWidget(self.model_line_edit)
        model_layout.addWidget(self.model_button)
        layout.addLayout(model_layout)
        
        # Auswahl Testdatenverzeichnis
        folder_layout = QHBoxLayout()
        self.folder_line_edit = QLineEdit()
        self.folder_button = QPushButton("Testordner auswählen")
        self.folder_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(QLabel("Testdatenverzeichnis:"))
        folder_layout.addWidget(self.folder_line_edit)
        folder_layout.addWidget(self.folder_button)
        layout.addLayout(folder_layout)
        
        # Threshold-Schieberegler
        threshold_layout = QHBoxLayout()
        self.threshold_label = QLabel("Threshold:")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(25)  # Standard = 0.25
        self.threshold_value_label = QLabel("0.25")
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        layout.addLayout(threshold_layout)
        
        # Start-Button
        self.start_button = QPushButton("Live Annotation starten")
        self.start_button.clicked.connect(self.start_live_annotation)
        layout.addWidget(self.start_button)
        
        # Fortschrittsbalken
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Anzeige des Mosaiks
        self.mosaic_label = QLabel()
        self.mosaic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mosaic_label)
        
        # Zusammenfassungsausgabefeld (wird nach Beendigung angezeigt)
        self.summary_output = QPlainTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.hide()
        layout.addWidget(self.summary_output)
        
        self.worker = None
        self.image_list = []

    def update_threshold_label(self, value):
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Modell auswählen", "", "PT Dateien (*.pt)")
        if file_path:
            self.model_line_edit.setText(file_path)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Testordner auswählen")
        if folder:
            self.folder_line_edit.setText(folder)

    def start_live_annotation(self):
        model_path = self.model_line_edit.text().strip()
        test_folder = self.folder_line_edit.text().strip()
        if not os.path.exists(model_path):
            return
        if not os.path.isdir(test_folder):
            return

        # Alle Bilder im Testordner sammeln
        self.image_list = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.image_list.extend(glob.glob(os.path.join(test_folder, ext)))
        self.image_list.sort()
        if not self.image_list:
            return

        # Erstelle ein Verzeichnis für falsch annotierte Bilder mit Zeitstempel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        misannotated_dir = os.path.join(test_folder, f"misannotated_{timestamp}")
        os.makedirs(misannotated_dir, exist_ok=True)
        logger.info("Misannotated-Verzeichnis: " + misannotated_dir)
        
        # Füge einen FileHandler zum Logger hinzu, um in eine Logdatei zu schreiben
        log_file = os.path.join(misannotated_dir, "live_annotation.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Hole den Threshold-Wert aus dem Schieberegler
        threshold = self.threshold_slider.value() / 100.0
        
        # Starte den Worker-Thread (ungebremst)
        self.worker = AnnotationWorker(model_path, self.image_list, misannotated_dir, threshold, tile_size=200)
        self.worker.mosaic_updated.connect(self.update_mosaic)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.summary_signal.connect(self.show_summary)
        self.worker.finished.connect(self.annotation_finished)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.summary_output.hide()

    def update_mosaic(self, pixmap):
        self.mosaic_label.setPixmap(pixmap)

    def show_summary(self, summary_text):
        self.summary_output.setPlainText(summary_text)
        self.summary_output.show()

    def annotation_finished(self):
        self.start_button.setEnabled(True)
        logger.info("Live Annotation abgeschlossen.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveAnnotationApp()
    window.show()
    sys.exit(app.exec())
