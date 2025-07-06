import sys
import os
import glob
from pathlib import Path
import platform
import subprocess
from typing import List, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QTextEdit, QScrollArea, 
    QFileDialog, QMessageBox, QProgressDialog, QFrame, QGridLayout,
    QSplitter, QGroupBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont

from PIL import Image, ImageDraw
import io
import warnings
from project_manager import WorkflowStep


# PIL Sicherheitseinstellungen
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", ".*DecompressionBombWarning.*")

# Konstanten
THUMBNAIL_SIZE = (150, 150)
DETAIL_SIZE = (800, 800)
IMAGES_PER_PAGE = 12  # 4x3 Grid


class SafeImageLoader:
    """Sichere und effiziente Bildlade-Klasse"""
    
    @staticmethod
    def load_thumbnail_safe(img_path: str, size=THUMBNAIL_SIZE):
        """Lädt sicheres Thumbnail"""
        try:
            with Image.open(img_path) as img:
                # Große Bilder erst verkleinern
                if img.size[0] > 2048 or img.size[1] > 2048:
                    img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                
                img = img.convert('RGB')
                img.thumbnail(size, Image.Resampling.LANCZOS)
                return img.copy()
        except Exception:
            # Fallback-Thumbnail
            thumb = Image.new('RGB', size, color='gray')
            draw = ImageDraw.Draw(thumb)
            draw.text((10, 10), "Error", fill='white')
            return thumb
    
    @staticmethod
    def load_detail_image_safe(img_path: str, size=DETAIL_SIZE):
        """Lädt sicheres Detail-Bild"""
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                
                # Größe begrenzen
                if img.size[0] > size[0] or img.size[1] > size[1]:
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                
                return img.copy()
        except Exception as e:
            error_img = Image.new('RGB', (800, 600), color='red')
            draw = ImageDraw.Draw(error_img)
            draw.text((10, 10), f"Fehler:\n{str(e)}", fill='white')
            return error_img


class QualityAnalyzer:
    """Schnelle Qualitätsanalyse ohne Threading"""
    
    def __init__(self, quality_thresholds: Dict):
        self.quality_thresholds = quality_thresholds
    
    def check_image_quality(self, img_path: str) -> List[str]:
        """Prüft die Qualität einer einzelnen Annotation"""
        issues = []
        
        label_path = os.path.splitext(img_path)[0] + '.txt'
        if not os.path.exists(label_path):
            # Kein Issue - wird als Background markiert
            return []
        
        try:
            # Schnelle Bildgröße ohne vollständiges Laden
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                # Kein Issue - wird als Background markiert
                return []
            
            # Prüfen ob nur Leerzeilen vorhanden sind
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            if not non_empty_lines:
                # Kein Issue - wird als Background markiert
                return []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    
                    if len(parts) < 5:
                        issues.append(f"❌ Zeile {line_num}: Unvollständig")
                        continue
                    
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    
                    # Schnelle Basis-Checks
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        issues.append(f"🚫 Zeile {line_num}: Koordinaten ungültig")
                    
                    # Größe-Checks
                    if width * height < self.quality_thresholds['min_box_size']:
                        issues.append(f"📏 Zeile {line_num}: Box zu klein")
                    
                    if min(width, height) > 0:
                        aspect_ratio = max(width/height, height/width)
                        if aspect_ratio > self.quality_thresholds['aspect_ratio_extreme']:
                            issues.append(f"⚡ Zeile {line_num}: Extremes Verhältnis")
                
                except ValueError:
                    issues.append(f"💥 Zeile {line_num}: Ungültige Werte")
        
        except Exception as e:
            issues.append(f"🔥 Fehler: {str(e)[:50]}")
        
        return issues


class ThumbnailWidget(QLabel):
    """Einfaches Thumbnail Widget"""
    
    def __init__(self, img_path: str, has_issues: bool, main_window, parent=None):
        super().__init__(parent)
        self.img_path = img_path
        self.has_issues = has_issues
        self.main_window = main_window
        
        self.setFixedSize(160, 160)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(self.get_style())
        self.setText("⏳ Lädt...")
        
        # Sofort laden (nur aktuelle Seite)
        self.load_thumbnail()
    
    def get_style(self):
        border_color = "#dc3545" if self.has_issues else "#28a745"
        return f"""
            QLabel {{
                border: 3px solid {border_color};
                border-radius: 8px;
                background-color: #ffffff;
                margin: 4px;
                color: #495057;
                font-size: 12px;
                font-weight: bold;
            }}
            QLabel:hover {{
                border-color: #007bff;
                background-color: #f0f8ff;
            }}
        """
    
    def load_thumbnail(self):
        """Lädt Thumbnail synchron"""
        try:
            thumbnail_img = SafeImageLoader.load_thumbnail_safe(self.img_path)
            
            # PIL zu QPixmap
            byte_array = io.BytesIO()
            thumbnail_img.save(byte_array, format='PNG')
            
            pixmap = QPixmap()
            if pixmap.loadFromData(byte_array.getvalue()):
                self.setPixmap(pixmap)
            else:
                self.setText("❌ Error")
        
        except Exception:
            self.setText("❌ Error")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.main_window.select_image(self.img_path)


class FastYOLOChecker(QMainWindow):
    def __init__(self):
        super().__init__()

        self.project_manager = None
        
        # Daten
        self.image_files = []
        self.current_image_index = 0
        self.dataset_path = ""
        self.quality_issues = {}
        self.background_images = set()  # Neue Kategorie für Background-Bilder
        self.deleted_files = []
        
        # Paginierung
        self.current_page = 1
        self.images_per_page = IMAGES_PER_PAGE
        self.total_pages = 0
        
        # Cache nur für aktuelle Seite
        self.current_page_cache = {}
        self.detail_image_cache = None
        self.current_detail_path = None
        
        # Qualitätsprüfer
        self.quality_analyzer = QualityAnalyzer({
            'min_box_size': 0.01,
            'edge_threshold': 0.02,
            'min_width_height': 0.005,
            'aspect_ratio_extreme': 20
        })
        
        self.setup_ui()
        self.apply_theme()
    
    def setup_ui(self):
        self.setWindowTitle("🚀 YOLO Quality Checker - Lightning Fast")
        self.setGeometry(100, 100, 1600, 1000)
        self.showMaximized()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 12, 12, 12)
        
        # Header - exakt 40px
        self.setup_header(main_layout)
        
        # Hauptinhalt
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        self.setup_gallery(splitter)
        self.setup_detail_view(splitter)
        
        splitter.setSizes([700, 900])
        
        self.statusBar().showMessage("🚀 Bereit - Dataset-Ordner wählen")
    
    def setup_header(self, parent_layout):
        """Header mit exakt 40px"""
        header_frame = QFrame()
        header_frame.setFixedHeight(40)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #343a40;
                border-radius: 6px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 6, 12, 6)
        header_layout.setSpacing(12)
        
        # Dataset Button
        self.folder_btn = QPushButton("📁 Dataset wählen")
        self.folder_btn.clicked.connect(self.select_dataset_folder)
        self.folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
                min-width: 120px;
            }
            QPushButton:hover { background-color: #0056b3; }
        """)
        header_layout.addWidget(self.folder_btn)
        
        header_layout.addStretch()
        
        # Filter
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet("QLabel { color: white; font-weight: bold; }")
        header_layout.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Alle", "Kritisch", "OK", "Background"])
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        self.filter_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: #212529;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                min-width: 80px;
                font-weight: bold;
            }
            QComboBox:hover {
                border-color: #007bff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border: 2px solid #212529;
                width: 0;
                height: 0;
                border-top: 4px solid #212529;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #212529;
                selection-background-color: #007bff;
                selection-color: white;
                border: 1px solid #ced4da;
                font-weight: bold;
                font-size: 11px;
                padding: 2px;
            }
        """)
        header_layout.addWidget(self.filter_combo)
        
        # Statistik
        self.stats_label = QLabel("Keine Daten")
        self.stats_label.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 4px;
                font-size: 11px;
            }
        """)
        header_layout.addWidget(self.stats_label)

        # Next button to open splitter
        self.next_button = QPushButton("Weiter zum Splitter")
        self.next_button.clicked.connect(self.open_splitter_app)
        self.next_button.setStyleSheet(
            "QPushButton {background-color: #28a745; color: white; border: none; padding: 6px 12px;"
            "border-radius: 4px; font-weight: bold; font-size: 11px;}"
            "QPushButton:hover {background-color: #218838;}"
        )
        header_layout.addWidget(self.next_button)

        parent_layout.addWidget(header_frame)
    
    def setup_gallery(self, parent):
        gallery_group = QGroupBox("🖼️ Galerie")
        gallery_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #495057;
                padding-top: 15px;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        
        gallery_layout = QVBoxLayout(gallery_group)
        
        # Paginierung oben
        self.setup_pagination_top(gallery_layout)
        
        # Galerie-Grid
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setSpacing(8)
        gallery_layout.addWidget(self.gallery_widget)
        
        # Paginierung unten
        self.setup_pagination_bottom(gallery_layout)
        
        parent.addWidget(gallery_group)
    
    def setup_pagination_top(self, parent_layout):
        """Paginierung oben"""
        page_frame = QFrame()
        page_layout = QHBoxLayout(page_frame)
        page_layout.setContentsMargins(5, 5, 5, 5)
        
        self.page_info_label = QLabel("Seite 0 von 0")
        self.page_info_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #495057;
                font-size: 12px;
            }
        """)
        page_layout.addWidget(self.page_info_label)
        
        page_layout.addStretch()
        
        # Schnell-Navigation
        self.first_page_btn = QPushButton("⏮️")
        self.first_page_btn.clicked.connect(lambda: self.go_to_page(1))
        self.first_page_btn.setFixedSize(32, 24)
        page_layout.addWidget(self.first_page_btn)
        
        self.prev_page_btn = QPushButton("◀️")
        self.prev_page_btn.clicked.connect(self.previous_page)
        self.prev_page_btn.setFixedSize(32, 24)
        page_layout.addWidget(self.prev_page_btn)
        
        self.next_page_btn = QPushButton("▶️")
        self.next_page_btn.clicked.connect(self.next_page)
        self.next_page_btn.setFixedSize(32, 24)
        page_layout.addWidget(self.next_page_btn)
        
        self.last_page_btn = QPushButton("⏭️")
        self.last_page_btn.clicked.connect(lambda: self.go_to_page(self.total_pages))
        self.last_page_btn.setFixedSize(32, 24)
        page_layout.addWidget(self.last_page_btn)
        
        parent_layout.addWidget(page_frame)
    
    def setup_pagination_bottom(self, parent_layout):
        """Paginierung unten"""
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(5, 5, 5, 5)
        
        self.load_time_label = QLabel("Ladezeit: 0s")
        self.load_time_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 10px;
                font-style: italic;
            }
        """)
        bottom_layout.addWidget(self.load_time_label)
        
        bottom_layout.addStretch()
        
        parent_layout.addWidget(bottom_frame)
    
    def setup_detail_view(self, parent):
        detail_group = QGroupBox("🔍 Detailansicht")
        detail_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #495057;
                padding-top: 15px;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        
        detail_layout = QVBoxLayout(detail_group)
        
        # Bildanzeige
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(400)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #adb5bd;
                border-radius: 8px;
                background-color: #ffffff;
                color: #6c757d;
                font-size: 16px;
            }
        """)
        self.image_label.setText("🖼️ Kein Bild ausgewählt")
        detail_layout.addWidget(self.image_label)
        
        # Navigation
        nav_layout = QHBoxLayout()
        
        self.prev_img_btn = QPushButton("⬅️ Vorheriges Bild")
        self.prev_img_btn.clicked.connect(self.previous_image)
        self.prev_img_btn.setStyleSheet(self.get_button_style("#6c757d"))
        nav_layout.addWidget(self.prev_img_btn)
        
        self.image_info_label = QLabel("Kein Bild")
        self.image_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_info_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #495057;
                padding: 8px;
                background-color: #e9ecef;
                border-radius: 6px;
            }
        """)
        nav_layout.addWidget(self.image_info_label)
        
        self.next_img_btn = QPushButton("Nächstes Bild ➡️")
        self.next_img_btn.clicked.connect(self.next_image)
        self.next_img_btn.setStyleSheet(self.get_button_style("#6c757d"))
        nav_layout.addWidget(self.next_img_btn)
        
        detail_layout.addLayout(nav_layout)
        
        # Qualitätsinformationen
        quality_group = QGroupBox("⚠️ Qualitätsprobleme")
        quality_layout = QVBoxLayout(quality_group)
        
        self.quality_text = QTextEdit()
        self.quality_text.setMaximumHeight(120)
        self.quality_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
                background-color: #ffffff;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        quality_layout.addWidget(self.quality_text)
        detail_layout.addWidget(quality_group)
        
        # Aktionen
        action_layout = QHBoxLayout()
        
        self.delete_btn = QPushButton("🗑️ Löschen")
        self.delete_btn.clicked.connect(self.delete_current_pair)
        self.delete_btn.setStyleSheet(self.get_button_style("#dc3545"))
        action_layout.addWidget(self.delete_btn)
        
        action_layout.addStretch()
        
        self.explorer_btn = QPushButton("📁 Explorer")
        self.explorer_btn.clicked.connect(self.open_in_explorer)
        self.explorer_btn.setStyleSheet(self.get_button_style("#28a745"))
        action_layout.addWidget(self.explorer_btn)
        
        detail_layout.addLayout(action_layout)
        
        parent.addWidget(detail_group)
    
    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                min-width: 100px;
            }}
            QPushButton:hover {{ background-color: {color}dd; }}
            QPushButton:pressed {{ background-color: {color}aa; }}
            QPushButton:disabled {{ 
                background-color: #adb5bd; 
                color: #6c757d; 
            }}
        """
    
    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QWidget {
                background-color: #f8f9fa;
                color: #495057;
            }
        """)
    
    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, 
            "YOLO Dataset Ordner wählen",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.dataset_path = folder
            self.load_dataset()
    
    def load_dataset(self):
        """Dataset laden mit Progress-Dialog"""
        try:
            # Progress Dialog
            progress = QProgressDialog("Scanne Dataset...", "Abbrechen", 0, 100, self)
            progress.setWindowTitle("Dataset laden")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            # Bilder finden
            progress.setValue(25)
            QApplication.processEvents()
            
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            self.image_files = []
            
            for ext in image_extensions:
                self.image_files.extend(glob.glob(os.path.join(self.dataset_path, ext)))
                self.image_files.extend(glob.glob(os.path.join(self.dataset_path, ext.upper())))
            
            if not self.image_files:
                progress.close()
                QMessageBox.critical(self, "Fehler", "Keine Bilder gefunden!")
                return
            
            self.image_files.sort()
            progress.setValue(50)
            QApplication.processEvents()
            
            # Qualitätsanalyse (nur schnelle Checks)
            progress.setLabelText("Analysiere Qualität...")
            self.quality_issues = {}
            
            for i, img_path in enumerate(self.image_files):
                if progress.wasCanceled():
                    return
                
                # Erst prüfen ob Background-Bild
                label_path = os.path.splitext(img_path)[0] + '.txt'
                is_background = False
                
                if not os.path.exists(label_path):
                    is_background = True
                else:
                    try:
                        with open(label_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        non_empty_lines = [line.strip() for line in lines if line.strip()]
                        if not non_empty_lines:
                            is_background = True
                    except:
                        pass
                
                if is_background:
                    self.background_images.add(img_path)
                else:
                    # Nur echte Quality-Issues prüfen
                    issues = self.quality_analyzer.check_image_quality(img_path)
                    if issues:
                        self.quality_issues[img_path] = issues
                if i % 50 == 0:  # Update alle 50 Bilder
                    progress.setValue(50 + int((i / len(self.image_files)) * 40))
                    QApplication.processEvents()
            
            progress.setValue(100)
            progress.close()
            
            # Paginierung setup
            self.current_image_index = 0
            self.current_page = 1
            self.total_pages = (len(self.image_files) + self.images_per_page - 1) // self.images_per_page
            
            # UI aktualisieren
            self.load_current_page()
            self.update_stats()
            
            critical_count = len(self.quality_issues)
            self.statusBar().showMessage(f"✅ {len(self.image_files)} Bilder, {critical_count} kritische")
            
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden:\n{str(e)}")
    
    def load_current_page(self):
        """Lädt nur die aktuelle Seite - BLITZSCHNELL"""
        import time
        start_time = time.time()
        
        # Alte Widgets löschen
        self.clear_gallery()
        
        # Gefilterte Bilder für aktuelle Seite
        filtered_images = self.get_filtered_images()
        
        # Sicherheitscheck für leere Ergebnisse
        if not filtered_images:
            self.page_info_label.setText("Keine Bilder gefunden")
            self.load_time_label.setText("Ladezeit: 0.00s")
            self.update_pagination_controls()
            return
        
        start_idx = (self.current_page - 1) * self.images_per_page
        end_idx = start_idx + self.images_per_page
        page_images = filtered_images[start_idx:end_idx]
        
        # Grid 4x3
        cols = 4
        for i, img_path in enumerate(page_images):
            row = i // cols
            col = i % cols
            
            try:
                has_issues = img_path in self.quality_issues
                thumbnail = ThumbnailWidget(img_path, has_issues, self)
                self.gallery_layout.addWidget(thumbnail, row, col)
            except Exception as e:
                # Fehler-Platzhalter nur bei echten Fehlern
                error_label = QLabel("❌ Fehler")
                error_label.setFixedSize(160, 160)
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                error_label.setStyleSheet("""
                    QLabel {
                        border: 3px solid #dc3545;
                        border-radius: 8px;
                        background-color: #f8d7da;
                        color: #721c24;
                        font-weight: bold;
                    }
                """)
                self.gallery_layout.addWidget(error_label, row, col)
        
        # Paginierung aktualisieren
        self.update_pagination_controls()
        
        load_time = time.time() - start_time
        self.load_time_label.setText(f"Ladezeit: {load_time:.2f}s")
    
    def clear_gallery(self):
        """Galerie leeren"""
        while self.gallery_layout.count():
            child = self.gallery_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def update_pagination_controls(self):
        """Paginierungs-Controls aktualisieren"""
        filtered_images = self.get_filtered_images()
        
        if not filtered_images:
            filtered_total_pages = 0
            self.page_info_label.setText("Keine Bilder")
        else:
            filtered_total_pages = (len(filtered_images) + self.images_per_page - 1) // self.images_per_page
            self.page_info_label.setText(f"Seite {self.current_page} von {filtered_total_pages}")
        
        # Sicherstellen, dass aktuelle Seite gültig ist
        if self.current_page > filtered_total_pages and filtered_total_pages > 0:
            self.current_page = filtered_total_pages
        
        self.prev_page_btn.setEnabled(self.current_page > 1)
        self.first_page_btn.setEnabled(self.current_page > 1)
        self.next_page_btn.setEnabled(self.current_page < filtered_total_pages)
        self.last_page_btn.setEnabled(self.current_page < filtered_total_pages)
    
    def get_filtered_images(self):
        """Gefilterte Bildliste"""
        filter_text = self.filter_combo.currentText().lower()
        
        if filter_text == "kritisch":
            return [img for img in self.image_files if img in self.quality_issues]
        elif filter_text == "ok":
            return [img for img in self.image_files 
                   if img not in self.quality_issues and img not in self.background_images]
        elif filter_text == "background":
            return [img for img in self.image_files if img in self.background_images]
        else:
            return self.image_files
    
    def previous_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.load_current_page()
    
    def next_page(self):
        filtered_images = self.get_filtered_images()
        max_pages = (len(filtered_images) + self.images_per_page - 1) // self.images_per_page
        if self.current_page < max_pages:
            self.current_page += 1
            self.load_current_page()
    
    def go_to_page(self, page_num):
        filtered_images = self.get_filtered_images()
        max_pages = (len(filtered_images) + self.images_per_page - 1) // self.images_per_page
        if 1 <= page_num <= max_pages:
            self.current_page = page_num
            self.load_current_page()
    
    def select_image(self, img_path):
        """Bild auswählen"""
        try:
            self.current_image_index = self.image_files.index(img_path)
            self.show_loading_transition()
            self.update_detail_view()
        except ValueError:
            pass
    
    def show_loading_transition(self):
        """Zeigt kurz schwarzen Bildschirm für 0.1s"""
        # Schwarzes Bild anzeigen
        black_pixmap = QPixmap(800, 600)
        black_pixmap.fill(QColor(0, 0, 0))
        self.image_label.setPixmap(black_pixmap)
        
        # Nach 100ms das richtige Bild laden
        QTimer.singleShot(100, self.update_detail_view)
    
    def update_detail_view(self):
        """Detail-Ansicht aktualisieren"""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_image_index]
        
        # Cache prüfen
        if self.current_detail_path == img_path and self.detail_image_cache:
            self._update_info_only(img_path)
            return
        
        try:
            # Bild mit Annotationen laden
            annotated_img = self.load_image_with_annotations(img_path)
            
            # QPixmap erstellen
            byte_array = io.BytesIO()
            annotated_img.save(byte_array, format='PNG')
            
            pixmap = QPixmap()
            if pixmap.loadFromData(byte_array.getvalue()):
                scaled_pixmap = pixmap.scaled(
                    DETAIL_SIZE[0], DETAIL_SIZE[1], 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                
                # Cache speichern
                self.detail_image_cache = scaled_pixmap
                self.current_detail_path = img_path
            else:
                self.image_label.setText("❌ Fehler beim Laden")
            
            self._update_info_only(img_path)
            
        except Exception as e:
            self.image_label.setText(f"❌ Fehler:\n{str(e)}")
    
    def _update_info_only(self, img_path):
        """Nur Info-Labels aktualisieren"""
        filename = os.path.basename(img_path)
        info = f"📸 {self.current_image_index + 1}/{len(self.image_files)}: {filename}"
        self.image_info_label.setText(info)
        
        # Qualitätsinformationen
        self.quality_text.clear()
        if img_path in self.quality_issues:
            issues_text = "\n".join(self.quality_issues[img_path])
            self.quality_text.setPlainText(issues_text)
            self.quality_text.setStyleSheet("""
                QTextEdit {
                    background-color: #f8d7da;
                    border: 1px solid #dc3545;
                    color: #721c24;
                    font-family: 'Consolas', monospace;
                    font-size: 11px;
                }
            """)
        elif img_path in self.background_images:
            self.quality_text.setPlainText("🌄 Background-Bild (keine Annotationen)")
            self.quality_text.setStyleSheet("""
                QTextEdit {
                    background-color: #e2e3e5;
                    border: 1px solid #6c757d;
                    color: #383d41;
                    font-family: 'Consolas', monospace;
                    font-size: 11px;
                    font-style: italic;
                }
            """)
        else:
            self.quality_text.setPlainText("✅ Keine Qualitätsprobleme!")
            self.quality_text.setStyleSheet("""
                QTextEdit {
                    background-color: #d4edda;
                    border: 1px solid #28a745;
                    color: #155724;
                    font-family: 'Consolas', monospace;
                    font-size: 11px;
                }
            """)
    
    def load_image_with_annotations(self, img_path):
        """Lädt Bild mit Annotationen"""
        img = SafeImageLoader.load_detail_image_safe(img_path)
        draw = ImageDraw.Draw(img)
        
        label_path = os.path.splitext(img_path)[0] + '.txt'
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                img_width, img_height = img.size
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            
                            x_min = int((x_center - width/2) * img_width)
                            y_min = int((y_center - height/2) * img_height)
                            x_max = int((x_center + width/2) * img_width)
                            y_max = int((y_center + height/2) * img_height)
                            
                            color = '#dc3545' if img_path in self.quality_issues else '#28a745'
                            
                            # 6px dicke Bounding Box
                            for thickness in range(6):
                                draw.rectangle([x_min-thickness, y_min-thickness, 
                                              x_max+thickness, y_max+thickness], 
                                             outline=color)
                            
                            # Klassen-ID
                            draw.text((x_min, max(0, y_min-25)), f"Class: {int(class_id)}", 
                                    fill=color)
                    
                    except (ValueError, IndexError):
                        continue
            except Exception:
                pass
        
        return img
    
    def previous_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.detail_image_cache = None
            self.current_detail_path = None
            self.show_loading_transition()
    
    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.detail_image_cache = None
            self.current_detail_path = None
            self.show_loading_transition()
    
    def delete_current_pair(self):
        """Löscht aktuelles Bild-Paar"""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_image_index]
        label_path = os.path.splitext(img_path)[0] + '.txt'
        filename = os.path.basename(img_path)
        
        reply = QMessageBox.question(
            self, 
            "Löschen bestätigen",
            f"'{filename}' und Label-Datei löschen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # Dateien löschen
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            
            # Aus Listen entfernen
            self.deleted_files.append(img_path)
            self.image_files.remove(img_path)
            
            if img_path in self.quality_issues:
                del self.quality_issues[img_path]
            if img_path in self.background_images:
                self.background_images.discard(img_path)
            
            # Cache leeren
            self.detail_image_cache = None
            self.current_detail_path = None
            
            # Index anpassen
            if self.current_image_index >= len(self.image_files) and self.image_files:
                self.current_image_index = len(self.image_files) - 1
            
            # Paginierung neu berechnen
            filtered_images = self.get_filtered_images()
            new_total_pages = (len(filtered_images) + self.images_per_page - 1) // self.images_per_page
            
            # Sicherstellen, dass aktuelle Seite gültig bleibt
            if self.current_page > new_total_pages and new_total_pages > 0:
                self.current_page = new_total_pages
            
            # Seite neu laden
            self.load_current_page()
            
            # Detail-View nur aktualisieren wenn noch Bilder vorhanden
            if self.image_files:
                self.update_detail_view()
            else:
                self.image_label.setText("🖼️ Keine Bilder mehr vorhanden")
                self.image_info_label.setText("Keine Bilder")
                self.quality_text.clear()
            
            self.update_stats()
            
            self.statusBar().showMessage(f"🗑️ '{filename}' gelöscht")
            
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Löschen:\n{str(e)}")
    
    def open_in_explorer(self):
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_image_index]
        
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", "/select,", img_path])
            elif platform.system() == "Darwin":
                subprocess.run(["open", "-R", img_path])
            else:
                subprocess.run(["xdg-open", os.path.dirname(img_path)])
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Explorer-Fehler:\n{str(e)}")
    
    def apply_filter(self):
        """Filter anwenden"""
        self.current_page = 1  # Zurück zur ersten Seite
        self.load_current_page()
    
    def update_stats(self):
        if not self.image_files:
            self.stats_label.setText("📊 Keine Daten")
            return
        
        total = len(self.image_files)
        critical = len([img for img in self.image_files if img in self.quality_issues])
        background = len([img for img in self.image_files if img in self.background_images])
        ok = total - critical - background
        
        stats_text = f"📊 {total} | ⚠️ {critical} | ✅ {ok} | 🌄 {background}"
        if self.deleted_files:
            stats_text += f" | 🗑️ {len(self.deleted_files)}"
        
        self.stats_label.setText(stats_text)

    def open_splitter_app(self):
        """Öffnet den Dataset-Splitter und schließt den Label Checker."""
        try:
            from gui.dataset_splitter import DatasetSplitterApp
            app = DatasetSplitterApp()
            app.project_manager = getattr(self, 'project_manager', None)
            if app.project_manager:
                aug_dir = app.project_manager.get_augmented_dir()
                labeled_dir = app.project_manager.get_labeled_dir()
                aug_files = list(aug_dir.glob("*.jpg")) + list(aug_dir.glob("*.png"))
                source_dir = str(aug_dir) if aug_files else str(labeled_dir)
                app.source_path.setText(source_dir)
                app.output_path.setText(str(app.project_manager.get_split_dir()))

                if source_dir:
                    app.analyze_classes(source_dir)
                    classes = app.project_manager.get_classes()
                    for class_id, class_name in classes.items():
                        if class_id in app.class_inputs:
                            app.class_inputs[class_id].setText(class_name)

            app.show()
            if self.project_manager:
                self.project_manager.mark_step_completed(WorkflowStep.AUGMENTATION)
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Splitter konnte nicht geöffnet werden:\n{str(e)}")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.previous_image()
        elif event.key() == Qt.Key.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key.Key_Delete:
            self.delete_current_pair()
        elif event.key() == Qt.Key.Key_PageUp:
            self.previous_page()
        elif event.key() == Qt.Key.Key_PageDown:
            self.next_page()
        else:
            super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    
    app.setApplicationName("YOLO Quality Checker")
    app.setApplicationVersion("5.0 Lightning Fast")
    app.setOrganizationName("Michel Marty")
    
    window = FastYOLOChecker()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()