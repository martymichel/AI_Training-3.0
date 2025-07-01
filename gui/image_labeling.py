# image_labeling.py
import sys, os, shutil
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QMessageBox, QInputDialog,
    QScrollArea
)
from PyQt6.QtGui import QPixmap, QPen, QColor, QFont, QBrush, QPainter
from PyQt6.QtCore import Qt, QRectF, QPointF, QSizeF, QTimer, QPoint

import utils.labeling_utils as utils
from project_manager import ProjectManager, WorkflowStep

# -------------------------------
# Zoomable Graphics View
# -------------------------------
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.zoom_factor = 1.15
        
    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Store the scene pos
            old_pos = self.mapToScene(event.position().toPoint())
            
            # Zoom
            if event.angleDelta().y() > 0:
                factor = self.zoom_factor
            else:
                factor = 1.0 / self.zoom_factor
            self.scale(factor, factor)
            
            # Get the new position
            new_pos = self.mapToScene(event.position().toPoint())
            
            # Move scene to old position
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
        else:
            super().wheelEvent(event)

# -------------------------------
# Resizable Bounding Box Item
# -------------------------------
class ResizableRectItem(QGraphicsRectItem):
    def __init__(self, rect, class_id, parent=None):
        super().__init__(rect, parent)
        self.scene_ref = None  # Reference to scene for updating annotations
        self.class_id = class_id
        self.handles = []
        self.handle_size = QSizeF(20.0, 20.0)  # Larger handles for easier grabbing
        self.setFlags(
            QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.updateHandles()

    def updateHandles(self):
        """Update the resize handles."""
        self.handles.clear()
        rect = self.rect()
        self.handles = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight()
        ]

    def paint(self, painter, option, widget):
        """Paint the rectangle and its handles."""
        super().paint(painter, option, widget)
        
        # Draw resize handles
        if self.isSelected():
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            for handle in self.handles:
                painter.drawRect(QRectF(
                    handle.x() - self.handle_size.width() / 2,
                    handle.y() - self.handle_size.height() / 2,
                    self.handle_size.width(),
                    self.handle_size.height()
                ))

    def mousePressEvent(self, event):
        """Handle mouse press events for resizing."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            # Check if we clicked on a handle
            for i, handle in enumerate(self.handles):
                if (pos - handle).manhattanLength() < 10:
                    self.setSelected(True)
                    self.resize_handle = i
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for resizing."""
        if hasattr(self, 'resize_handle'):
            pos = event.pos()
            rect = self.rect()
            if self.resize_handle == 0:  # Top-left
                rect.setTopLeft(pos)
            elif self.resize_handle == 1:  # Top-right
                rect.setTopRight(pos)
            elif self.resize_handle == 2:  # Bottom-left
                rect.setBottomLeft(pos)
            elif self.resize_handle == 3:  # Bottom-right
                rect.setBottomRight(pos)
            self.setRect(rect.normalized())
            self.updateHandles()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if hasattr(self, 'resize_handle'):
            # Store position after resize
            del self.resize_handle
            self.store_position()
        super().mouseReleaseEvent(event)

    def store_position(self):
        """Store the current position and dimensions."""
        if hasattr(self, 'scene_ref') and self.scene_ref and isinstance(self.scene_ref, ImageScene):
            QTimer.singleShot(0, self.scene_ref.store_item_changes)

    def itemChange(self, change, value):
        """Handle item changes to prevent trailing effect."""
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionHasChanged:
            # Store position after move
            self.scene().update()  # Force scene update to prevent trailing
            self.store_position()
        return super().itemChange(change, value)

# -------------------------------
# Scene zur Annotation
# -------------------------------
class ImageScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None  # Reference to main window for storing annotations
        self.drawing = False
        self.start = QPointF()
        self.temp_rect = None
        self.min_size = 20  # Mindestgröße in Pixeln
        self.current_color = QColor("black")
        self.current_class = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.start = event.scenePos()
            self.temp_rect = QGraphicsRectItem(QRectF(self.start, self.start))
            pen = QPen(self.current_color, 2)
            self.temp_rect.setPen(pen)
            self.addItem(self.temp_rect)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.temp_rect:
            rect = QRectF(self.start, event.scenePos()).normalized()
            self.temp_rect.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            final_rect = self.temp_rect.rect()
            if final_rect.width() < self.min_size or final_rect.height() < self.min_size:
                self.removeItem(self.temp_rect)
            else:
                # Create resizable rect item with proper parameters
                self.removeItem(self.temp_rect)
                rect_item = ResizableRectItem(final_rect, self.current_class, None)
                pen = QPen(self.current_color, 2)
                rect_item.setPen(pen)
                rect_item.scene_ref = self  # Set scene reference
                self.addItem(rect_item)
                self.store_item_changes()  # Store initial box position
            self.temp_rect = None
        super().mouseReleaseEvent(event)

    def store_item_changes(self):
        """Store current annotations when boxes change."""
        if self.main_window:
            # Use QTimer to avoid multiple rapid saves
            if not hasattr(self, '_save_timer'):
                self._save_timer = QTimer()
                self._save_timer.setSingleShot(True)
                self._save_timer.timeout.connect(self.main_window.store_annotations)
            self._save_timer.start(100)  # Wait 100ms before saving

# -------------------------------
# Hauptapplikation
# -------------------------------
class ImageLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bounding Box Annotation Tool")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.source_dir = ""
        self.dest_dir = ""
        self.image_files = []
        self.current_index = -1
        # Annotationen werden als Liste von Tupeln gespeichert:
        # (class_id, x_min, y_min, x_max, y_max)
        self.annotations = {}
        # Keine vordefinierten Klassen – der Benutzer fügt Klassen hinzu.
        self.classes = []  
        # Farbpalette für automatische Zuordnung (maximal 8 unterschiedliche Farben)
        self.color_palette = [
            QColor("red"), QColor("green"), QColor("blue"),
            QColor("yellow"), QColor("magenta"), QColor("cyan"),
            QColor("orange"), QColor("purple")
        ]
        self.init_ui()

    def init_ui(self):
        # Hauptlayout und zentraler Widget
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # QGraphicsView setup
        self.scene = ImageScene()
        self.scene.main_window = self  # Set main window reference
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        main_layout.addWidget(self.view, stretch=4)

        # Sidebar setup
        sidebar_container = QWidget()
        sidebar_layout = QHBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        sidebar_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-left: 1px solid #ddd;
            }
        """)

        # Scrollbare Seitenleiste
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setFixedWidth(300)
        self.sidebar_widget.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border: none;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                color: #333;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border-color: #ccc;
            }
            QLabel {
                color: #333;
                padding: 4px;
            }
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px;
                margin: 2px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: rgba(0, 0, 0, 0.1);
            }
        """)
        scroll = QScrollArea()
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f5f5f5;
            }
            QScrollBar:vertical {
                border: none;
                background: #f5f5f5;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c1c1c1;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a8a8a8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        scroll.setWidget(self.sidebar_widget)
        scroll.setWidgetResizable(True)
        sidebar_layout.addWidget(scroll)

        side_layout = QVBoxLayout(self.sidebar_widget)

        # Verzeichniswahl und Anzeige
        btn_load_source = QPushButton("Quellverzeichnis wählen")
        btn_load_source.clicked.connect(self.choose_source_dir)
        side_layout.addWidget(btn_load_source)

        self.lbl_source_dir = QLabel("Quellverzeichnis: nicht gesetzt")
        side_layout.addWidget(self.lbl_source_dir)

        btn_load_dest = QPushButton("Zielverzeichnis wählen")
        btn_load_dest.clicked.connect(self.choose_dest_dir)
        side_layout.addWidget(btn_load_dest)

        self.lbl_dest_dir = QLabel("Zielverzeichnis: nicht gesetzt")
        side_layout.addWidget(self.lbl_dest_dir)

        # Bildinfo und Navigation
        self.lbl_image_info = QLabel("Kein Bild geladen")
        side_layout.addWidget(self.lbl_image_info)

        nav_layout = QHBoxLayout()
        btn_prev = QPushButton("Vorheriges Bild")
        btn_prev.clicked.connect(self.previous_image)
        nav_layout.addWidget(btn_prev)
        btn_next = QPushButton("Nächstes Bild")
        btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(btn_next)
        side_layout.addLayout(nav_layout)

        # Button zum Löschen selektierter Boxen
        btn_delete_box = QPushButton("Ausgewählte Box(en) löschen")
        btn_delete_box.clicked.connect(self.delete_selected_boxes)
        side_layout.addWidget(btn_delete_box)

        # Klassenverwaltung: Liste der verfügbaren Objektklassen
        side_layout.addWidget(QLabel("Aktive Klasse:"))
        self.class_list = QListWidget()
        self.update_class_list()
        if self.classes:
            self.class_list.setCurrentRow(0)
            self.scene.current_class = 0
            self.scene.current_color = self.classes[0][1]
        self.class_list.currentRowChanged.connect(self.change_active_class)
        side_layout.addWidget(self.class_list)

        btn_add_class = QPushButton("Klasse hinzufügen")
        btn_add_class.clicked.connect(self.add_class)
        side_layout.addWidget(btn_add_class)

        btn_remove_class = QPushButton("Klasse entfernen")
        btn_remove_class.clicked.connect(self.remove_class)
        side_layout.addWidget(btn_remove_class)

        # Info-Button mit Shortcuts-Legende
        btn_info = QPushButton("Info / Shortcuts")
        btn_info.clicked.connect(self.show_info)
        side_layout.addWidget(btn_info)

        # Dataset Generation Button
        self.generate_button = QPushButton("Dataset generieren")
        self.generate_button.setMinimumHeight(60)
        self.generate_button.setFont(QFont("", 14, QFont.Weight.Bold))
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.generate_button.clicked.connect(self.generate_dataset)
        side_layout.addWidget(self.generate_button)

        # Add stretch at the bottom
        side_layout.addStretch()
        
        main_layout.addWidget(sidebar_container)

    def update_class_list(self):
        """Aktualisiert die Anzeige der Klassenliste mit doppelter, fetter Schrift und farbigem Hintergrund."""
        self.class_list.clear()
        font = QFont("Arial", 12, QFont.Weight.Bold)
        for idx, (name, color) in enumerate(self.classes):
            item = QListWidgetItem(name)
            item.setFont(font)
            # Create a semi-transparent version of the color for better readability
            bg_color = QColor(color)
            bg_color.setAlpha(40)  # 40% opacity
            item.setBackground(QBrush(bg_color))
            item.setForeground(QBrush(color.darker(150)))  # Darker text color
            self.class_list.addItem(item)

    def choose_source_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Quellverzeichnis wählen")
        if directory:
            self.source_dir = directory
            self.lbl_source_dir.setText(f"Quellverzeichnis: {directory}")
            self.load_images()
    
    def choose_dest_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Zielverzeichnis wählen")
        if directory:
            self.dest_dir = directory
            self.lbl_dest_dir.setText(f"Zielverzeichnis: {directory}")

    def load_images(self):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_files = [os.path.join(self.source_dir, f) for f in os.listdir(self.source_dir)
                            if f.lower().endswith(exts)]
        self.image_files.sort()
        if not self.image_files:
            QMessageBox.warning(self, "Fehler", "Keine unterstützten Bilder im Quellverzeichnis gefunden!")
            return
        self.current_index = 0
        self.load_current_image()

    def load_current_image(self):
        self.scene.clear()
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return
            
        image_path = self.image_files[self.current_index]
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Fehler", f"Bild {image_path} konnte nicht geladen werden.")
            return
            
        img_width = pixmap.width()
        img_height = pixmap.height()
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.scene.addPixmap(pixmap)
        self.lbl_image_info.setText(f"Bild {self.current_index+1}/{len(self.image_files)}: {os.path.basename(image_path)}")
        
        # Vorhandene Annotationen wieder einfügen
        if image_path in self.annotations:
            for ann in self.annotations[image_path]:
                # Convert normalized coordinates back to pixel values
                class_id, x_center, y_center, width, height = ann
                x_min = (x_center - width/2) * img_width
                y_min = (y_center - height/2) * img_height
                box_width = width * img_width
                box_height = height * img_height
                
                if class_id < len(self.classes):
                    color = self.classes[class_id][1]
                else:
                    color = QColor("black")
                    
                rect = QRectF(x_min, y_min, box_width, box_height)
                box = ResizableRectItem(rect, class_id)
                box.scene_ref = self.scene  # Set scene reference
                pen = QPen(color, 2)
                box.setPen(pen)
                self.scene.addItem(box)
        # Aktive Klasse in der Scene aktualisieren
        current_row = self.class_list.currentRow()
        if current_row >= 0 and current_row < len(self.classes):
            self.scene.current_class = current_row
            self.scene.current_color = self.classes[current_row][1]

    def change_active_class(self, index):
        if index < 0 or index >= len(self.classes):
            return
        self.scene.current_class = index
        self.scene.current_color = self.classes[index][1]

    def add_class(self):
        if len(self.classes) >= len(self.color_palette):
            QMessageBox.information(self, "Info", "Maximal verfügbare Klassen erreicht.")
            return
        name, ok = QInputDialog.getText(self, "Neue Klasse", "Name der Klasse:")
        if not ok or not name.strip():
            return
        # Automatische Farbauswahl aus der Palette (in Reihenfolge)
        color = self.color_palette[len(self.classes) % len(self.color_palette)]
        self.classes.append((name.strip(), color))
        self.update_class_list()
        # Setze die aktive Klasse, falls dies die erste Klasse ist.
        if len(self.classes) == 1:
            self.class_list.setCurrentRow(0)
            self.scene.current_class = 0
            self.scene.current_color = color

    def remove_class(self):
        row = self.class_list.currentRow()
        if row < 0 or row >= len(self.classes):
            return
        reply = QMessageBox.question(self, "Klasse entfernen", 
                                     f"Möchten Sie die Klasse '{self.classes[row][0]}' wirklich entfernen?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            del self.classes[row]
            self.update_class_list()
            if self.classes:
                self.scene.current_class = 0
                self.scene.current_color = self.classes[0][1]

    def show_info(self):
        help_text = (
            "Shortcuts:\n"
            "N: Nächstes Bild\n"
            "P: Vorheriges Bild\n"
            "K: Kopiere Annotationen vom vorherigen Bild\n"
            "Entf/Backspace: Löscht ausgewählte Box\n"
            "Strg + Mausrad: Zoom in/out\n\n"
            "Zusätzlich:\n"
            "- Mit den Buttons in der Seitenleiste kann zwischen den Bildern gewechselt werden.\n"
            "- Mit 'Klasse hinzufügen' werden Klassen erstellt, denen automatisch eine Hintergrundfarbe aus einer vordefinierten Palette zugewiesen wird.\n"
            "- Mit 'Klasse entfernen' lassen sich vorhandene Klassen löschen.\n"
            "- Nach Verzeichniswahl werden die Pfade in der Seitenleiste angezeigt.\n"
            "- Mit der Maus lassen sich Bounding Boxen an den Ecken skalieren."
        )
        QMessageBox.information(self, "Info / Shortcuts", help_text)

    def delete_selected_boxes(self):
        for item in self.scene.selectedItems():
            if isinstance(item, ResizableRectItem):
                self.scene.removeItem(item)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_N:
            self.next_image()
        elif event.key() == Qt.Key.Key_P:
            self.previous_image()
        elif event.key() == Qt.Key.Key_K:
            self.copy_previous_annotations()
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_boxes()
        else:
            super().keyPressEvent(event)

    def store_annotations(self):
        """Speichert die aktuellen Annotationen des angezeigten Bildes."""
        image_path = self.image_files[self.current_index]
        image = cv2.imread(image_path)
        if image is None:
            return
        
        img_height, img_width = image.shape[:2]
        ann_list = []
        
        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                r = item.rect()
                # Convert to YOLO format (normalized)
                x_center = (r.x() + r.width()/2) / img_width
                y_center = (r.y() + r.height()/2) / img_height
                width = r.width() / img_width
                height = r.height() / img_height
                
                # Ensure values are within valid range
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                ann_list.append((item.class_id, x_center, y_center, width, height))
        
        self.annotations[image_path] = ann_list

    def next_image(self):
        if not self.image_files:
            return
        self.store_annotations()
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()

    def previous_image(self):
        if not self.image_files:
            return
        self.store_annotations()
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def copy_previous_annotations(self):
        if self.current_index <= 0:
            QMessageBox.information(self, "Info", "Keine vorherigen Annotationen vorhanden.")
            return
        prev_image = self.image_files[self.current_index - 1]
        if prev_image not in self.annotations:
            QMessageBox.information(self, "Info", "Vorheriges Bild hat keine Annotationen.")
            return
        self.store_annotations()
        self.annotations[self.image_files[self.current_index]] = self.annotations[prev_image].copy()
        self.load_current_image()

    def generate_dataset(self):
        if not self.dest_dir:
            QMessageBox.warning(self, "Fehler", "Bitte wählen Sie zuerst ein Zielverzeichnis!")
            return
            
        self.store_annotations()
        
        for image_path in self.image_files:
            base = os.path.splitext(os.path.basename(image_path))[0]
            txt_file = os.path.join(self.dest_dir, base + ".txt")
            ann_list = self.annotations.get(image_path, [])
            if ann_list:
                # Annotations are already normalized, save directly
                with open(txt_file, 'w') as f:
                    for cls_id, x_center, y_center, width, height in ann_list:
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            dest_image = os.path.join(self.dest_dir, os.path.basename(image_path))
            try:
                shutil.copy(image_path, dest_image)
            except Exception as e:
                QMessageBox.warning(self, "Fehler", f"Bild konnte nicht kopiert werden: {e}")
        classes_file = os.path.join(self.dest_dir, "classes.txt")
        class_names = [name for name, _ in self.classes]
        utils.save_classes_file(classes_file, class_names)
        QMessageBox.information(self, "Erfolg", "Dataset wurde generiert!")

        if hasattr(self, 'project_manager') and self.project_manager:
            self.save_classes_to_project()
            self.project_manager.mark_step_completed(WorkflowStep.LABELING)        

    def save_classes_to_project(self):
        """Speichert definierte Klassen ins Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            # Klassen ins Projekt übertragen
            for idx, (name, color) in enumerate(self.classes):
                self.project_manager.add_class(idx, name, color.name())
            
            print(f"Klassen gespeichert: {self.project_manager.get_classes()}")
    
    def load_classes_from_project(self):
        """Lädt Klassen aus dem Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            classes = self.project_manager.get_classes()
            colors = self.project_manager.get_class_colors()
            
            self.classes = []
            for class_id in sorted(classes.keys()):
                class_name = classes[class_id]
                color_hex = colors.get(class_id, "#FF0000")
                from PyQt6.QtGui import QColor
                self.classes.append((class_name, QColor(color_hex)))
            
            self.update_class_list()
    
    def generate_dataset_with_project_integration(self):
        """Erweiterte Dataset-Generierung mit Projekt-Integration"""
        # Originale generate_dataset Funktion ausführen
        self.generate_dataset()
        
        # Zusätzlich: Klassen ins Projekt speichern und Workflow markieren
        if hasattr(self, 'project_manager') and self.project_manager:
            self.save_classes_to_project()
            self.project_manager.mark_step_completed(WorkflowStep.LABELING)

"""
Ergänzungen für gui/image_labeling.py
Diese Methoden sollten zur ImageLabelingApp-Klasse hinzugefügt werden
"""

class ImageLabelingAppExtensions:
    """Erweiterungen für die Labeling App zur Projekt-Integration"""
    
    def save_classes_to_project(self):
        """Speichert definierte Klassen ins Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            # Klassen ins Projekt übertragen
            for idx, (name, color) in enumerate(self.classes):
                self.project_manager.add_class(idx, name, color.name())
            
            print(f"Klassen gespeichert: {self.project_manager.get_classes()}")
    
    def load_classes_from_project(self):
        """Lädt Klassen aus dem Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            classes = self.project_manager.get_classes()
            colors = self.project_manager.get_class_colors()
            
            self.classes = []
            for class_id in sorted(classes.keys()):
                class_name = classes[class_id]
                color_hex = colors.get(class_id, "#FF0000")
                from PyQt6.QtGui import QColor
                self.classes.append((class_name, QColor(color_hex)))
            
            self.update_class_list()
    
    def generate_dataset_with_project_integration(self):
        """Erweiterte Dataset-Generierung mit Projekt-Integration"""
        # Originale generate_dataset Funktion ausführen
        self.generate_dataset()
        
        # Zusätzlich: Klassen ins Projekt speichern und Workflow markieren
        if hasattr(self, 'project_manager') and self.project_manager:
            self.save_classes_to_project()
            self.project_manager.mark_step_completed(WorkflowStep.LABELING)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageLabelingApp()
    window.show()  # Window will be maximized due to WindowState setting
    sys.exit(app.exec())