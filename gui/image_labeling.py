# image_labeling.py
import sys, os, shutil
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsPolygonItem, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QMessageBox, QInputDialog,
    QScrollArea, QRadioButton, QButtonGroup
)
from PyQt6.QtGui import QPixmap, QPen, QColor, QFont, QBrush, QPainter, QPolygonF
from PyQt6.QtCore import Qt, QRectF, QPointF, QSizeF, QTimer, QPoint

import utils.labeling_utils as utils
from project_manager import ProjectManager, WorkflowStep
from gui.augmentation_preview import load_sample_image

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
        else:
            # Also store position after move
            self.store_position()
        super().mouseReleaseEvent(event)

    def store_position(self):
        """Store the current position and dimensions."""
        if hasattr(self, 'scene_ref') and self.scene_ref and isinstance(self.scene_ref, ImageScene):
            # Immediate storage without timer to ensure changes are saved
            self.scene_ref.store_item_changes()

    def itemChange(self, change, value):
        """Handle item changes to prevent trailing effect."""
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionHasChanged:
            # Force scene update to prevent trailing
            self.scene().update()
            # Store position immediately when position changes
            self.store_position()
        elif change == QGraphicsRectItem.GraphicsItemChange.ItemPositionChange:
            # Also store during position change to catch all movements
            return super().itemChange(change, value)
        return super().itemChange(change, value)

# -------------------------------
# Resizable Polygon Item
# -------------------------------
class ResizablePolygonItem(QGraphicsPolygonItem):
    def __init__(self, polygon, class_id, parent=None):
        super().__init__(polygon, parent)
        self.scene_ref = None  # Reference to scene for updating annotations
        self.class_id = class_id
        self.points = []
        self.handle_size = QSizeF(20.0, 20.0)
        self.setFlags(
            QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsPolygonItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsPolygonItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.update_points_from_polygon()

    def update_points_from_polygon(self):
        """Update internal points list from polygon."""
        self.points = [point for point in self.polygon()]

    def paint(self, painter, option, widget):
        """Paint the polygon and its handles."""
        super().paint(painter, option, widget)

        # Draw handles for vertices
        if self.isSelected():
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            for point in self.points:
                painter.drawEllipse(QRectF(
                    point.x() - self.handle_size.width() / 2,
                    point.y() - self.handle_size.height() / 2,
                    self.handle_size.width(),
                    self.handle_size.height()
                ))

    def mousePressEvent(self, event):
        """Handle mouse press events for vertex editing."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            # Check if we clicked on a vertex handle
            for i, point in enumerate(self.points):
                if (pos - point).manhattanLength() < 15:
                    self.setSelected(True)
                    self.edit_vertex = i
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for vertex editing."""
        if hasattr(self, 'edit_vertex'):
            pos = event.pos()
            self.points[self.edit_vertex] = pos
            # Update polygon
            polygon = QPolygonF(self.points)
            self.setPolygon(polygon)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if hasattr(self, 'edit_vertex'):
            del self.edit_vertex
            self.store_position()
        else:
            self.store_position()
        super().mouseReleaseEvent(event)

    def store_position(self):
        """Store the current position and shape."""
        if hasattr(self, 'scene_ref') and self.scene_ref and isinstance(self.scene_ref, ImageScene):
            self.scene_ref.store_item_changes()

    def itemChange(self, change, value):
        """Handle item changes."""
        if change == QGraphicsPolygonItem.GraphicsItemChange.ItemPositionHasChanged:
            self.scene().update()
            self.store_position()
        elif change == QGraphicsPolygonItem.GraphicsItemChange.ItemPositionChange:
            return super().itemChange(change, value)
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

        # Polygon drawing state
        self.polygon_mode = False
        self.drawing_polygon = False
        self.polygon_points = []
        self.temp_polygon_lines = []  # Temporary lines for visual feedback
        self.start_point_threshold = 15  # Distance threshold to close polygon

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.polygon_mode:
                self.handle_polygon_click(event.scenePos())
            else:
                # Bounding box mode
                self.drawing = True
                self.start = event.scenePos()
                self.temp_rect = QGraphicsRectItem(QRectF(self.start, self.start))
                pen = QPen(self.current_color, 2)
                self.temp_rect.setPen(pen)
                self.addItem(self.temp_rect)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.polygon_mode and self.drawing_polygon and self.polygon_points:
            # Show preview line from last point to current mouse position
            self.update_polygon_preview(event.scenePos())
        elif self.drawing and self.temp_rect:
            rect = QRectF(self.start, event.scenePos()).normalized()
            self.temp_rect.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing and not self.polygon_mode:
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

    def handle_polygon_click(self, pos):
        """Handle mouse clicks for polygon drawing."""
        if not self.drawing_polygon:
            # Start new polygon
            self.drawing_polygon = True
            self.polygon_points = [pos]
            self.temp_polygon_lines = []
        else:
            # Check if we clicked near the start point to close polygon
            start_point = self.polygon_points[0]
            if (pos - start_point).manhattanLength() < self.start_point_threshold:
                self.finish_polygon()
            else:
                # Add new point
                self.polygon_points.append(pos)
                # Add line from previous point to current point
                if len(self.polygon_points) > 1:
                    line = self.addLine(
                        self.polygon_points[-2].x(), self.polygon_points[-2].y(),
                        self.polygon_points[-1].x(), self.polygon_points[-1].y(),
                        QPen(self.current_color, 2)
                    )
                    self.temp_polygon_lines.append(line)

    def update_polygon_preview(self, mouse_pos):
        """Update preview line while drawing polygon."""
        # Remove old preview line
        if hasattr(self, 'preview_line') and self.preview_line:
            self.removeItem(self.preview_line)

        # Add preview line from last point to current mouse position
        if self.polygon_points:
            last_point = self.polygon_points[-1]
            self.preview_line = self.addLine(
                last_point.x(), last_point.y(),
                mouse_pos.x(), mouse_pos.y(),
                QPen(self.current_color, 1, Qt.PenStyle.DashLine)
            )

    def finish_polygon(self):
        """Complete the current polygon."""
        if len(self.polygon_points) < 3:
            # Need at least 3 points for a polygon
            self.cancel_polygon()
            return

        # Remove temporary lines and preview
        for line in self.temp_polygon_lines:
            self.removeItem(line)
        self.temp_polygon_lines = []

        if hasattr(self, 'preview_line') and self.preview_line:
            self.removeItem(self.preview_line)
            self.preview_line = None

        # Create polygon item
        polygon = QPolygonF(self.polygon_points)
        polygon_item = ResizablePolygonItem(polygon, self.current_class, None)
        pen = QPen(self.current_color, 2)
        brush = QBrush(self.current_color)
        brush.setStyle(Qt.BrushStyle.Dense4Pattern)  # Semi-transparent fill
        polygon_item.setPen(pen)
        polygon_item.setBrush(brush)
        polygon_item.scene_ref = self
        self.addItem(polygon_item)

        # Reset polygon drawing state
        self.drawing_polygon = False
        self.polygon_points = []
        self.store_item_changes()

    def cancel_polygon(self):
        """Cancel current polygon drawing."""
        # Remove temporary lines
        for line in self.temp_polygon_lines:
            self.removeItem(line)
        self.temp_polygon_lines = []

        if hasattr(self, 'preview_line') and self.preview_line:
            self.removeItem(self.preview_line)
            self.preview_line = None

        # Reset state
        self.drawing_polygon = False
        self.polygon_points = []

    def store_item_changes(self):
        """Store current annotations when boxes change."""
        if self.main_window:
            # Direct storage to ensure changes are immediately saved
            self.main_window.store_annotations()

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
        
        # Set initial drawing mode
        self.scene.polygon_mode = False  # Start with bounding box mode

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

        # Button zum Löschen selektierter Annotationen
        btn_delete_annotation = QPushButton("Ausgewählte Annotation(en) löschen")
        btn_delete_annotation.clicked.connect(self.delete_selected_boxes)
        side_layout.addWidget(btn_delete_annotation)

        # Drawing mode selection
        mode_group = QWidget()
        mode_layout = QVBoxLayout(mode_group)
        
        # Add mode description
        mode_label = QLabel("Zeichenmodus:")
        mode_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        mode_layout.addWidget(mode_label)
        
        mode_desc = QLabel("⚠️ Achtung: Wechsel löscht alle bestehenden Annotationen im Bild!")
        mode_desc.setStyleSheet("color: #d32f2f; font-size: 10px; font-weight: bold; padding: 4px;")
        mode_desc.setWordWrap(True)
        mode_layout.addWidget(mode_desc)

        self.mode_button_group = QButtonGroup()
        self.bbox_radio = QRadioButton("Bounding Box")
        self.polygon_radio = QRadioButton("Polygon")
        self.bbox_radio.setChecked(True)  # Default to bounding box mode

        self.mode_button_group.addButton(self.bbox_radio, 0)
        self.mode_button_group.addButton(self.polygon_radio, 1)
        self.mode_button_group.buttonClicked.connect(self.change_drawing_mode)

        mode_layout.addWidget(self.bbox_radio)
        mode_layout.addWidget(self.polygon_radio)
        side_layout.addWidget(mode_group)

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

        # Button to continue with augmentation
        self.next_button = QPushButton("Weiter zur Augmentation")
        self.next_button.setMinimumHeight(40)
        self.next_button.clicked.connect(self.open_augmentation_app)
        side_layout.addWidget(self.next_button)

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

    def load_paths_from_project(self):
        """Load source and destination directories from the project manager."""
        if hasattr(self, "project_manager") and self.project_manager:
            self.source_dir = str(self.project_manager.get_raw_images_dir())
            self.dest_dir = str(self.project_manager.get_labeled_dir())
            self.lbl_source_dir.setText(f"Quellverzeichnis: {self.source_dir}")
            self.lbl_dest_dir.setText(f"Zielverzeichnis: {self.dest_dir}")
            if os.path.isdir(self.source_dir):
                self.load_images()         

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
                if len(ann) < 3:
                    continue

                # Check if this is an old format annotation (backward compatibility)
                if len(ann) == 5 and isinstance(ann[0], (int, float)):
                    # Old format: (class_id, x_center, y_center, width, height)
                    class_id, x_center, y_center, width, height = ann
                    ann = ('bbox', class_id, x_center, y_center, width, height)

                ann_type = ann[0]
                class_id = ann[1]

                if class_id < len(self.classes):
                    color = self.classes[class_id][1]
                else:
                    color = QColor("black")

                if ann_type == 'bbox' and len(ann) == 6:
                    # Bounding box annotation
                    _, class_id, x_center, y_center, width, height = ann
                    x_min = (x_center - width/2) * img_width
                    y_min = (y_center - height/2) * img_height
                    box_width = width * img_width
                    box_height = height * img_height

                    rect = QRectF(x_min, y_min, box_width, box_height)
                    box = ResizableRectItem(rect, class_id)
                    box.scene_ref = self.scene
                    pen = QPen(color, 2)
                    box.setPen(pen)
                    self.scene.addItem(box)

                elif ann_type == 'polygon' and len(ann) >= 8:  # At least 3 points (class_id + 6 coordinates)
                    # Polygon annotation
                    coords = ann[2:]  # Skip type and class_id
                    if len(coords) % 2 != 0:
                        continue  # Skip invalid polygon (odd number of coordinates)

                    # Convert normalized coordinates back to pixel values
                    points = []
                    for i in range(0, len(coords), 2):
                        x = coords[i] * img_width
                        y = coords[i+1] * img_height
                        points.append(QPointF(x, y))

                    if len(points) >= 3:
                        polygon = QPolygonF(points)
                        polygon_item = ResizablePolygonItem(polygon, class_id)
                        polygon_item.scene_ref = self.scene
                        pen = QPen(color, 2)
                        brush = QBrush(color)
                        brush.setStyle(Qt.BrushStyle.Dense4Pattern)
                        polygon_item.setPen(pen)
                        polygon_item.setBrush(brush)
                        self.scene.addItem(polygon_item)
        # Aktive Klasse in der Scene aktualisieren
        current_row = self.class_list.currentRow()
        if current_row >= 0 and current_row < len(self.classes):
            self.scene.current_class = current_row
            self.scene.current_color = self.classes[current_row][1]

    def change_drawing_mode(self, button):
        """Change between bounding box and polygon drawing modes."""
        # Store current annotations before switching
        self.store_annotations()
        
        # Check if there are annotations of the other type
        current_image = self.image_files[self.current_index] if self.current_index >= 0 else None
        if current_image and current_image in self.annotations:
            current_annotations = self.annotations[current_image]
            has_bboxes = any(ann[0] == 'bbox' for ann in current_annotations)
            has_polygons = any(ann[0] == 'polygon' for ann in current_annotations)
            
            new_mode_is_polygon = (button == self.polygon_radio)
            
            # Check for conflicting annotations
            if new_mode_is_polygon and has_bboxes:
                reply = QMessageBox.question(
                    self, "Zeichenmodus wechseln",
                    "Sie wechseln zu Polygon-Modus, aber das aktuelle Bild hat Bounding Box Annotationen.\n\n"
                    "Diese werden gelöscht! Möchten Sie fortfahren?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    # Revert radio button selection
                    self.bbox_radio.setChecked(True)
                    return
                    
            elif not new_mode_is_polygon and has_polygons:
                reply = QMessageBox.question(
                    self, "Zeichenmodus wechseln",
                    "Sie wechseln zu Bounding Box-Modus, aber das aktuelle Bild hat Polygon Annotationen.\n\n"
                    "Diese werden gelöscht! Möchten Sie fortfahren?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    # Revert radio button selection
                    self.polygon_radio.setChecked(True)
                    return
            
            # Clear conflicting annotations
            if new_mode_is_polygon and has_bboxes:
                # Remove all bbox annotations
                self.annotations[current_image] = [ann for ann in current_annotations if ann[0] != 'bbox']
            elif not new_mode_is_polygon and has_polygons:
                # Remove all polygon annotations
                self.annotations[current_image] = [ann for ann in current_annotations if ann[0] != 'polygon']
        
        if button == self.bbox_radio:
            self.scene.polygon_mode = False
            self.setWindowTitle("Bounding Box Annotation Tool")
        elif button == self.polygon_radio:
            self.scene.polygon_mode = True
            self.setWindowTitle("Polygon Annotation Tool")

        # Cancel any ongoing polygon drawing when switching modes
        if hasattr(self.scene, 'cancel_polygon'):
            self.scene.cancel_polygon()
        
        # Reload current image to update display
        if current_image:
            self.load_current_image()

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
            "Entf/Backspace: Löscht ausgewählte Annotation\n"
            "B: Wechsel zu Bounding Box Modus\n"
            "G: Wechsel zu Polygon Modus\n"
            "ESC: Polygon-Zeichnung abbrechen\n"
            "Strg + Mausrad: Zoom in/out\n\n"
            "Zeichenmodi:\n"
            "- Bounding Box: Klicken und ziehen für Rechteck\n"
            "- Polygon: Einzelne Punkte klicken, Startpunkt nochmals klicken zum Beenden\n\n"
            "Zusätzlich:\n"
            "- Mit den Buttons in der Seitenleiste kann zwischen den Bildern gewechselt werden.\n"
            "- Mit 'Klasse hinzufügen' werden Klassen erstellt, denen automatisch eine Hintergrundfarbe aus einer vordefinierten Palette zugewiesen wird.\n"
            "- Mit 'Klasse entfernen' lassen sich vorhandene Klassen löschen.\n"
            "- Nach Verzeichniswahl werden die Pfade in der Seitenleiste angezeigt.\n"
            "- Bounding Boxen können an den Ecken skaliert werden.\n"
            "- Polygon-Punkte können einzeln verschoben werden."
        )
        QMessageBox.information(self, "Info / Shortcuts", help_text)

    def delete_selected_boxes(self):
        deleted_any = False
        for item in self.scene.selectedItems():
            if isinstance(item, (ResizableRectItem, ResizablePolygonItem)):
                self.scene.removeItem(item)
                deleted_any = True

        # Store annotations after deletion to persist changes
        if deleted_any:
            self.store_annotations()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_N:
            self.next_image()
        elif event.key() == Qt.Key.Key_P:
            self.previous_image()
        elif event.key() == Qt.Key.Key_K:
            self.copy_previous_annotations()
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_boxes()
        elif event.key() == Qt.Key.Key_Escape:
            # Cancel current polygon drawing
            if hasattr(self.scene, 'cancel_polygon'):
                self.scene.cancel_polygon()
        elif event.key() == Qt.Key.Key_B:
            # Switch to bounding box mode
            self.bbox_radio.setChecked(True)
            self.change_drawing_mode(self.bbox_radio)
        elif event.key() == Qt.Key.Key_G:
            # Switch to polygon mode
            self.polygon_radio.setChecked(True)
            self.change_drawing_mode(self.polygon_radio)
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event):
        """Save annotations when application loses focus."""
        self.store_annotations()
        super().focusOutEvent(event)

    def closeEvent(self, event):
        """Save annotations when application is closed."""
        self.store_annotations()
        super().closeEvent(event)

    def store_annotations(self):
        """Speichert die aktuellen Annotationen des angezeigten Bildes."""
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return

        image_path = self.image_files[self.current_index]
        image = cv2.imread(image_path)
        if image is None:
            return

        img_height, img_width = image.shape[:2]
        ann_list = []

        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                r = item.rect()
                # Skip invalid rectangles
                if r.width() <= 0 or r.height() <= 0:
                    continue

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

                # Only add if the box has valid dimensions
                if width > 0 and height > 0:
                    ann_list.append(('bbox', item.class_id, x_center, y_center, width, height))

            elif isinstance(item, ResizablePolygonItem):
                polygon = item.polygon()
                if len(polygon) < 3:
                    continue

                # Convert polygon points to normalized coordinates
                normalized_points = []
                for point in polygon:
                    norm_x = max(0.0, min(1.0, point.x() / img_width))
                    norm_y = max(0.0, min(1.0, point.y() / img_height))
                    normalized_points.extend([norm_x, norm_y])

                if len(normalized_points) >= 6:  # At least 3 points (6 coordinates)
                    ann_list.append(('polygon', item.class_id, *normalized_points))

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

        # Store current annotations first to preserve any changes
        self.store_annotations()

        # Only copy annotations if current image has no annotations yet
        current_image = self.image_files[self.current_index]
        if current_image not in self.annotations or len(self.annotations[current_image]) == 0:
            self.annotations[current_image] = self.annotations[prev_image].copy()
            self.load_current_image()
        else:
            # If current image already has annotations, ask user if they want to replace them
            reply = QMessageBox.question(self, "Annotationen ersetzen",
                                       "Das aktuelle Bild hat bereits Annotationen. Möchten Sie diese durch die vorherigen ersetzen?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.annotations[current_image] = self.annotations[prev_image].copy()
                self.load_current_image()

    def generate_dataset(self):
        if not self.dest_dir:
            QMessageBox.warning(self, "Fehler", "Bitte wählen Sie zuerst ein Zielverzeichnis!")
            return
            
        self.store_annotations()
        
        # Determine which format to save based on current mode
        save_polygons = self.scene.polygon_mode
        save_bboxes = not self.scene.polygon_mode
        
        total_polygons = 0
        total_bboxes = 0
        ignored_annotations = 0
        
        for image_path in self.image_files:
            base = os.path.splitext(os.path.basename(image_path))[0]
            txt_file = os.path.join(self.dest_dir, base + ".txt")
            ann_list = self.annotations.get(image_path, [])
            
            # Count annotations for user feedback
            polygons_in_image = [ann for ann in ann_list if ann[0] == 'polygon']
            bboxes_in_image = [ann for ann in ann_list if ann[0] == 'bbox']
            total_polygons += len(polygons_in_image)
            total_bboxes += len(bboxes_in_image)
            
            # Always create a label file, even if no annotations exist
            with open(txt_file, 'w') as f:
                for ann in ann_list:
                    if len(ann) < 3:
                        continue

                    ann_type = ann[0]
                    class_id = ann[1]

                    if ann_type == 'bbox' and len(ann) == 6 and save_bboxes:
                        # Bounding box: class_id x_center y_center width height
                        _, class_id, x_center, y_center, width, height = ann
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    elif ann_type == 'polygon' and len(ann) >= 8 and save_polygons:  # type + class_id + at least 6 coordinates (3 points)
                        # YOLO11 Segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                        coords = ann[2:]  # Skip type and class_id
                        if len(coords) >= 6 and len(coords) % 2 == 0:  # Must have even number of coordinates
                            coords_str = ' '.join([f"{coord:.6f}" for coord in coords])
                            f.write(f"{class_id} {coords_str}\n")
                    else:
                        # Annotation ignored due to mode mismatch
                        ignored_annotations += 1
        
        # Show user what was saved and what was ignored
        if save_polygons:
            mode_name = "Polygon (Segmentation)"
            saved_count = total_polygons
            ignored_count = total_bboxes
            ignored_type = "Bounding Boxes"
        else:
            mode_name = "Bounding Box (Detection)"
            saved_count = total_bboxes
            ignored_count = total_polygons
            ignored_type = "Polygone"
        
        message = f"Dataset im {mode_name} Format generiert!\n\n"
        message += f"✅ Gespeichert: {saved_count} Annotationen\n"
        
        if ignored_count > 0:
            message += f"⚠️ Ignoriert: {ignored_count} {ignored_type}\n"
            message += f"(Grund: Aktueller Modus ist '{mode_name}')\n\n"
            message += f"Tipp: Wechseln Sie den Zeichenmodus um alle Annotationen zu inkludieren."
            
        # Show comprehensive feedback
        QMessageBox.information(self, "Dataset generiert", message)
            # Copy images
            for image_path in self.image_files:
                dest_image = os.path.join(self.dest_dir, os.path.basename(image_path))
                try:
                    shutil.copy(image_path, dest_image)
                except Exception as e:
                    QMessageBox.warning(self, "Fehler", f"Bild konnte nicht kopiert werden: {e}")
        
        classes_file = os.path.join(self.dest_dir, "classes.txt")
        class_names = [name for name, _ in self.classes]
        utils.save_classes_file(classes_file, class_names)

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

    def open_augmentation_app(self):
        """Open augmentation tool and close labeling window."""
        try:
            from gui.augmentation_app import ImageAugmentationApp
            app = ImageAugmentationApp()
            app.project_manager = self.project_manager
            if self.project_manager:
                app.source_path = str(self.project_manager.get_labeled_dir())
                app.dest_path = str(self.project_manager.get_augmented_dir())
                app.source_label.setText(f"Quellverzeichnis: {app.source_path}")
                app.dest_label.setText(f"Zielverzeichnis: {app.dest_path}")

                saved_settings = self.project_manager.get_augmentation_settings()
                if saved_settings:
                    app.settings.update(saved_settings)

                app.update_expected_count()
                load_sample_image(app)

            self.augmentation_window = app
            app.show()
            self.close()
        except Exception as e:
            print(f"Fehler beim Öffnen der Augmentation-App: {e}")

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
    import argparse

    parser = argparse.ArgumentParser(description="Image Labeling App")
    parser.add_argument("project_dir", nargs="?", help="Projektverzeichnis")
    args, qt_args = parser.parse_known_args()

    app = QApplication(sys.argv[:1] + qt_args)
    window = ImageLabelingApp()

    if args.project_dir:
        from project_manager import ProjectManager

        window.project_manager = ProjectManager(args.project_dir)
        window.load_paths_from_project()
        window.load_classes_from_project()

    window.show()  # Window will be maximized due to WindowState setting
    sys.exit(app.exec())