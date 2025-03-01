# image_labeling.py
import sys, os, shutil
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QMessageBox, QInputDialog
)
from PyQt6.QtGui import QPixmap, QPen, QColor, QFont, QBrush
from PyQt6.QtCore import Qt, QRectF, QPointF
import utils.labeling_utils as utils

# -------------------------------
# Resizable Bounding Box Item
# -------------------------------
class ResizableRectItem(QGraphicsRectItem):
    def __init__(self, rect, class_id, color, *args, **kwargs):
        super().__init__(rect, *args, **kwargs)
        self.class_id = class_id
        self.handleSize = 8.0
        self.handles = {}
        self.resizing = False
        self.currentHandle = None
        self.color = color
        self.setPen(QPen(self.color, 2))
        self.setFlags(
            QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable
        )
        self.updateHandlesPos()

    def updateHandlesPos(self):
        r = self.rect()
        self.handles = {
            "tl": QRectF(r.topLeft().x() - self.handleSize/2, r.topLeft().y() - self.handleSize/2, self.handleSize, self.handleSize),
            "tr": QRectF(r.topRight().x() - self.handleSize/2, r.topRight().y() - self.handleSize/2, self.handleSize, self.handleSize),
            "bl": QRectF(r.bottomLeft().x() - self.handleSize/2, r.bottomLeft().y() - self.handleSize/2, self.handleSize, self.handleSize),
            "br": QRectF(r.bottomRight().x() - self.handleSize/2, r.bottomRight().y() - self.handleSize/2, self.handleSize, self.handleSize),
        }

    def paint(self, painter, option, widget):
        # Zeichne die Box
        super().paint(painter, option, widget)
        # Zeichne die Handles, wenn selektiert
        if self.isSelected():
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            for rect in self.handles.values():
                painter.drawRect(rect)

    def mousePressEvent(self, event):
        # Prüfe, ob auf einen Handle geklickt wurde.
        for handle, rect in self.handles.items():
            if rect.contains(event.pos()):
                self.resizing = True
                self.currentHandle = handle
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resizing:
            pos = event.pos()
            r = self.rect()
            if self.currentHandle == "tl":
                newRect = QRectF(pos, r.bottomRight()).normalized()
            elif self.currentHandle == "tr":
                newRect = QRectF(QPointF(r.left(), pos.y()), QPointF(pos.x(), r.bottom())).normalized()
            elif self.currentHandle == "bl":
                newRect = QRectF(QPointF(pos.x(), r.top()), QPointF(r.right(), pos.y())).normalized()
            elif self.currentHandle == "br":
                newRect = QRectF(r.topLeft(), pos).normalized()
            minSize = 10
            if newRect.width() < minSize or newRect.height() < minSize:
                return
            self.setRect(newRect)
            self.updateHandlesPos()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.resizing = False
        self.currentHandle = None
        super().mouseReleaseEvent(event)

# -------------------------------
# Scene zur Annotation
# -------------------------------
class ImageScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.start = QPointF()
        self.temp_rect = None
        self.min_size = 50  # Mindestgröße in Pixeln
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
            rect = self.temp_rect.rect()
            if rect.width() < self.min_size or rect.height() < self.min_size:
                self.removeItem(self.temp_rect)
            else:
                self.removeItem(self.temp_rect)
                final_rect = ResizableRectItem(rect, self.current_class, self.current_color)
                self.addItem(final_rect)
            self.temp_rect = None
        super().mouseReleaseEvent(event)

# -------------------------------
# Hauptapplikation
# -------------------------------
class ImageLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bounding Box Annotation Tool")
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
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # QGraphicsView mit eigener Scene zur Bild- und Boxanzeige
        self.scene = ImageScene()
        self.view = QGraphicsView(self.scene)
        main_layout.addWidget(self.view, stretch=4)

        # Seitenleiste für Bedienungselemente
        side_layout = QVBoxLayout()

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

        btn_generate = QPushButton("Dataset generieren")
        btn_generate.clicked.connect(self.generate_dataset)
        side_layout.addWidget(btn_generate)

        side_layout.addStretch()
        main_layout.addLayout(side_layout, stretch=1)

    def update_class_list(self):
        """Aktualisiert die Anzeige der Klassenliste mit doppelter, fetter Schrift und farbigem Hintergrund."""
        self.class_list.clear()
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        for idx, (name, color) in enumerate(self.classes):
            item = QListWidgetItem(name)
            item.setFont(font)
            item.setBackground(QBrush(color))
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
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.scene.addPixmap(pixmap)
        self.lbl_image_info.setText(f"Bild {self.current_index+1}/{len(self.image_files)}: {os.path.basename(image_path)}")
        # Vorhandene Annotationen wieder einfügen
        if image_path in self.annotations:
            for ann in self.annotations[image_path]:
                class_id, x_min, y_min, x_max, y_max = ann
                if class_id < len(self.classes):
                    color = self.classes[class_id][1]
                else:
                    color = QColor("black")
                rect = QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
                box = ResizableRectItem(rect, class_id, color)
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
            "Entf/Backspace: Löscht ausgewählte Box\n\n"
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
        ann_list = []
        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                r = item.rect()
                ann_list.append((item.class_id, r.x(), r.y(), r.x() + r.width(), r.y() + r.height()))
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
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                continue
            img_width, img_height = pixmap.width(), pixmap.height()
            ann_list = self.annotations.get(image_path, [])
            if ann_list:
                utils.save_yolo_labels(txt_file, ann_list, img_width, img_height)
            dest_image = os.path.join(self.dest_dir, os.path.basename(image_path))
            try:
                shutil.copy(image_path, dest_image)
            except Exception as e:
                QMessageBox.warning(self, "Fehler", f"Bild konnte nicht kopiert werden: {e}")
        classes_file = os.path.join(self.dest_dir, "classes.txt")
        class_names = [name for name, _ in self.classes]
        utils.save_classes_file(classes_file, class_names)
        QMessageBox.information(self, "Erfolg", "Dataset wurde generiert!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageLabelingApp()
    window.showFullScreen()  # Start im Vollbildmodus (Windows)
    sys.exit(app.exec())
