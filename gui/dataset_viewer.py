"""Dataset viewer application module."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QScrollArea, QGridLayout, QMessageBox, QProgressBar,
    QDialog
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import time
import logging

class DatasetAnalyzer(QThread):
    """Thread for analyzing dataset."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    def run(self):
        """Analyze dataset in background thread."""
        try:
            stats = {
                'total_images': 0,
                'annotated_images': 0,
                'background_images': 0,
                'total_labels': 0,
                'image_paths': [],
                'label_paths': {}
            }

            # Find all images
            self.progress.emit("Scanning for images...")
            for ext in self.image_extensions:
                paths = list(Path(self.dataset_path).rglob(f"*{ext}"))
                stats['image_paths'].extend(paths)

            stats['total_images'] = len(stats['image_paths'])
            if stats['total_images'] == 0:
                self.error.emit("No images found in dataset")
                return

            # Find matching labels
            self.progress.emit("Finding label files...")
            unique_labels = set()
            empty_label_files = 0
            
            for img_path in stats['image_paths']:
                # Check for label in same directory
                label_path = img_path.with_suffix('.txt')
                # Check in labels directory if not found
                if not label_path.exists() and 'images' in str(img_path.parent).lower():
                    label_dir = str(img_path.parent).lower().replace('images', 'labels')
                    label_path = Path(label_dir) / f"{img_path.stem}.txt"

                if label_path.exists():
                    # Check if label file is empty (background image)
                    with open(label_path) as f:
                        if not f.read().strip():
                            empty_label_files += 1
                            continue
                        
                    stats['label_paths'][img_path] = label_path
                    stats['annotated_images'] += 1
                    # Count unique class IDs
                    try:
                        with open(label_path) as f:
                            for line in f:
                                class_id = int(float(line.split()[0]))
                                unique_labels.add(class_id)
                    except Exception as e:
                        logging.warning(f"Error reading {label_path}: {e}")
                        
            stats['background_images'] = stats['total_images'] - stats['annotated_images'] + empty_label_files
            stats['total_labels'] = len(unique_labels)
            
            self.finished.emit(stats)

        except Exception as e:
            self.error.emit(str(e))

class ThumbnailWidget(QLabel):
    """Widget for displaying image thumbnails with bounding boxes."""
    
    clicked = pyqtSignal(str, str)  # Signals image_path and label_path
    
    def __init__(self, image_path, label_path=None, parent=None):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setMaximumSize(200, 200) 
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: white;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image_path = str(image_path)
        self.label_path = str(label_path) if label_path else None
        
        # Load and process image
        try:
            # Use PIL for initial loading and drawing boxes
            with Image.open(image_path) as pil_img:
                # Convert to RGB to ensure consistent color space
                pil_img = pil_img.convert('RGB')
                
                # Draw bounding boxes if label exists
                if label_path:
                    self.draw_boxes(pil_img, label_path)
                
                # Convert to QPixmap for display
                img_array = np.array(pil_img)
                
                q_img = QImage(
                    img_array.data,
                    img_array.shape[1],
                    img_array.shape[0],
                    img_array.shape[1] * 3,
                    QImage.Format.Format_RGB888
                )
                
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(
                    200, 200,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.setPixmap(scaled_pixmap)
                
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            self.setText("Error loading image")

    def mousePressEvent(self, event):
        """Handle mouse click events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_path, self.label_path or "")
            
    def draw_boxes(self, image, label_path):
        """Draw bounding boxes on image."""
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        
        try:
            with open(label_path) as f:
                lines = f.readlines()
                    
                for line in lines:  # Use lines instead of f to avoid reading twice
                    try:
                        # Parse YOLO format: class_id, x_center, y_center, width, height
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        
                        # Convert to pixel coordinates
                        x_center_px = x_center * img_width
                        y_center_px = y_center * img_height
                        width_px = width * img_width
                        height_px = height * img_height
                        
                        # Calculate box coordinates
                        left = x_center_px - width_px / 2
                        top = y_center_px - height_px / 2
                        right = x_center_px + width_px / 2
                        bottom = y_center_px + height_px / 2
                        
                        # Draw box with color based on class
                        color = self.get_class_color(int(class_id))
                        draw.rectangle([left, top, right, bottom], outline=color, width=3)
                        
                    except Exception as e:
                        logging.warning(f"Error parsing label line: {e}")
                        continue
                        
        except Exception as e:
            logging.error(f"Error reading label file {label_path}: {e}")

    def get_class_color(self, class_id):
        """Get color for class ID."""
        colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FFA500",  # Orange
            "#800080"   # Purple
        ]
        return colors[class_id % len(colors)]

class FullscreenImageViewer(QDialog):
    """Dialog for fullscreen image viewing."""
    
    def __init__(self, image_paths, current_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Viewer")
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        
        self.image_paths = image_paths
        self.current_index = current_index
        
        layout = QVBoxLayout(self)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("Previous (←)")
        self.prev_btn.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_btn)
        
        close_btn = QPushButton("Close (Esc)")
        close_btn.clicked.connect(self.close)
        nav_layout.addWidget(close_btn)
        
        self.next_btn = QPushButton("Next (→)")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
        self.load_current_image()
        
    def load_current_image(self):
        """Load and display current image."""
        image_path = self.image_paths[self.current_index]
        
        # Load image with PIL to draw boxes
        with Image.open(image_path) as pil_img:
            # Find matching label file
            label_path = Path(image_path).with_suffix('.txt')
            if not label_path.exists():
                # Check in labels directory
                if 'images' in str(Path(image_path).parent).lower():
                    label_dir = str(Path(image_path).parent).lower().replace('images', 'labels')
                    label_path = Path(label_dir) / f"{Path(image_path).stem}.txt"
            
            # Draw boxes if label exists
            if label_path.exists():
                self.draw_boxes(pil_img, label_path)
            
            # Convert to QPixmap
            img_array = np.array(pil_img)
            height, width = img_array.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
        
        # Scale to fit screen while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.width() - 100,
            self.height() - 100,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        
        # Update navigation buttons
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.image_paths) - 1)
        
    def show_previous(self):
        """Show previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
            
    def show_next(self):
        """Show next image."""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_current_image()
            
    def keyPressEvent(self, event):
        """Handle keyboard navigation."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_Left:
            self.show_previous()
        elif event.key() == Qt.Key.Key_Right:
            self.show_next()
        else:
            super().keyPressEvent(event)

    def draw_boxes(self, image, label_path):
        """Draw bounding boxes on image."""
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        
        try:
            with open(label_path) as f:
                for line in f:
                    try:
                        # Parse YOLO format: class_id, x_center, y_center, width, height
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        
                        # Convert to pixel coordinates
                        x_center_px = x_center * img_width
                        y_center_px = y_center * img_height
                        width_px = width * img_width
                        height_px = height * img_height
                        
                        # Calculate box coordinates
                        left = x_center_px - width_px / 2
                        top = y_center_px - height_px / 2
                        right = x_center_px + width_px / 2
                        bottom = y_center_px + height_px / 2
                        
                        # Draw box with color based on class
                        color = self.get_class_color(int(class_id))
                        draw.rectangle([left, top, right, bottom], outline=color, width=3)
                        
                    except Exception as e:
                        logging.warning(f"Error parsing label line: {e}")
                        continue
                        
        except Exception as e:
            logging.error(f"Error reading label file {label_path}: {e}")

    def get_class_color(self, class_id):
        """Get color for class ID."""
        colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FFA500",  # Orange
            "#800080"   # Purple
        ]
        return colors[class_id % len(colors)]

class DatasetViewerApp(QMainWindow):
    """Main window for the dataset viewer application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Viewer")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
        # Initialize UI
        self.init_ui()
        
        # Dataset state
        self.dataset_path = None
        self.current_page = 1
        self.images_per_page = 24
        self.dataset_stats = None
        
    def init_ui(self):
        """Initialize the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Controls
        controls = QHBoxLayout()
        
        self.path_label = QLabel("Dataset: Not selected")
        controls.addWidget(self.path_label)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_dataset)
        controls.addWidget(browse_btn)
        
        layout.addLayout(controls)
        
        # Statistics
        self.stats_layout = QHBoxLayout()
        layout.addLayout(self.stats_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Scroll area for thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        scroll.setWidget(self.grid_widget)
        layout.addWidget(scroll)
        
        # Navigation
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.previous_page)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.page_label = QLabel("Page 1")
        nav_layout.addWidget(self.page_label)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
    def browse_dataset(self):
        """Open file dialog to select dataset directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory"
        )
        if path:
            self.dataset_path = path
            self.path_label.setText(f"Dataset: {path}")
            self.analyze_dataset()
            
    def analyze_dataset(self):
        """Start dataset analysis in background thread."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        
        self.analyzer = DatasetAnalyzer(self.dataset_path)
        self.analyzer.progress.connect(self.update_progress)
        self.analyzer.finished.connect(self.analysis_complete)
        self.analyzer.error.connect(self.show_error)
        self.analyzer.start()
        
    def update_progress(self, message):
        """Update progress bar message."""
        self.progress_bar.setFormat(message)
        
    def analysis_complete(self, stats):
        """Handle completed dataset analysis."""
        self.dataset_stats = stats
        self.progress_bar.setVisible(False)
        
        # Update statistics display
        self.update_stats()
        
        # Load first page
        self.current_page = 1
        self.load_page()
        
        # Enable navigation if multiple pages
        total_pages = -(-(len(stats['image_paths']) // self.images_per_page))
        self.next_btn.setEnabled(total_pages > 1)
        
    def update_stats(self):
        """Update statistics display."""
        # Clear existing stats
        for i in reversed(range(self.stats_layout.count())): 
            self.stats_layout.itemAt(i).widget().setParent(None)
            
        # Add stat cards
        stats = [
            ("Total Images", self.dataset_stats['total_images']),
            ("Annotated Images", self.dataset_stats['annotated_images']),
            ("Background Images", self.dataset_stats['background_images']),
            ("Unique Labels", self.dataset_stats['total_labels'])
        ]
        
        for label, value in stats:
            card = QWidget()
            card.setStyleSheet("""
                background-color: white;
                color: #333;
                border-radius: 5px;
                padding: 10px;
                margin: 5px;
                border: 1px solid #ddd;
            """)
            card_layout = QVBoxLayout(card)
            
            value_label = QLabel(str(value))
            value_label.setStyleSheet("""
                font-size: 24px;
                font-weight: bold;
                color: #4a90e2;
                margin-bottom: 5px;
            """)
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(value_label)
            
            desc_label = QLabel(label)
            desc_label.setStyleSheet("color: #666;")
            desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(desc_label)
            
            self.stats_layout.addWidget(card)
            
    def load_page(self):
        """Load current page of images."""
        if not self.dataset_stats:
            return
            
        # Clear existing thumbnails
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)
            
        # Calculate page range
        start_idx = (self.current_page - 1) * self.images_per_page
        end_idx = start_idx + self.images_per_page
        page_images = self.dataset_stats['image_paths'][start_idx:end_idx]
        
        # Add thumbnails to grid
        for i, image_path in enumerate(page_images):
            label_path = self.dataset_stats['label_paths'].get(image_path)
            thumbnail = ThumbnailWidget(image_path, label_path)
            thumbnail.clicked.connect(self.show_fullscreen)
            row = i // 6
            col = i % 6
            self.grid_layout.addWidget(thumbnail, row, col)
            
        # Update navigation
        total_pages = -(-(len(self.dataset_stats['image_paths']) // self.images_per_page))
        self.page_label.setText(f"Page {self.current_page} of {total_pages}")
        self.prev_btn.setEnabled(self.current_page > 1)
        self.next_btn.setEnabled(self.current_page < total_pages)
        
    def previous_page(self):
        """Go to previous page."""
        if self.current_page > 1:
            self.current_page -= 1
            self.load_page()
            
    def next_page(self):
        """Go to next page."""
        total_pages = -(-(len(self.dataset_stats['image_paths']) // self.images_per_page))
        if self.current_page < total_pages:
            self.current_page += 1
            self.load_page()
            
    def show_fullscreen(self, image_path, label_path):
        """Show image in fullscreen."""
        try:
            # Find index of clicked image
            current_index = self.dataset_stats['image_paths'].index(Path(image_path))
            
            # Create and show fullscreen viewer
            viewer = FullscreenImageViewer(
                [str(p) for p in self.dataset_stats['image_paths']],
                current_index,
                self
            )
            viewer.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open image: {str(e)}")
            
    def show_error(self, message):
        """Show error message."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", message)