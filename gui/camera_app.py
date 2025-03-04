import sys
import os
import cv2
import logging
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QDialog, QMenu,
    QLabel, QComboBox, QFileDialog, QMessageBox, QProgressBar, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QFileSystemWatcher, QMutex, QMutexLocker
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QFont, QColor, QPalette, QAction

import glob

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

try:
    from nxt_camera import NxtCamera
    NXT_AVAILABLE = True
except ImportError:
    NXT_AVAILABLE = False
    logger.warning("NXT camera support not available")

class CameraSelectionDialog(QDialog):
    """Dialog for selecting a camera device."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kamera auswählen")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Camera list
        self.camera_list = QListWidget()
        layout.addWidget(self.camera_list)
        
        # Populate camera list
        self.available_cameras = self.find_cameras()
        for i, (name, _) in enumerate(self.available_cameras):
            item = QListWidgetItem(f"Kamera {i}: {name}")
            self.camera_list.addItem(item)
        
        if not self.available_cameras:
            self.camera_list.addItem("Keine Kameras gefunden")
            self.camera_list.setEnabled(False)
        else:
            self.camera_list.setCurrentRow(0)
        
        # Buttons
        button_layout = QHBoxLayout()
        select_btn = QPushButton("Auswählen")
        select_btn.clicked.connect(self.accept)
        select_btn.setEnabled(bool(self.available_cameras))
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(select_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def find_cameras(self):
        """Find available camera devices."""
        available_cameras = []
        # Try opening each camera index
        for i in range(10):  # Check first 10 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera name if possible
                name = cap.getBackendName()
                available_cameras.append((name, i))
                cap.release()
        return available_cameras
    
    def get_selected_camera(self):
        """Get the selected camera index."""
        if not self.available_cameras:
            return None
        current_row = self.camera_list.currentRow()
        if current_row >= 0:
            return self.available_cameras[current_row][1]
        return None

class CameraThread(QThread):
    """Thread for camera operations."""
    
    frame_ready = pyqtSignal(QImage)
    error = pyqtSignal(str)
    
    def __init__(self, camera_type='usb', camera_id=0, nxt_config=None):
        super().__init__()
        self.camera_type = camera_type
        self.camera_id = int(camera_id)  # Ensure camera_id is int
        self.nxt_config = nxt_config
        self.running = False
        self.camera = None
        self.current_frame = None  # Store the original frame
        self.paused = False
        self._lock = QMutex()  # Add mutex for thread safety
        
    def run(self):
        """Main camera loop."""
        try:
            with QMutexLocker(self._lock):
                if self.camera_type == 'usb':
                    self.camera = cv2.VideoCapture(self.camera_id)
                    if not self.camera.isOpened():
                        raise Exception(f"Konnte USB-Kamera {self.camera_id} nicht öffnen")
                else:
                    if not NXT_AVAILABLE:
                        raise Exception("NXT camera support not available")
                    self.camera = NxtCamera(
                        ip=self.nxt_config['ip'],
                        username=self.nxt_config['username'],
                        password=self.nxt_config['password'],
                        ssl=self.nxt_config['ssl']
                    )
            
            self.running = True
            while self.running:
                if self.paused:
                    self.msleep(100)
                    continue
                    
                with QMutexLocker(self._lock):
                    if self.camera_type == 'usb':
                        ret, frame = self.camera.read()
                        if not ret:
                            raise Exception("Konnte kein Bild von der USB-Kamera empfangen")
                    else:
                        frame = self.camera.get_frame()
                    
                    # Store original frame
                    self.current_frame = frame.copy()
                    
                    # Convert frame to QImage
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(
                        rgb_frame.data,
                        w, h,
                        bytes_per_line,
                        QImage.Format.Format_RGB888
                    )
                    self.frame_ready.emit(qt_image)
                
        except Exception as e:
            self.error.emit(f"Kamera-Fehler: {str(e)}")
        finally:
            if self.camera:
                if self.camera_type == 'usb':
                    self.camera.release()
                else:
                    self.camera.disconnect()
    
    @property
    def is_running(self):
        """Check if camera thread is running."""
        return self.running and not self.paused
    
    def stop(self):
        """Stop camera thread."""
        try:
            with QMutexLocker(self._lock):
                self.running = False
                self.paused = True  # Pause first to avoid frame grab errors
            
            # Wait for thread to finish with timeout
            if not self.wait(500):  # 500ms timeout
                logger.warning("Camera thread did not stop gracefully, forcing termination")
                self.terminate()
                self.wait()
            
            # Final cleanup
            with QMutexLocker(self._lock):
                if self.camera:
                    logger.info("Releasing camera resources...")
                    if self.camera_type == 'usb':
                        self.camera.release()
                    else:
                        self.camera.disconnect()
                self.camera = None
                
        except Exception as e:
            logger.error(f"Error stopping camera thread: {e}")
            # Ensure thread is terminated even if error occurs
            self.terminate()
            self.wait()
            raise  # Re-raise exception to be handled by caller

class ThumbnailWidget(QLabel):
    """Widget for displaying image thumbnails."""
    
    clicked = pyqtSignal(str)  # Signal for image click
    delete_requested = pyqtSignal(str)  # Signal for delete request
    
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setMinimumSize(150, 150)
        self.setMaximumSize(150, 150)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: white;
            }
            QLabel:hover {
                border-color: #4a90e2;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Load and display thumbnail
        image = cv2.imread(str(image_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                140, 140,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
    
    def mousePressEvent(self, event):
        """Handle mouse click events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_path)

    def show_context_menu(self, position):
        """Show context menu on right click."""
        menu = QMenu()
        delete_action = QAction("Bild löschen", self)
        delete_action.triggered.connect(lambda: self.delete_requested.emit(self.image_path))
        menu.addAction(delete_action)
        menu.exec(self.mapToGlobal(position))

class CameraApp(QMainWindow):
    """Main window for camera application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Capture")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)  # Prevent auto-deletion
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
        # Initialize attributes
        self.camera_thread = None
        self.output_dir = None
        
        # Split into left (camera) and right (gallery) panels
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel (camera)
        camera_panel = QWidget()
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setSpacing(10)
        
        # Camera selection
        camera_controls = QHBoxLayout()
        
        camera_label = QLabel("Camera Type:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("USB Camera")
        if NXT_AVAILABLE:
            self.camera_combo.addItem("IDS NXT Camera")
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_camera)
        
        camera_controls.addWidget(camera_label)
        camera_controls.addWidget(self.camera_combo)
        camera_controls.addWidget(self.connect_btn)
        camera_layout.addLayout(camera_controls)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        
        self.dir_label = QLabel("Output Directory: Not selected")
        dir_btn = QPushButton("Browse")
        dir_btn.clicked.connect(self.browse_directory)
        dir_btn.setMinimumWidth(100)
        
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(dir_btn)
        camera_layout.addLayout(dir_layout)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        camera_layout.addWidget(self.video_label)
        
        # Capture button
        self.capture_btn = QPushButton("📸 Schnappschuss (Leertaste)")
        capture_font = QFont()
        capture_font.setPointSize(14)
        capture_font.setBold(True)
        self.capture_btn.setFont(capture_font)
        self.capture_btn.setMinimumHeight(60)
        self.capture_btn.setStyleSheet("""
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
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        camera_layout.addWidget(self.capture_btn)
        
        # Add camera panel to main layout
        main_layout.addWidget(camera_panel, stretch=2)
        
        # Right panel (gallery)
        gallery_panel = QWidget()
        gallery_layout = QVBoxLayout(gallery_panel)
        
        gallery_label = QLabel("Recent Captures")
        gallery_font = QFont()
        gallery_font.setPointSize(12)
        gallery_font.setBold(True)
        gallery_label.setFont(gallery_font)
        gallery_layout.addWidget(gallery_label)
        
        # Scroll area for thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.gallery_widget = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_widget)
        self.gallery_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.gallery_widget)
        gallery_layout.addWidget(scroll)

        # Initialize file system watcher
        self.watcher = QFileSystemWatcher()
        self.watcher.directoryChanged.connect(self.refresh_gallery)
        
        # Add gallery panel to main layout
        main_layout.addWidget(gallery_panel, stretch=1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def on_camera_changed(self, camera_type):
        """Handle camera type change."""
        if self.camera_thread and self.camera_thread.running:
            self.toggle_camera()  # Disconnect current camera
    
    def toggle_camera(self):
        """Connect or disconnect camera."""
        if self.camera_thread and self.camera_thread.running:
            try:
                # Stop camera thread first
                logger.info("Stopping camera thread...")
                self.camera_thread.stop()
                
                # Reset UI state
                self.camera_thread = None
                self.connect_btn.setText("Connect")
                self.capture_btn.setEnabled(False)
                self.statusBar().showMessage("Camera disconnected")
                self.video_label.clear()
                self.video_label.setStyleSheet("background-color: black;")
                self.capture_btn.setFocus()  # Reset focus to capture button
                
            except Exception as e:
                logger.error(f"Error disconnecting camera: {e}")
                QMessageBox.critical(self, "Fehler", f"Fehler beim Trennen der Kamera: {str(e)}")
                return
                
        else:
            # Connect
            try:
                camera_type = 'nxt' if "NXT" in self.camera_combo.currentText() else 'usb'
                
                if camera_type == 'nxt':
                    # Show NXT camera config dialog
                    nxt_config = {
                        'ip': '169.254.100.99',
                        'username': 'admin',
                        'password': 'Flex',
                        'ssl': False
                    }
                    camera_id = 0
                else:
                    # Show USB camera selection dialog
                    camera_dialog = CameraSelectionDialog(self)
                    if camera_dialog.exec() != QDialog.DialogCode.Accepted:
                        return
                    camera_id = camera_dialog.get_selected_camera()
                    if camera_id is None:
                        QMessageBox.warning(self, "Warnung", "Keine Kamera ausgewählt")
                        return
                    nxt_config = None
                
                self.camera_thread = CameraThread(
                    camera_type=camera_type,
                    camera_id=camera_id,
                    nxt_config=nxt_config
                )
                self.camera_thread.frame_ready.connect(self.update_frame)
                self.camera_thread.error.connect(self.handle_error)
                self.camera_thread.start()
                
                self.connect_btn.setText("Disconnect")
                self.capture_btn.setEnabled(True)
                self.statusBar().showMessage("Camera connected")
                
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Fehler beim Verbinden mit der Kamera: {str(e)}")
    
    def update_frame(self, image):
        """Update video display with new frame."""
        scaled_pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def handle_error(self, message):
        """Handle camera errors."""
        try:
            QMessageBox.critical(self, "Kamera-Fehler", message)
            if self.camera_thread and self.camera_thread.running:
                self.toggle_camera()  # Disconnect on error
        except Exception as e:
            logger.error(f"Error handling camera error: {e}")
    
    def browse_directory(self):
        """Open file dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            try:
                # Test write permissions
                test_file = Path(dir_path) / ".test"
                test_file.touch()
                test_file.unlink()
                
                self.output_dir = dir_path
                self.dir_label.setText(f"Output Directory: {dir_path}")
                
                # Set up directory monitoring
                if self.output_dir in self.watcher.directories():
                    self.watcher.removePath(self.output_dir)
                self.watcher.addPath(self.output_dir)
                
                # Load existing images
                self.refresh_gallery()
                
                self.statusBar().showMessage("Output directory set")
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Cannot write to selected directory: {str(e)}"
                )
    
    def capture_image(self):
        """Capture and save current frame."""
        if not self.output_dir:
            QMessageBox.critical(
                self,
                "Fehler",
                "Bitte wählen Sie zuerst ein Zielverzeichnis aus"
            )
            return
        
        try:
            # Pause camera thread while capturing
            self.camera_thread.paused = True
            
            # Get original resolution frame
            if not self.camera_thread or not self.camera_thread.current_frame is not None:
                raise Exception("No frame available")
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = Path(self.output_dir) / filename
            
            # Save original resolution image
            success = cv2.imwrite(
                str(filepath),
                self.camera_thread.current_frame
            )
            if not success:
                raise Exception("Failed to save image")
            
            self.statusBar().showMessage(f"Image saved: {filename}")
            # Resume camera thread
            self.camera_thread.paused = False
            
            # Add thumbnail to gallery
            self.add_thumbnail(str(filepath))
            
            # Refresh gallery to ensure proper order
            self.refresh_gallery()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save image: {str(e)}"
            )
            if self.camera_thread:
                self.camera_thread.paused = False
    
    def add_thumbnail(self, image_path):
        """Add thumbnail to gallery."""
        thumbnail = ThumbnailWidget(image_path)
        thumbnail.clicked.connect(lambda p=image_path: self.show_full_image(p))
        thumbnail.delete_requested.connect(self.delete_image)
        self.gallery_layout.insertWidget(0, thumbnail)
    
    def refresh_gallery(self):
        """Refresh gallery contents."""
        if not self.output_dir:
            return
            
        # Clear existing thumbnails
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Load all images from directory
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(glob.glob(os.path.join(self.output_dir, f"*{ext}")))
        
        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Add thumbnails
        for image_path in image_files:
            self.add_thumbnail(image_path)
    
    def show_full_image(self, image_path):
        """Show image in full size."""
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", "Image file no longer exists")
            self.refresh_gallery()
            return
            
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create fullscreen dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Full Image")
            dialog.setWindowState(Qt.WindowState.WindowFullScreen)
            dialog.setStyleSheet("background-color: black;")
            
            # Add image to dialog
            layout = QVBoxLayout(dialog)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                dialog.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            layout.addWidget(label)
            
            # Add close button at the bottom
            close_layout = QHBoxLayout()
            
            close_btn = QPushButton("Close (Esc)")
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #666666;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #888888;
                }
            """)
            close_btn.clicked.connect(dialog.close)
            close_layout.addWidget(close_btn)
            
            layout.addLayout(close_layout)
            
            dialog.exec()
            
    def delete_image(self, image_path):
        """Delete image file."""
        try:
            # Use StandardButton enum for message box buttons
            response = QMessageBox.question(
                self,
                "Confirm Deletion",
                "Are you sure you want to delete this image?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if response == QMessageBox.StandardButton.Yes:
                os.remove(image_path)
                self.statusBar().showMessage(f"Deleted: {os.path.basename(image_path)}")
                self.refresh_gallery()
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to delete image: {str(e)}"
            )
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events."""
        if event.key() == Qt.Key.Key_Space:
            if self.capture_btn.isEnabled() and self.capture_btn.hasFocus():
                self.capture_image()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        """Handle window show event."""
        super().showEvent(event)
        # Set initial focus to capture button
        self.capture_btn.setFocus()
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            if self.camera_thread and self.camera_thread.running:
                self.camera_thread.stop()
                self.camera_thread = None
            self.hide()  # Hide window instead of closing
            event.ignore()  # Prevent window from being destroyed
        except Exception as e:
            logger.error(f"Error in closeEvent: {e}")
            event.ignore()

    def hideEvent(self, event):
        """Handle window hide event."""
        super().hideEvent(event)
        # Ensure camera is disconnected when window is hidden
        if self.camera_thread and self.camera_thread.running:
            try:
                self.toggle_camera()
            except Exception as e:
                logger.error(f"Error disconnecting camera on hide: {e}")