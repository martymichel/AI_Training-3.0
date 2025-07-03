"""Main application module for camera capture."""

import sys
import os
import cv2
import logging
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QDialog, QMenu,
    QLabel, QComboBox, QFileDialog, QMessageBox, QProgressBar, QListWidget, QListWidgetItem,
    QApplication, QLineEdit, QFormLayout, QCheckBox, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QFileSystemWatcher, QMutex, QMutexLocker
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QFont, QColor, QPalette, QAction

import glob
import logging

from project_manager import ProjectManager, WorkflowStep

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

# Add IDS Peak API support
try:
    import ids_peak.ids_peak as ids_peak
    import ids_peak_ipl.ids_peak_ipl as ids_ipl
    import ids_peak.ids_peak_ipl_extension as ids_ipl_extension
    IDS_PEAK_AVAILABLE = True
except ImportError:
    IDS_PEAK_AVAILABLE = False
    logger.warning("IDS Peak API nicht verfügbar. Installiere IDS Peak API und IDS Peak Software.")

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
        
        # Populate camera list with various camera types
        self.available_cameras = []
        
        # Add USB cameras
        usb_cameras = self.find_usb_cameras()
        self.available_cameras.extend([("usb", name, idx) for name, idx in usb_cameras])
        
        # Add IDS Peak cameras if available
        if IDS_PEAK_AVAILABLE:
            ids_cameras = self.find_ids_peak_cameras()
            self.available_cameras.extend([("ids_peak", name, idx) for name, idx in ids_cameras])
        
        # Add NXT cameras if available
        if NXT_AVAILABLE:
            self.available_cameras.append(("nxt", "IDS NXT Camera", 0))
        
        # Add camera items to the list
        for i, (cam_type, name, _) in enumerate(self.available_cameras):
            if cam_type == "usb":
                item = QListWidgetItem(f"USB-Kamera {i}: {name}")
            elif cam_type == "ids_peak":
                item = QListWidgetItem(f"IDS Peak: {name}")
            elif cam_type == "nxt":
                item = QListWidgetItem(f"IDS NXT: {name}")
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
class NxtConfigDialog(QDialog):
    """Dialog to configure IDS NXT connection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IDS NXT Konfiguration")

        layout = QFormLayout(self)

        self.ip_edit = QLineEdit("192.168.1.99")
        self.user_edit = QLineEdit("admin")
        self.pw_edit = QLineEdit("Flex")
        self.pw_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.ssl_check = QCheckBox("SSL verwenden")

        layout.addRow("IP-Adresse:", self.ip_edit)
        layout.addRow("Benutzer:", self.user_edit)
        layout.addRow("Passwort:", self.pw_edit)
        layout.addRow(self.ssl_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self):
        """Return the entered configuration."""
        return {
            'ip': self.ip_edit.text().strip(),
            'username': self.user_edit.text().strip(),
            'password': self.pw_edit.text(),
            'ssl': self.ssl_check.isChecked(),
        }
    
    def find_usb_cameras(self):
        """Find available USB camera devices."""
        available_cameras = []
        # Try opening each camera index
        for i in range(10):  # Check first 10 indexes
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Get camera name if possible
                    name = cap.getBackendName()
                    available_cameras.append((name, i))
                    cap.release()
            except Exception as e:
                logger.warning(f"Error checking USB camera {i}: {e}")
        return available_cameras
    
    def find_ids_peak_cameras(self):
        """Find available IDS Peak camera devices."""
        try:
            # Initialize IDS Peak library
            ids_peak.Library.Initialize()
            
            # Get device manager and update device list
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            
            # Get available devices
            device_descriptors = device_manager.Devices()
            
            # Return list of available cameras
            cameras = [(device.DisplayName(), i) for i, device in enumerate(device_descriptors)]
            
            return cameras
        except Exception as e:
            logger.warning(f"Error finding IDS Peak cameras: {e}")
            return []
    
    def get_selected_camera(self):
        """Get the selected camera info."""
        if not self.available_cameras:
            return None
        current_row = self.camera_list.currentRow()
        if current_row >= 0:
            return self.available_cameras[current_row]
        return None

class IDSPeakCameraThread(QThread):
    """Thread for IDS Peak camera operations."""
    
    frame_ready = pyqtSignal(QImage)
    error = pyqtSignal(str)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.device = None
        self.datastream = None
        self.remote_device_nodemap = None
        self.current_frame = None
        self.paused = False
        self._lock = QMutex()
    
    def run(self):
        """Main camera loop."""
        try:
            with QMutexLocker(self._lock):
                # Initialize IDS Peak API
                ids_peak.Library.Initialize()
                
                # Get device manager and update device list
                device_manager = ids_peak.DeviceManager.Instance()
                device_manager.Update()
                
                # Get available devices
                device_descriptors = device_manager.Devices()
                if not device_descriptors:
                    raise Exception("No IDS Peak cameras found")
                
                # Open selected camera
                if self.camera_id < len(device_descriptors):
                    device_descriptor = device_descriptors[self.camera_id]
                else:
                    device_descriptor = device_descriptors[0]
                
                self.device = device_descriptor.OpenDevice(ids_peak.DeviceAccessType_Control)
                self.remote_device_nodemap = self.device.RemoteDevice().NodeMaps()[1]
                
                # Configure camera for continuous acquisition
                self.remote_device_nodemap.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
                self.remote_device_nodemap.FindNode("TriggerSource").SetCurrentEntry("Software")
                self.remote_device_nodemap.FindNode("TriggerMode").SetCurrentEntry("Off")
                
                # Prepare datastream
                self.datastream = self.device.DataStreams()[0].OpenDataStream()
                payload_size = self.remote_device_nodemap.FindNode("PayloadSize").Value()
                
                # Allocate buffers
                for i in range(self.datastream.NumBuffersAnnouncedMinRequired()):
                    buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
                    self.datastream.QueueBuffer(buffer)
                
                # Start acquisition
                self.datastream.StartAcquisition()
                self.remote_device_nodemap.FindNode("AcquisitionStart").Execute()
                self.remote_device_nodemap.FindNode("AcquisitionStart").WaitUntilDone()
            
            self.running = True
            
            while self.running:
                if self.paused:
                    self.msleep(100)
                    continue
                
                try:
                    with QMutexLocker(self._lock):
                        # Capture frame from camera
                        buffer = self.datastream.WaitForFinishedBuffer(1000)
                        
                        # Convert buffer to image
                        raw_image = ids_ipl_extension.BufferToImage(buffer)
                        color_image = raw_image.ConvertTo(ids_ipl.PixelFormatName_RGB8)
                        
                        # Queue buffer for next frame
                        self.datastream.QueueBuffer(buffer)
                        
                        # Convert to NumPy array
                        frame = color_image.get_numpy_3D()
                        
                        # Convert to BGR for OpenCV
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # Store frame
                        self.current_frame = frame.copy()
                        
                        # Convert to QImage for display
                        h, w, ch = frame.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data,
                            w, h,
                            bytes_per_line,
                            QImage.Format.Format_RGB888
                        ).copy()
                        
                        # Emit frame
                        self.frame_ready.emit(qt_image)
                except Exception as e:
                    logger.error(f"Error capturing IDS Peak frame: {e}")
                    # Don't emit error here, just log it and continue
                    self.msleep(100)
        
        except Exception as e:
            logger.error(f"IDS Peak camera error: {e}")
            self.error.emit(f"IDS Peak camera error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up IDS Peak resources."""
        try:
            with QMutexLocker(self._lock):
                # Stop acquisition
                if self.datastream:
                    try:
                        self.datastream.StopAcquisition()
                    except Exception as e:
                        logger.error(f"Error stopping datastream acquisition: {e}")
                
                if self.remote_device_nodemap:
                    try:
                        self.remote_device_nodemap.FindNode("AcquisitionStop").Execute()
                    except Exception as e:
                        logger.error(f"Error executing acquisition stop: {e}")
                
                # Release resources
                self.datastream = None
                self.device = None
                self.remote_device_nodemap = None
                
                # Close IDS Peak library
                try:
                    ids_peak.Library.Close()
                except Exception as e:
                    logger.error(f"Error closing IDS Peak library: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up IDS Peak camera: {e}")
    
    def stop(self):
        """Stop camera thread."""
        try:
            with QMutexLocker(self._lock):
                self.running = False
                self.paused = False

            if not self.wait(1000):  # allow more time for a clean shutdown
                logger.warning("Camera thread did not stop gracefully, forcing termination")
                self.terminate()
                self.wait()

        finally:
            self.cleanup()

class CameraThread(QThread):
    """Thread for camera operations."""
    
    frame_ready = pyqtSignal(QImage)
    error = pyqtSignal(str)
    
    def __init__(self, camera_type='usb', camera_id=0, nxt_config=None):
        super().__init__()
        self.camera_type = camera_type
        self.camera_id = int(camera_id) if camera_id is not None else 0  # Ensure camera_id is int
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
                elif self.camera_type == 'nxt':
                    if not NXT_AVAILABLE:
                        raise Exception("NXT camera support not available")
                    self.camera = NxtCamera(
                        ip=self.nxt_config['ip'],
                        username=self.nxt_config['username'],
                        password=self.nxt_config['password'],
                        ssl=self.nxt_config['ssl']
                    )
                else:
                    raise Exception(f"Unsupported camera type: {self.camera_type}")
            
            self.running = True
            while self.running:
                if self.paused:
                    self.msleep(100)
                    continue
                    
                try:
                    with QMutexLocker(self._lock):
                        if self.camera_type == 'usb':
                            ret, frame = self.camera.read()
                            if not ret:
                                raise Exception("Konnte kein Bild von der USB-Kamera empfangen")
                        elif self.camera_type == 'nxt':
                            frame = self.camera.get_frame()
                        else:
                            raise Exception(f"Unsupported camera type: {self.camera_type}")
                        
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
                        ).copy()
                        self.frame_ready.emit(qt_image)
                except Exception as e:
                    logger.error(f"Error capturing frame: {e}")
                    # Don't emit error here, just log it and continue
                    self.msleep(100)
                
        except Exception as e:
            self.error.emit(f"Kamera-Fehler: {str(e)}")
        finally:
            if self.camera:
                if self.camera_type == 'usb':
                    self.camera.release()
                elif self.camera_type == 'nxt':
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
                self.paused = False

            if not self.wait(1000):
                logger.warning("Camera thread did not stop gracefully, forcing termination")
                self.terminate()
                self.wait()

        finally:
            with QMutexLocker(self._lock):
                if self.camera:
                    logger.info("Releasing camera resources...")
                    if self.camera_type == 'usb':
                        self.camera.release()
                    elif self.camera_type == 'nxt':
                        self.camera.disconnect()
                self.camera = None

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
        self.ids_peak_thread = None
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
        if IDS_PEAK_AVAILABLE:
            self.camera_combo.addItem("IDS Peak Camera")
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
        if (self.camera_thread and self.camera_thread.running) or \
           (self.ids_peak_thread and self.ids_peak_thread.running):
            self.toggle_camera()  # Disconnect current camera
    
    def toggle_camera(self):
        """Connect or disconnect camera."""
        if (self.camera_thread and self.camera_thread.running) or \
           (self.ids_peak_thread and self.ids_peak_thread.running):
            try:
                # Stop camera thread first
                logger.info("Stopping camera thread...")
                if self.camera_thread:
                    self.camera_thread.stop()
                    self.camera_thread = None
                if self.ids_peak_thread:
                    self.ids_peak_thread.stop()
                    self.ids_peak_thread = None
                
                # Reset UI state
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
                camera_type = self.camera_combo.currentText()
                
                if "IDS NXT" in camera_type:
                    dialog = NxtConfigDialog(self)
                    if dialog.exec() != QDialog.DialogCode.Accepted:
                        return
                    nxt_config = dialog.get_config()
                    self.camera_thread = CameraThread(
                        camera_type='nxt',
                        camera_id=0,
                        nxt_config=nxt_config
                    )
                    self.camera_thread.frame_ready.connect(self.update_frame)
                    self.camera_thread.error.connect(self.handle_error)
                    self.camera_thread.start()
                elif "IDS Peak" in camera_type:
                    # Show IDS Peak camera selection dialog
                    if IDS_PEAK_AVAILABLE:
                        camera_dialog = CameraSelectionDialog(self)
                        if camera_dialog.exec() != QDialog.DialogCode.Accepted:
                            return
                        camera_info = camera_dialog.get_selected_camera()
                        if camera_info is None or camera_info[0] != "ids_peak":
                            QMessageBox.warning(self, "Warnung", "Keine IDS Peak Kamera ausgewählt")
                            return
                        
                        # Start IDS Peak camera thread
                        self.ids_peak_thread = IDSPeakCameraThread(
                            camera_id=camera_info[2]
                        )
                        self.ids_peak_thread.frame_ready.connect(self.update_frame)
                        self.ids_peak_thread.error.connect(self.handle_error)
                        self.ids_peak_thread.start()
                    else:
                        QMessageBox.critical(self, "Fehler", "IDS Peak API nicht verfügbar")
                        return
                else:
                    # Show USB camera selection dialog
                    camera_dialog = CameraSelectionDialog(self)
                    if camera_dialog.exec() != QDialog.DialogCode.Accepted:
                        return
                    camera_info = camera_dialog.get_selected_camera()
                    if camera_info is None:
                        QMessageBox.warning(self, "Warnung", "Keine Kamera ausgewählt")
                        return
                    camera_id = camera_info[2]
                    
                    self.camera_thread = CameraThread(
                        camera_type='usb',
                        camera_id=camera_id
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
            if self.ids_peak_thread and self.ids_peak_thread.running:
                self.toggle_camera()  # Disconnect on error
        except Exception as e:
            logger.error(f"Error handling camera error: {e}")
    
    def browse_directory(self):
        """Open file dialog to select output directory."""
        try:
            # Pause camera threads while showing dialog
            camera_was_running = False
            ids_peak_was_running = False
            
            if self.camera_thread and self.camera_thread.running:
                camera_was_running = True
                self.camera_thread.paused = True
            if self.ids_peak_thread and self.ids_peak_thread.running:
                ids_peak_was_running = True
                self.ids_peak_thread.paused = True
            
            # Allow UI to update before showing dialog
            QApplication.processEvents()
            
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
        except Exception as e:
            logger.error(f"Error selecting output directory: {e}")
        finally:
            # Resume camera threads if they were running
            if camera_was_running and self.camera_thread and self.camera_thread.running:
                self.camera_thread.paused = False
            if ids_peak_was_running and self.ids_peak_thread and self.ids_peak_thread.running:
                self.ids_peak_thread.paused = False
    
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
            # Get frame from correct thread
            if self.camera_thread and self.camera_thread.running:
                # Pause camera thread while capturing
                self.camera_thread.paused = True
                
                # Get original resolution frame
                if self.camera_thread.current_frame is None:
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
                
                # Resume camera thread
                self.camera_thread.paused = False
            elif self.ids_peak_thread and self.ids_peak_thread.running:
                # Pause camera thread while capturing
                self.ids_peak_thread.paused = True
                
                # Get original resolution frame
                if self.ids_peak_thread.current_frame is None:
                    raise Exception("No frame available")
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = Path(self.output_dir) / filename
                
                # Save original resolution image
                success = cv2.imwrite(
                    str(filepath),
                    self.ids_peak_thread.current_frame
                )
                
                # Resume camera thread
                self.ids_peak_thread.paused = False
            else:
                raise Exception("No active camera")
            
            if not success:
                raise Exception("Failed to save image")
            
            self.statusBar().showMessage(f"Image saved: {filename}")

            if hasattr(self, 'project_manager') and self.project_manager:
                if not self.project_manager.is_step_completed(WorkflowStep.CAMERA):
                    self.project_manager.mark_step_completed(WorkflowStep.CAMERA)            
                    
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
            # Resume camera if error
            if self.camera_thread and self.camera_thread.running:
                self.camera_thread.paused = False
            if self.ids_peak_thread and self.ids_peak_thread.running:
                self.ids_peak_thread.paused = False
    
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
            
            # Pause camera while viewing image
            was_camera_running = False
            was_ids_running = False
            
            if self.camera_thread and self.camera_thread.running:
                was_camera_running = True
                self.camera_thread.paused = True
            
            if self.ids_peak_thread and self.ids_peak_thread.running:
                was_ids_running = True
                self.ids_peak_thread.paused = True
                
            dialog.exec()
            
            # Resume camera after dialog closes
            if was_camera_running and self.camera_thread and self.camera_thread.running:
                self.camera_thread.paused = False
                
            if was_ids_running and self.ids_peak_thread and self.ids_peak_thread.running:
                self.ids_peak_thread.paused = False
            
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
            if self.ids_peak_thread and self.ids_peak_thread.running:
                self.ids_peak_thread.stop()
                self.ids_peak_thread = None
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
        if self.ids_peak_thread and self.ids_peak_thread.running:
            try:
                self.toggle_camera()
            except Exception as e:
                logger.error(f"Error disconnecting IDS Peak camera on hide: {e}")

    def save_camera_settings_to_project(self):
        """Speichert Kamera-Settings ins Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            settings = {
                'camera_type': self.camera_combo.currentText(),
                'output_directory': self.output_dir or ""
            }
            
            self.project_manager.update_camera_settings(settings)
    
    def capture_image_with_project_integration(self):
        """Erweiterte Bilderfassung mit Projekt-Integration"""
        # Normale capture_image Funktion ausführen
        self.capture_image()
        
        # Settings speichern und Workflow markieren
        if hasattr(self, 'project_manager') and self.project_manager:
            self.save_camera_settings_to_project()
            # Nur beim ersten Bild markieren
            if not self.project_manager.is_step_completed(WorkflowStep.CAMERA):
                self.project_manager.mark_step_completed(WorkflowStep.CAMERA)                




"""
Ergänzungen für gui/camera_app.py
"""

class CameraAppExtensions:
    """Erweiterungen für die Camera App"""
    
    def save_camera_settings_to_project(self):
        """Speichert Kamera-Settings ins Projekt"""
        if hasattr(self, 'project_manager') and self.project_manager:
            settings = {
                'camera_type': self.camera_combo.currentText(),
                'output_directory': self.output_dir or ""
            }
            
            self.project_manager.update_camera_settings(settings)
    
    def capture_image_with_project_integration(self):
        """Erweiterte Bilderfassung mit Projekt-Integration"""
        # Normale capture_image Funktion ausführen
        self.capture_image()
        
        # Settings speichern und Workflow markieren
        if hasattr(self, 'project_manager') and self.project_manager:
            self.save_camera_settings_to_project()
            # Nur beim ersten Bild markieren
            if not self.project_manager.is_step_completed(WorkflowStep.CAMERA):
                self.project_manager.mark_step_completed(WorkflowStep.CAMERA)                