"""
Modern Camera Application with IDS NXT Rio Support (REST API) and IDS Peak Support
Fixed: Text readability issues, proper API separation, and compatibility with main_menu.py
"""

import sys
import os
import cv2
import logging
import json
import warnings
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import numpy as np
import glob
import base64
from io import BytesIO

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QComboBox, QFileDialog, QMessageBox, QProgressBar,
    QApplication, QLineEdit, QFormLayout, QCheckBox, QDialogButtonBox,
    QDialog, QScrollArea, QFrame, QSizePolicy, QSlider, QSpinBox,
    QTabWidget, QGroupBox, QGridLayout, QTextEdit, QSplitter
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, QSize, QFileSystemWatcher, 
    QMutex, QMutexLocker, QPropertyAnimation, QEasingCurve, QRect
)
from PyQt6.QtGui import (
    QImage, QPixmap, QKeyEvent, QFont, QColor, QPalette, QAction,
    QLinearGradient, QPainter, QBrush, QPen, QIcon, QMovie, QWheelEvent
)

# ProjectManager import (only for type hints)
if TYPE_CHECKING:
    from project_manager import ProjectManager, WorkflowStep

# Suppress OpenCV warnings to avoid camera detection spam
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress OpenCV camera detection errors
cv2.setLogLevel(0)

# Check for IDS Peak availability (traditional API)
try:
    import ids_peak.ids_peak as ids_peak
    import ids_peak_ipl.ids_peak_ipl as ids_ipl
    IDS_PEAK_AVAILABLE = True
    logger.info("IDS Peak API available")
except ImportError:
    IDS_PEAK_AVAILABLE = False
    logger.warning("IDS Peak API not available")

# IDS NXT uses REST API over HTTP - no special library needed
IDS_NXT_AVAILABLE = True  # Always available if requests is available
logger.info("IDS NXT REST API available")


class AppTheme:
    """Theme configuration for light and dark modes with proper contrast"""
    
    LIGHT_THEME = {
        'bg_primary': '#FFFFFF',
        'bg_secondary': '#F8F9FA',
        'bg_tertiary': '#E9ECEF',
        'text_primary': '#212529',
        'text_secondary': '#495057',
        'text_muted': '#6C757D',
        'accent_primary': '#007BFF',
        'accent_success': '#28A745',
        'accent_danger': '#DC3545',
        'accent_warning': '#FFC107',
        'border_color': '#DEE2E6',
        'shadow_color': 'rgba(0, 0, 0, 0.1)',
        'input_bg': '#FFFFFF',
        'input_text': '#212529',
        'dropdown_bg': '#FFFFFF',
        'dropdown_text': '#212529',
        'dropdown_selected': '#007BFF',
        'dropdown_selected_text': '#FFFFFF'
    }
    
    DARK_THEME = {
        'bg_primary': '#212529',
        'bg_secondary': '#343A40',
        'bg_tertiary': '#495057',
        'text_primary': '#F8F9FA',
        'text_secondary': '#E9ECEF',
        'text_muted': '#ADB5BD',
        'accent_primary': '#0D6EFD',
        'accent_success': '#198754',
        'accent_danger': '#DC3545',
        'accent_warning': '#FFC107',
        'border_color': '#6C757D',
        'shadow_color': 'rgba(255, 255, 255, 0.1)',
        'input_bg': '#495057',
        'input_text': '#F8F9FA',
        'dropdown_bg': '#495057',
        'dropdown_text': '#F8F9FA',
        'dropdown_selected': '#0D6EFD',
        'dropdown_selected_text': '#FFFFFF'
    }
    
    @staticmethod
    def get_theme(dark_mode=False):
        return AppTheme.DARK_THEME if dark_mode else AppTheme.LIGHT_THEME


class ZoomableImageDialog(QDialog):
    """Dialog für zoombare Bildansicht"""
    
    def __init__(self, image_path: str, dark_mode=False, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.dark_mode = dark_mode
        self.setWindowTitle(f"Image Viewer - {os.path.basename(image_path)}")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
        # Zoom-Parameter
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 0.1
        
        # Drag-Parameter
        self.dragging = False
        self.last_drag_pos = None
        
        self.setup_ui()
        self.load_image()
    
    def setup_ui(self):
        """Setup zoomable image viewer UI"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        self.setStyleSheet(f"""
            QDialog {{
                background: {theme['bg_primary']};
                color: {theme['text_primary']};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 5, 10, 5)
        
        # Zoom controls
        zoom_out_btn = ModernButton("🔍-", dark_mode=self.dark_mode)
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_out_btn.setMaximumWidth(50)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet(f"""
            QLabel {{
                color: {theme['text_primary']};
                font-weight: bold;
                font-size: 14px;
                padding: 5px;
            }}
        """)
        
        zoom_in_btn = ModernButton("🔍+", dark_mode=self.dark_mode)
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_in_btn.setMaximumWidth(50)
        
        fit_btn = ModernButton("📐 Fit", dark_mode=self.dark_mode)
        fit_btn.clicked.connect(self.fit_to_window)
        fit_btn.setMaximumWidth(80)
        
        reset_btn = ModernButton("🔄 Reset", dark_mode=self.dark_mode)
        reset_btn.clicked.connect(self.reset_zoom)
        reset_btn.setMaximumWidth(80)
        
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(self.zoom_label)
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(fit_btn)
        toolbar.addWidget(reset_btn)
        toolbar.addStretch()
        
        # Close button
        close_btn = ModernButton("❌ Close (ESC)", dark_mode=self.dark_mode)
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(close_btn)
        
        layout.addLayout(toolbar)
        
        # Scroll area for image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: {theme['bg_secondary']};
            }}
        """)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background: {theme['bg_primary']};
                border: none;
            }}
        """)
        self.image_label.setMinimumSize(1, 1)
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
        
        # Enable mouse tracking for dragging
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.start_drag
        self.image_label.mouseMoveEvent = self.drag_image
        self.image_label.mouseReleaseEvent = self.end_drag
        
        # Enable wheel events for zooming
        self.scroll_area.wheelEvent = self.wheel_event
    
    def load_image(self):
        """Load and display image"""
        try:
            self.pixmap = QPixmap(self.image_path)
            if self.pixmap.isNull():
                raise Exception("Could not load image")
            
            self.original_size = self.pixmap.size()
            self.fit_to_window()
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            self.image_label.setText("Error loading image")
    
    def update_image_display(self):
        """Update image display with current zoom"""
        if hasattr(self, 'pixmap') and not self.pixmap.isNull():
            scaled_size = self.original_size * self.zoom_factor
            scaled_pixmap = self.pixmap.scaled(
                scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_size)
            
            # Update zoom label
            self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")
    
    def zoom_in(self):
        """Zoom in"""
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor = min(self.zoom_factor * 1.2, self.max_zoom)
            self.update_image_display()
    
    def zoom_out(self):
        """Zoom out"""
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor = max(self.zoom_factor / 1.2, self.min_zoom)
            self.update_image_display()
    
    def fit_to_window(self):
        """Fit image to window size"""
        if hasattr(self, 'pixmap') and not self.pixmap.isNull():
            scroll_size = self.scroll_area.size()
            # Account for scrollbars
            available_size = QSize(scroll_size.width() - 20, scroll_size.height() - 20)
            
            scale_x = available_size.width() / self.original_size.width()
            scale_y = available_size.height() / self.original_size.height()
            
            self.zoom_factor = min(scale_x, scale_y, 1.0)  # Don't zoom in beyond original size
            self.update_image_display()
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        self.update_image_display()
    
    def wheel_event(self, event):
        """Handle mouse wheel for zooming"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom with Ctrl+Wheel
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            # Default scroll behavior
            super(QScrollArea, self.scroll_area).wheelEvent(event)
    
    def start_drag(self, event):
        """Start dragging image"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_drag_pos = event.pos()
            self.image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def drag_image(self, event):
        """Drag image around"""
        if self.dragging and self.last_drag_pos:
            # Calculate drag delta
            delta = event.pos() - self.last_drag_pos
            
            # Get current scrollbar values
            h_scroll = self.scroll_area.horizontalScrollBar()
            v_scroll = self.scroll_area.verticalScrollBar()
            
            # Update scrollbar positions
            h_scroll.setValue(h_scroll.value() - delta.x())
            v_scroll.setValue(v_scroll.value() - delta.y())
            
            self.last_drag_pos = event.pos()
    
    def end_drag(self, event):
        """End dragging"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.last_drag_pos = None
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
        elif event.key() == Qt.Key.Key_F:
            self.fit_to_window()
        elif event.key() == Qt.Key.Key_R:
            self.reset_zoom()
        else:
            super().keyPressEvent(event)
    
    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        # Optionally re-fit to window when dialog is resized
        # self.fit_to_window()


class ModernButton(QPushButton):
    """Custom styled button with modern design and proper contrast"""
    
    def __init__(self, text="", icon=None, primary=False, danger=False, dark_mode=False):
        super().__init__(text)
        self.primary = primary
        self.danger = danger
        self.dark_mode = dark_mode
        self.setMinimumHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_style()
        
    def apply_style(self):
        theme = AppTheme.get_theme(self.dark_mode)
        
        if self.primary:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {theme['accent_success']}, stop:1 #45A049);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5CBF60, stop:1 {theme['accent_success']});
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3D8B40, stop:1 #357A38);
                }}
                QPushButton:disabled {{
                    background: {theme['text_muted']};
                    color: {theme['bg_primary']};
                }}
            """)
        elif self.danger:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {theme['accent_danger']}, stop:1 #D32F2F);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #F66356, stop:1 {theme['accent_danger']});
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #C62828, stop:1 #B71C1C);
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {theme['bg_primary']}, stop:1 {theme['bg_secondary']});
                    color: {theme['text_primary']};
                    border: 2px solid {theme['border_color']};
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {theme['bg_secondary']}, stop:1 {theme['bg_tertiary']});
                    border-color: {theme['text_muted']};
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {theme['bg_tertiary']}, stop:1 {theme['border_color']});
                }}
            """)


class ModernCard(QFrame):
    """Card-style container with shadow effect and proper theming"""
    
    def __init__(self, title="", dark_mode=False, parent=None):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.setFrameStyle(QFrame.Shape.Box)
        self.apply_style()
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(16)
        
        if title:
            theme = AppTheme.get_theme(dark_mode)
            title_label = QLabel(title)
            title_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 18px;
                    font-weight: bold;
                    color: {theme['text_primary']};
                    border: none;
                    padding: 0px;
                    background: transparent;
                }}
            """)
            self.layout.addWidget(title_label)
    
    def apply_style(self):
        theme = AppTheme.get_theme(self.dark_mode)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme['bg_primary']};
                border: 1px solid {theme['border_color']};
                border-radius: 12px;
                padding: 16px;
            }}
        """)


class CameraDetector:
    """Enhanced camera detection with robust error handling"""
    
    @staticmethod
    def detect_all_cameras() -> List[Dict[str, Any]]:
        """Detect all available cameras with error suppression"""
        cameras = []
        
        # Detect USB cameras (with error suppression)
        cameras.extend(CameraDetector.detect_usb_cameras())
        
        # Detect IDS Peak cameras (traditional API)
        if IDS_PEAK_AVAILABLE:
            cameras.extend(CameraDetector.detect_ids_peak_cameras())
        
        # Add IDS NXT cameras (REST API) - manual entry option
        if IDS_NXT_AVAILABLE:
            cameras.extend(CameraDetector.detect_ids_nxt_cameras())
        
        return cameras
    
    @staticmethod
    def detect_usb_cameras() -> List[Dict[str, Any]]:
        """Detect USB cameras using OpenCV with error suppression"""
        cameras = []
        
        # Redirect stderr to suppress OpenCV errors
        original_stderr = os.dup(2)
        os.close(2)
        os.open(os.devnull, os.O_RDWR)
        
        try:
            for i in range(5):  # Reduced range to avoid spam
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                    if cap.isOpened():
                        # Test if camera actually works
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            # Get camera properties
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            cameras.append({
                                'type': 'usb',
                                'id': i,
                                'name': f'USB Camera {i}',
                                'resolution': f'{width}x{height}' if width > 0 and height > 0 else 'Unknown',
                                'fps': fps if fps > 0 else 'Unknown',
                                'backend': 'DirectShow'
                            })
                        cap.release()
                except Exception:
                    # Silently ignore errors during detection
                    pass
        finally:
            # Restore stderr
            os.dup2(original_stderr, 2)
            os.close(original_stderr)
        
        return cameras
    
    @staticmethod
    def detect_ids_peak_cameras() -> List[Dict[str, Any]]:
        """Detect IDS Peak cameras with fixed API calls"""
        cameras = []
        try:
            ids_peak.Library.Initialize()
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            
            devices = device_manager.Devices()
            for i, device in enumerate(devices):
                # Fixed: Use correct attributes
                cameras.append({
                    'type': 'ids_peak',
                    'id': i,
                    'name': f'IDS Peak {device.DisplayName()}',
                    'serial': device.SerialNumber() if hasattr(device, 'SerialNumber') else 'Unknown',
                    'model': device.ModelName() if hasattr(device, 'ModelName') else 'Unknown',
                    'interface': device.ParentInterface().DisplayName() if hasattr(device, 'ParentInterface') else 'Unknown'
                })
        except Exception as e:
            logger.debug(f"Error detecting IDS Peak cameras: {e}")
        
        return cameras
    
    @staticmethod
    def detect_ids_nxt_cameras() -> List[Dict[str, Any]]:
        """Add IDS NXT cameras as manual entry option (REST API)"""
        cameras = []
        try:
            # IDS NXT cameras need manual configuration since they use REST API
            cameras.append({
                'type': 'ids_nxt',
                'id': 0,
                'name': 'IDS NXT Rio (REST API)',
                'ip': '192.168.1.99',
                'port': None,  # Optional port
                'protocol': 'http',
                'username': 'admin',
                'password': 'Flex',
                'description': 'Configure via REST API',
                'manual': True
            })
        except Exception as e:
            logger.debug(f"Error adding IDS NXT cameras: {e}")
        
        return cameras


class NXTConfigDialog(QDialog):
    """Dialog for configuring IDS NXT camera connection (REST API)"""
    
    def __init__(self, dark_mode=False, parent=None):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.setWindowTitle("IDS NXT Camera Configuration (REST API)")
        self.setModal(True)
        self.setMinimumWidth(450)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup configuration dialog UI"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme['bg_primary']};
                color: {theme['text_primary']};
            }}
            QLabel {{
                color: {theme['text_primary']};
                font-weight: bold;
                padding: 4px;
            }}
            QLineEdit {{
                background-color: {theme['input_bg']};
                color: {theme['input_text']};
                border: 2px solid {theme['border_color']};
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border-color: {theme['accent_primary']};
            }}
            QCheckBox {{
                color: {theme['text_primary']};
                font-weight: bold;
                padding: 4px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
            }}
        """)
        
        layout = QFormLayout(self)
        layout.setSpacing(12)
        
        # Protocol
        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems(["http", "https"])
        self.protocol_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {theme['dropdown_bg']};
                color: {theme['dropdown_text']};
                border: 2px solid {theme['border_color']};
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme['dropdown_bg']};
                color: {theme['dropdown_text']};
                border: 1px solid {theme['border_color']};
                selection-background-color: {theme['dropdown_selected']};
                selection-color: {theme['dropdown_selected_text']};
            }}
        """)
        layout.addRow("Protocol:", self.protocol_combo)
        
        # IP Address
        self.ip_edit = QLineEdit("192.168.1.99")
        layout.addRow("IP Address:", self.ip_edit)
        
        # Port (optional)
        port_layout = QHBoxLayout()
        self.port_edit = QLineEdit("")
        self.port_edit.setPlaceholderText("Default: 80 (HTTP) / 443 (HTTPS)")
        port_layout.addWidget(self.port_edit)
        layout.addRow("Port (optional):", port_layout)
        
        # Username
        self.username_edit = QLineEdit("admin")
        layout.addRow("Username:", self.username_edit)
        
        # Password with show/hide toggle
        password_layout = QHBoxLayout()
        self.password_edit = QLineEdit("Flex")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.show_password_check = QCheckBox("Show password")
        self.show_password_check.toggled.connect(self.toggle_password_visibility)
        
        password_layout.addWidget(self.password_edit)
        password_layout.addWidget(self.show_password_check)
        layout.addRow("Password:", password_layout)
        
        # Test Connection Button
        self.test_btn = ModernButton("🔍 Test Connection", dark_mode=self.dark_mode)
        self.test_btn.clicked.connect(self.test_connection)
        layout.addRow("", self.test_btn)
        
        # Connection Status
        self.status_label = QLabel("Ready to connect...")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {theme['text_secondary']};
                font-style: italic;
                padding: 8px;
                border: 1px solid {theme['border_color']};
                border-radius: 4px;
                background-color: {theme['bg_secondary']};
            }}
        """)
        layout.addRow("Status:", self.status_label)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def toggle_password_visibility(self, checked):
        """Toggle password visibility"""
        if checked:
            self.password_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
    
    def test_connection(self):
        """Test connection to IDS NXT camera"""
        try:
            self.test_btn.setEnabled(False)
            self.status_label.setText("Testing connection...")
            
            config = self.get_config()
            
            # Build URL with optional port
            if config['port']:
                base_url = f"{config['protocol']}://{config['ip']}:{config['port']}"
            else:
                base_url = f"{config['protocol']}://{config['ip']}"
            
            # Test device info endpoint (more reliable than /api/info)
            response = requests.get(f"{base_url}/deviceinfo", 
                                  auth=(config['username'], config['password']),
                                  timeout=5)
            
            if response.status_code == 200:
                device_info = response.json()
                model = device_info.get('DeviceModel', 'Unknown')
                serial = device_info.get('Serialnumber', 'Unknown')
                self.status_label.setText(f"✅ Connected to {model} (S/N: {serial})")
                self.status_label.setStyleSheet(self.status_label.styleSheet() + "color: green;")
            else:
                self.status_label.setText(f"❌ Connection failed: HTTP {response.status_code}")
                self.status_label.setStyleSheet(self.status_label.styleSheet() + "color: red;")
                
        except Exception as e:
            self.status_label.setText(f"❌ Connection error: {str(e)}")
            self.status_label.setStyleSheet(self.status_label.styleSheet() + "color: red;")
        finally:
            self.test_btn.setEnabled(True)
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters"""
        port_text = self.port_edit.text().strip()
        port = None
        if port_text and port_text.isdigit():
            port = int(port_text)
        
        return {
            'protocol': self.protocol_combo.currentText(),
            'ip': self.ip_edit.text().strip(),
            'port': port,
            'username': self.username_edit.text().strip(),
            'password': self.password_edit.text().strip()
        }


class IDSNXTRestClient:
    """REST API client for IDS NXT cameras"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Build base URL with optional port
        if config.get('port'):
            self.base_url = f"{config['protocol']}://{config['ip']}:{config['port']}"
        else:
            self.base_url = f"{config['protocol']}://{config['ip']}"
            
        self.auth = (config['username'], config['password'])
        self.session = requests.Session()
        self.session.auth = self.auth
        self.connected = False
        
    def connect(self):
        """Connect to IDS NXT camera"""
        try:
            # Test connection with device info endpoint
            response = self.session.get(f"{self.base_url}/deviceinfo", timeout=5)
            response.raise_for_status()
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"IDS NXT connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IDS NXT camera"""
        self.connected = False
        self.session.close()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture frame from IDS NXT camera via REST API"""
        if not self.connected:
            return None
        
        try:
            # Get latest image from camera using the correct endpoint
            response = self.session.get(f"{self.base_url}/camera/image", timeout=10)
            response.raise_for_status()
            
            # Convert response to numpy array
            image_data = response.content
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return frame
        except Exception as e:
            logger.error(f"Error getting frame from IDS NXT: {e}")
            return None
    
    def get_device_info(self) -> Optional[Dict[str, Any]]:
        """Get device information"""
        if not self.connected:
            return None
        
        try:
            response = self.session.get(f"{self.base_url}/deviceinfo", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return None
    
    def get_camera_settings(self) -> Optional[Dict[str, Any]]:
        """Get camera settings"""
        if not self.connected:
            return None
        
        try:
            response = self.session.get(f"{self.base_url}/camera", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting camera settings: {e}")
            return None
    
    def set_exposure(self, exposure_us: int) -> bool:
        """Set camera exposure time in microseconds"""
        if not self.connected:
            return False
        
        try:
            # Use form data as specified in the API documentation
            data = {'ExposureTime': exposure_us}
            response = self.session.patch(f"{self.base_url}/camera", 
                                        data=data, timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error setting exposure: {e}")
            return False
    
    def set_gain(self, gain_percent: int) -> bool:
        """Set camera gain in percent (0-100)"""
        if not self.connected:
            return False
        
        try:
            # Use form data as specified in the API documentation
            data = {'Gain': gain_percent}
            response = self.session.patch(f"{self.base_url}/camera", 
                                        data=data, timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False
    
    def trigger_software(self) -> bool:
        """Trigger software trigger"""
        if not self.connected:
            return False
        
        try:
            response = self.session.post(f"{self.base_url}/camera/trigger", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error triggering software trigger: {e}")
            return False


class CameraThread(QThread):
    """Universal camera thread supporting multiple camera types"""
    
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    
    def __init__(self, camera_info: Dict[str, Any], nxt_config: Dict[str, Any] = None):
        super().__init__()
        self.camera_info = camera_info
        self.nxt_config = nxt_config or {}
        self.camera = None
        self.running = False
        self.paused = False
        self.current_frame = None
        self._lock = QMutex()
        
    def run(self):
        """Main camera loop"""
        try:
            self.initialize_camera()
            self.running = True
            self.status_changed.emit("Camera connected")
            
            while self.running:
                if self.paused:
                    self.msleep(100)
                    continue
                
                frame = self.capture_frame()
                if frame is not None:
                    with QMutexLocker(self._lock):
                        self.current_frame = frame.copy()
                    # Convert BGR to RGB for display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_ready.emit(rgb_frame)
                else:
                    self.msleep(50)
                    
        except Exception as e:
            logger.error(f"Camera thread error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.cleanup_camera()
    
    def initialize_camera(self):
        """Initialize camera based on type"""
        camera_type = self.camera_info['type']
        
        if camera_type == 'usb':
            self.camera = cv2.VideoCapture(self.camera_info['id'])
            if not self.camera.isOpened():
                raise Exception(f"Failed to open USB camera {self.camera_info['id']}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
        elif camera_type == 'ids_nxt':
            # Initialize IDS NXT camera via REST API
            self.camera = IDSNXTRestClient(self.nxt_config)
            if not self.camera.connect():
                raise Exception("Failed to connect to IDS NXT camera")
            
        elif camera_type == 'ids_peak':
            if not IDS_PEAK_AVAILABLE:
                raise Exception("IDS Peak API not available")
            
            # Initialize IDS Peak camera
            ids_peak.Library.Initialize()
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            
            devices = device_manager.Devices()
            if self.camera_info['id'] < len(devices):
                device = devices[self.camera_info['id']]
                self.camera = device.OpenDevice(ids_peak.DeviceAccessType_Control)
                # Additional IDS Peak setup would go here
            else:
                raise Exception(f"IDS Peak camera {self.camera_info['id']} not found")
        
        else:
            raise Exception(f"Unsupported camera type: {camera_type}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from camera"""
        try:
            camera_type = self.camera_info['type']
            
            if camera_type == 'usb':
                ret, frame = self.camera.read()
                return frame if ret else None
            
            elif camera_type == 'ids_nxt':
                # Get frame from IDS NXT via REST API
                return self.camera.get_frame()
            
            elif camera_type == 'ids_peak':
                # IDS Peak frame capture would go here
                # This is simplified - actual implementation depends on IDS Peak API
                return None  # Placeholder
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def cleanup_camera(self):
        """Clean up camera resources"""
        try:
            if self.camera:
                camera_type = self.camera_info['type']
                
                if camera_type == 'usb':
                    self.camera.release()
                elif camera_type == 'ids_nxt':
                    self.camera.disconnect()
                elif camera_type == 'ids_peak':
                    # Clean up IDS Peak resources
                    pass
                
                self.camera = None
                self.status_changed.emit("Camera disconnected")
        except Exception as e:
            logger.error(f"Error cleaning up camera: {e}")
    
    def stop(self):
        """Stop camera thread"""
        self.running = False
        self.paused = False
        if not self.wait(2000):
            self.terminate()
            self.wait()
    
    def pause(self):
        """Pause camera capture"""
        self.paused = True
    
    def resume(self):
        """Resume camera capture"""
        self.paused = False
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame thread-safely"""
        with QMutexLocker(self._lock):
            return self.current_frame.copy() if self.current_frame is not None else None


class ImageGallery(QWidget):
    """Modern image gallery with thumbnails and proper theming"""
    
    image_selected = pyqtSignal(str)
    image_deleted = pyqtSignal(str)
    
    def __init__(self, dark_mode=False):
        super().__init__()
        self.dark_mode = dark_mode
        self.setup_ui()
        self.image_paths = []
        
    def setup_ui(self):
        """Setup gallery UI"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("📸 Recent Captures")
        header.setStyleSheet(f"""
            QLabel {{
                font-size: 18px;
                font-weight: bold;
                color: {theme['text_primary']};
                padding: 16px;
                background: transparent;
            }}
        """)
        layout.addWidget(header)
        
        # Scroll area for thumbnails
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: {theme['bg_secondary']};
            }}
        """)
        
        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QVBoxLayout(self.thumbnail_widget)
        self.thumbnail_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll_area.setWidget(self.thumbnail_widget)
        layout.addWidget(scroll_area)
    
    def add_image(self, image_path: str):
        """Add image to gallery"""
        if image_path not in self.image_paths:
            self.image_paths.insert(0, image_path)  # Add to beginning
            self.refresh_thumbnails()
    
    def refresh_thumbnails(self):
        """Refresh thumbnail display"""
        # Clear existing thumbnails
        for i in reversed(range(self.thumbnail_layout.count())):
            child = self.thumbnail_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Add new thumbnails
        for image_path in self.image_paths:
            if os.path.exists(image_path):
                thumbnail = self.create_thumbnail(image_path)
                self.thumbnail_layout.addWidget(thumbnail)
    
    def create_thumbnail(self, image_path: str) -> QWidget:
        """Create thumbnail widget for image"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        container = QFrame()
        container.setFixedSize(200, 160)
        container.setStyleSheet(f"""
            QFrame {{
                background: {theme['bg_primary']};
                border: 2px solid {theme['border_color']};
                border-radius: 8px;
                margin: 4px;
            }}
            QFrame:hover {{
                border-color: {theme['accent_success']};
            }}
        """)
        
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        
        # Image
        image_label = QLabel()
        image_label.setFixedSize(180, 120)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet(f"""
            QLabel {{
                border: none; 
                background: {theme['bg_secondary']};
                color: {theme['text_secondary']};
            }}
        """)
        
        # Load and scale image
        try:
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                180, 120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            logger.error(f"Error loading thumbnail: {e}")
            image_label.setText("Error loading image")
        
        # Filename
        filename_label = QLabel(os.path.basename(image_path))
        filename_label.setStyleSheet(f"""
            QLabel {{
                font-size: 12px;
                color: {theme['text_secondary']};
                border: none;
                background: transparent;
            }}
        """)
        filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(image_label)
        layout.addWidget(filename_label)
        
        # Make clickable
        container.mousePressEvent = lambda e: self.image_selected.emit(image_path)
        container.setCursor(Qt.CursorShape.PointingHandCursor)
        
        return container
    
    def set_images(self, image_paths: List[str]):
        """Set gallery images"""
        self.image_paths = image_paths
        self.refresh_thumbnails()


class CameraApp(QMainWindow):
    """Modern camera application with enhanced design and functionality
    
    Provides compatibility attributes for integration with main_menu.py:
    - dir_label: Label showing current output directory
    - output_dir: Property for getting/setting output directory
    """
    
    def __init__(self, project_manager=None):
        super().__init__()
        self.project_manager = project_manager
        self.camera_thread = None
        self.ids_peak_thread = None  # Add IDS Peak thread reference
        self.output_directory = None
        self.available_cameras = []
        self.current_camera_info = None
        self.dark_mode = False
        
        # Initialize compatibility attributes for main_menu.py integration
        self.dir_label = None
        
        # Set output directory from project manager if available
        if self.project_manager:
            try:
                self.output_directory = str(self.project_manager.get_raw_images_dir())
                logger.info(f"Using project manager output directory: {self.output_directory}")
            except Exception as e:
                logger.error(f"Error getting project manager directory: {e}")
        
        self.setup_ui()
        self.setup_connections()
        self.detect_cameras()
        
        # Setup window
        if self.project_manager:
            self.setWindowTitle(f"Camera Studio - {self.project_manager.config.project_name}")
        else:
            self.setWindowTitle("Modern Camera Studio - IDS NXT & Peak Support")
        self.setMinimumSize(1400, 900)
        self.showMaximized()
    
    @property
    def output_dir(self) -> Optional[str]:
        """Property for compatibility with main_menu.py"""
        return self.output_directory
    
    @output_dir.setter
    def output_dir(self, path: str):
        """Property setter for compatibility with main_menu.py"""
        self.output_directory = path
        if self.dir_label:
            self.dir_label.setText(f"📁 {path}")
        self.load_existing_images()
    
    def setup_ui(self):
        """Setup main UI"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        # Set application style
        self.setStyleSheet(f"""
            QMainWindow {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {theme['bg_secondary']}, stop:1 {theme['bg_tertiary']});
                color: {theme['text_primary']};
            }}
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left panel - camera and controls
        left_panel = QVBoxLayout()
        
        # Camera selection card
        self.camera_card = ModernCard("📷 Camera Selection", self.dark_mode)
        self.setup_camera_selection()
        left_panel.addWidget(self.camera_card)
        
        # Video display card
        self.video_card = ModernCard("🎥 Live Preview", self.dark_mode)
        self.setup_video_display()
        left_panel.addWidget(self.video_card)
        
        # Control buttons
        self.setup_control_buttons()
        left_panel.addLayout(self.control_layout)
        
        # Right panel - gallery and settings
        right_panel = QVBoxLayout()
        
        # Theme toggle
        self.setup_theme_toggle()
        right_panel.addLayout(self.theme_layout)
        
        # Directory selection - only show if no project manager
        if not self.project_manager:
            self.directory_card = ModernCard("📁 Output Directory", self.dark_mode)
            self.setup_directory_selection()
            right_panel.addWidget(self.directory_card)
        else:
            # Show project info instead
            self.project_card = ModernCard("📋 Project Info", self.dark_mode)
            self.setup_project_info()
            right_panel.addWidget(self.project_card)
        
        # Image gallery
        self.gallery = ImageGallery(self.dark_mode)
        right_panel.addWidget(self.gallery)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        # Status bar
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background: {theme['bg_tertiary']};
                color: {theme['text_primary']};
                padding: 8px;
                border-top: 1px solid {theme['border_color']};
            }}
        """)
        self.statusBar().showMessage("Ready - Select a camera to begin")
    
    def setup_theme_toggle(self):
        """Setup theme toggle button"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        self.theme_layout = QHBoxLayout()
        
        theme_label = QLabel("🌙 Dark Mode:")
        theme_label.setStyleSheet(f"""
            QLabel {{
                color: {theme['text_primary']};
                font-weight: bold;
                font-size: 14px;
            }}
        """)
        
        self.theme_toggle = QCheckBox()
        self.theme_toggle.setChecked(self.dark_mode)
        self.theme_toggle.setStyleSheet(f"""
            QCheckBox {{
                color: {theme['text_primary']};
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
            }}
        """)
        
        self.theme_layout.addWidget(theme_label)
        self.theme_layout.addWidget(self.theme_toggle)
        self.theme_layout.addStretch()
    
    def toggle_theme(self, checked):
        """Toggle between light and dark theme"""
        self.dark_mode = checked
        
        # Recreate UI with new theme
        self.setup_ui()
        self.setup_connections()
        
        # Refresh gallery
        if hasattr(self, 'gallery'):
            self.gallery.dark_mode = self.dark_mode
            self.gallery.setup_ui()
            self.gallery.refresh_thumbnails()
        
        # Reload existing images if directory is set
        if self.output_directory:
            self.load_existing_images()
    
    def setup_camera_selection(self):
        """Setup camera selection UI"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        selection_layout = QVBoxLayout()
        
        # Camera label with proper styling
        camera_label = QLabel("Select Camera:")
        camera_label.setStyleSheet(f"""
            QLabel {{
                color: {theme['text_primary']};
                font-weight: bold;
                font-size: 14px;
                padding: 4px;
            }}
        """)
        
        # Camera dropdown with proper styling
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(40)
        self.camera_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {theme['dropdown_bg']};
                color: {theme['dropdown_text']};
                border: 2px solid {theme['border_color']};
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QComboBox:focus {{
                border-color: {theme['accent_primary']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme['dropdown_bg']};
                color: {theme['dropdown_text']};
                border: 1px solid {theme['border_color']};
                selection-background-color: {theme['dropdown_selected']};
                selection-color: {theme['dropdown_selected_text']};
                font-size: 14px;
                font-weight: bold;
            }}
        """)
        
        # Connection button
        self.connect_button = ModernButton("🔌 Connect Camera", primary=True, dark_mode=self.dark_mode)
        self.connect_button.setEnabled(False)
        
        selection_layout.addWidget(camera_label)
        selection_layout.addWidget(self.camera_combo)
        selection_layout.addWidget(self.connect_button)
        
        self.camera_card.layout.addLayout(selection_layout)
    
    def setup_video_display(self):
        """Setup video display"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background: #000000;
                border: 2px solid {theme['border_color']};
                border-radius: 8px;
                color: white;
                font-size: 16px;
                font-weight: bold;
            }}
        """)
        self.video_label.setText("📸 Camera Preview\n\nSelect and connect a camera to see live preview")
        
        self.video_card.layout.addWidget(self.video_label)
    
    def setup_control_buttons(self):
        """Setup control buttons"""
        self.control_layout = QHBoxLayout()
        self.control_layout.setSpacing(16)
        
        # Capture button
        self.capture_button = ModernButton("📸 Capture Photo", primary=True, dark_mode=self.dark_mode)
        self.capture_button.setMinimumHeight(60)
        self.capture_button.setEnabled(False)
        
        # Disconnect button
        self.disconnect_button = ModernButton("⏹️ Disconnect", danger=True, dark_mode=self.dark_mode)
        self.disconnect_button.setEnabled(False)
        
        # Refresh cameras button
        self.refresh_button = ModernButton("🔄 Refresh Cameras", dark_mode=self.dark_mode)
        
        self.control_layout.addWidget(self.capture_button, 2)
        self.control_layout.addWidget(self.disconnect_button, 1)
        self.control_layout.addWidget(self.refresh_button, 1)
    
    def setup_project_info(self):
        """Setup project information display"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        info_layout = QVBoxLayout()
        
        # Project name
        project_name = QLabel(self.project_manager.config.project_name)
        project_name.setStyleSheet(f"""
            QLabel {{
                color: {theme['text_primary']};
                font-weight: bold;
                font-size: 16px;
                padding: 4px;
            }}
        """)
        info_layout.addWidget(project_name)
        
        # Output directory (compatibility label)
        self.dir_label = QLabel(f"📁 {self.output_directory}")
        self.dir_label.setStyleSheet(f"""
            QLabel {{
                color: {theme['text_secondary']};
                font-size: 12px;
                padding: 4px;
            }}
        """)
        info_layout.addWidget(self.dir_label)
        
        # Classes info
        classes_count = len(self.project_manager.config.classes)
        if classes_count > 0:
            classes_info = QLabel(f"🏷️ {classes_count} classes defined")
            classes_info.setStyleSheet(f"""
                QLabel {{
                    color: {theme['text_secondary']};
                    font-size: 12px;
                    padding: 4px;
                }}
            """)
            info_layout.addWidget(classes_info)
        
        self.project_card.layout.addLayout(info_layout)
    
    def setup_directory_selection(self):
        """Setup directory selection (only when no project manager)"""
        theme = AppTheme.get_theme(self.dark_mode)
        
        dir_layout = QVBoxLayout()
        
        # Compatibility label for main_menu.py integration
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet(f"""
            QLabel {{
                color: {theme['text_secondary']};
                font-style: italic;
                border: none;
                padding: 8px;
                background: transparent;
                font-size: 14px;
            }}
        """)
        
        # Keep reference to the label as directory_label for internal use
        self.directory_label = self.dir_label
        
        browse_button = ModernButton("📁 Browse Directory", dark_mode=self.dark_mode)
        browse_button.clicked.connect(self.browse_directory)
        
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(browse_button)
        
        self.directory_card.layout.addLayout(dir_layout)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selection_changed)
        self.connect_button.clicked.connect(self.toggle_camera_connection)
        self.disconnect_button.clicked.connect(self.disconnect_camera)
        self.capture_button.clicked.connect(self.capture_image)
        self.refresh_button.clicked.connect(self.detect_cameras)
        self.gallery.image_selected.connect(self.show_full_image)
        
        # Theme toggle connection
        if hasattr(self, 'theme_toggle'):
            self.theme_toggle.toggled.connect(self.toggle_theme)
    
    def detect_cameras(self):
        """Detect available cameras"""
        self.statusBar().showMessage("Detecting cameras...")
        
        # Clear existing cameras
        self.camera_combo.clear()
        self.available_cameras = []
        
        # Detect cameras
        try:
            cameras = CameraDetector.detect_all_cameras()
            self.available_cameras = cameras
            
            if cameras:
                for camera in cameras:
                    if camera['type'] == 'usb':
                        display_name = f"🔌 {camera['name']} (USB)"
                    elif camera['type'] == 'ids_peak':
                        display_name = f"📷 {camera['name']} (IDS Peak)"
                    elif camera['type'] == 'ids_nxt':
                        display_name = f"🌐 {camera['name']} (REST API)"
                    else:
                        display_name = f"❓ {camera['name']}"
                    
                    self.camera_combo.addItem(display_name)
                
                self.connect_button.setEnabled(True)
                self.statusBar().showMessage(f"Found {len(cameras)} camera(s)")
            else:
                self.camera_combo.addItem("No cameras detected")
                self.connect_button.setEnabled(False)
                self.statusBar().showMessage("No cameras found")
                
        except Exception as e:
            logger.error(f"Error detecting cameras: {e}")
            self.statusBar().showMessage("Error detecting cameras")
            QMessageBox.critical(self, "Error", f"Error detecting cameras: {e}")
    
    def on_camera_selection_changed(self, index):
        """Handle camera selection change"""
        if 0 <= index < len(self.available_cameras):
            self.current_camera_info = self.available_cameras[index]
            self.connect_button.setEnabled(True)
        else:
            self.current_camera_info = None
            self.connect_button.setEnabled(False)
    
    def toggle_camera_connection(self):
        """Toggle camera connection"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.disconnect_camera()
        else:
            self.connect_camera()
    
    def connect_camera(self):
        """Connect to selected camera"""
        if not self.current_camera_info:
            QMessageBox.warning(self, "Warning", "Please select a camera first")
            return
        
        try:
            nxt_config = None
            
            # Handle IDS NXT configuration
            if self.current_camera_info['type'] == 'ids_nxt':
                config_dialog = NXTConfigDialog(self.dark_mode, self)
                if config_dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                nxt_config = config_dialog.get_config()
            
            # Create and start camera thread
            self.camera_thread = CameraThread(self.current_camera_info, nxt_config)
            self.camera_thread.frame_ready.connect(self.update_video_display)
            self.camera_thread.error_occurred.connect(self.handle_camera_error)
            self.camera_thread.status_changed.connect(self.statusBar().showMessage)
            
            self.camera_thread.start()
            
            # Update UI
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            self.capture_button.setEnabled(True)
            self.camera_combo.setEnabled(False)
            
            self.statusBar().showMessage("Connecting to camera...")
            
        except Exception as e:
            logger.error(f"Error connecting camera: {e}")
            QMessageBox.critical(self, "Error", f"Failed to connect camera: {e}")
    
    def disconnect_camera(self):
        """Disconnect camera"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        # Update UI
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
        
        # Clear video display
        self.video_label.clear()
        self.video_label.setText("📸 Camera Preview\n\nSelect and connect a camera to see live preview")
        
        self.statusBar().showMessage("Camera disconnected")
    
    def update_video_display(self, frame):
        """Update video display with new frame"""
        try:
            # Convert frame to QImage
            if len(frame.shape) == 3:
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                h, w = frame.shape
                bytes_per_line = w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            
            # Scale to fit display
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Error updating video display: {e}")
    
    def handle_camera_error(self, error_message):
        """Handle camera errors"""
        logger.error(f"Camera error: {error_message}")
        QMessageBox.critical(self, "Camera Error", error_message)
        self.disconnect_camera()
    
    def browse_directory(self):
        """Browse for output directory (only when no project manager)"""
        if self.project_manager:
            # Should not be called when project manager is present
            logger.warning("browse_directory called with project manager present")
            return
        
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        
        if directory:
            self.output_directory = directory
            theme = AppTheme.get_theme(self.dark_mode)
            self.dir_label.setText(f"📁 {directory}")
            self.dir_label.setStyleSheet(f"""
                QLabel {{
                    color: {theme['text_primary']};
                    font-weight: bold;
                    border: none;
                    padding: 8px;
                    background: transparent;
                    font-size: 14px;
                }}
            """)
            
            # Load existing images
            self.load_existing_images()
            
            self.statusBar().showMessage(f"Output directory set: {directory}")
    
    def load_existing_images(self):
        """Load existing images from output directory"""
        if not self.output_directory:
            return
        
        try:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                pattern = os.path.join(self.output_directory, f"*{ext}")
                image_files.extend(glob.glob(pattern))
            
            # Sort by modification time (newest first)
            image_files.sort(key=os.path.getmtime, reverse=True)
            
            self.gallery.set_images(image_files)
            
        except Exception as e:
            logger.error(f"Error loading existing images: {e}")
    
    def capture_image(self):
        """Capture image from camera"""
        if not self.camera_thread or not self.camera_thread.isRunning():
            QMessageBox.warning(self, "Warning", "No camera connected")
            return
        
        if not self.output_directory:
            QMessageBox.warning(self, "Warning", "Please select an output directory first")
            return
        
        try:
            # Get current frame
            frame = self.camera_thread.get_current_frame()
            if frame is None:
                QMessageBox.warning(self, "Warning", "No frame available")
                return
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(self.output_directory, filename)
            
            # Save image
            success = cv2.imwrite(filepath, frame)
            
            if success:
                self.statusBar().showMessage(f"Image saved: {filename}")
                self.gallery.add_image(filepath)
                
                # Show success animation
                self.show_capture_animation()
                
                # Mark workflow step as completed if project manager is available
                if self.project_manager:
                    try:
                        from project_manager import WorkflowStep
                        self.project_manager.mark_step_completed(WorkflowStep.CAMERA)
                    except Exception as e:
                        logger.warning(f"Could not mark workflow step as completed: {e}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save image")
                
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to capture image: {e}")
    
    def show_capture_animation(self):
        """Show capture animation"""
        # Simple flash effect
        original_style = self.video_label.styleSheet()
        self.video_label.setStyleSheet(original_style + "background: white;")
        
        # Timer to restore original style
        QTimer.singleShot(100, lambda: self.video_label.setStyleSheet(original_style))
    
    def show_full_image(self, image_path):
        """Show image in zoomable full size viewer"""
        try:
            # Check if image still exists
            if not os.path.exists(image_path):
                QMessageBox.warning(self, "Error", "Image file no longer exists")
                self.refresh_gallery()
                return
            
            # Pause camera while viewing image
            was_camera_running = False
            was_ids_running = False
            
            if self.camera_thread and self.camera_thread.running:
                was_camera_running = True
                self.camera_thread.paused = True
            
            if self.ids_peak_thread and self.ids_peak_thread.running:
                was_ids_running = True
                self.ids_peak_thread.paused = True
            
            # Create and show zoomable image dialog
            dialog = ZoomableImageDialog(image_path, self.dark_mode, self)
            dialog.exec()
            
            # Resume camera after dialog closes
            if was_camera_running and self.camera_thread and self.camera_thread.running:
                self.camera_thread.paused = False
                
            if was_ids_running and self.ids_peak_thread and self.ids_peak_thread.running:
                self.ids_peak_thread.paused = False
            
        except Exception as e:
            logger.error(f"Error showing full image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to show image: {e}")
            
            # Resume camera even if error occurred
            if self.camera_thread and self.camera_thread.running:
                self.camera_thread.paused = False
            if self.ids_peak_thread and self.ids_peak_thread.running:
                self.ids_peak_thread.paused = False
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Space and self.capture_button.isEnabled():
            self.capture_image()
        elif event.key() == Qt.Key.Key_Escape:
            # Close any open dialogs or exit fullscreen
            pass
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle close event"""
        self.disconnect_camera()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Modern Camera Studio")
    app.setApplicationVersion("2.1")
    app.setOrganizationName("Camera Studio")
    
    # Create and show main window
    # Note: project_manager can be passed as parameter when launched from project manager
    window = CameraApp(project_manager=None)  # Set to None for standalone mode
    window.show()
    
    # Run application
    sys.exit(app.exec())


def launch_camera_app(project_manager=None):
    """Launch camera app with optional project manager integration"""
    try:
        # Create and show camera app window
        window = CameraApp(project_manager=project_manager)
        window.show()
        return window
    except Exception as e:
        logger.error(f"Error launching camera app: {e}")
        return None


if __name__ == "__main__":
    main()