"""Dataset viewer application module."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QMessageBox,
    QApplication
)
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
import subprocess
import sys
import os
from pathlib import Path
import logging
import signal
import time

class DatasetViewerApp(QMainWindow):
    """Main window for the dataset viewer application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Flask server process
        self.server_process = None
        
        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Dataset selection
        self.dataset_label = QLabel("Dataset Directory:")
        self.dataset_button = QPushButton("Browse")
        self.dataset_button.clicked.connect(self.browse_dataset)
        controls_layout.addWidget(self.dataset_label)
        controls_layout.addWidget(self.dataset_button)
        
        # Add controls to main layout
        self.layout.addLayout(controls_layout)
        
        # Web view for Flask app
        self.web_view = QWebEngineView()
        QWebEngineProfile.defaultProfile().clearHttpCache()
        
        # Configure web settings
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        
        self.layout.addWidget(self.web_view)
        
        # Start Flask server
        self.start_flask_server()
        
        # Connect to closeEvent
        self.closeEvent = self.handle_close
    
    def browse_dataset(self):
        """Open file dialog to select dataset directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory"
        )
        if path:
            self.dataset_label.setText(f"Dataset Directory: {path}")
            # Load dataset in web view
            self.web_view.setUrl(QUrl("http://127.0.0.1:5003"))
    
    def start_flask_server(self):
        """Start the Flask server as a subprocess."""
        try:
            flask_script = Path(__file__).parent.parent / "dataset_viewer" / "app.py"
            if not flask_script.exists():
                raise FileNotFoundError(f"Flask script not found at {flask_script}")
            
            # Start Flask server
            self.server_process = subprocess.Popen(
                [sys.executable, str(flask_script)],
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(2)
            
            # Load initial page
            self.web_view.setUrl(QUrl("http://127.0.0.1:5003"))
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Server Error",
                f"Failed to start Flask server: {str(e)}"
            )
    
    def handle_close(self, event):
        """Handle window close event."""
        try:
            if self.server_process:
                # Send SIGTERM to Flask server
                if sys.platform == "win32":
                    self.server_process.terminate()
                else:
                    os.kill(self.server_process.pid, signal.SIGTERM)
                self.server_process.wait(timeout=5)
        except Exception as e:
            logging.error(f"Error shutting down Flask server: {e}")
        event.accept()