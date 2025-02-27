"""Main entry point for the KI Vision Tools application."""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from config import Config
from gui.main_menu import MainMenu

def start_app():
    """Initialize and start the application."""
    # Ensure proper QApplication initialization with sys.argv
    app = QApplication(sys.argv)
    
    # Dark mode aus Config anwenden
    if Config.ui.dark_mode:
        app.setStyle("Fusion")
        palette = app.palette()
        palette.setColor(palette.Window, QColor(53, 53, 53))
        palette.setColor(palette.WindowText, Qt.white)
        app.setPalette(palette)
    
    window = MainMenu()
    window.show()
    return app.exec()

def main():
    """Main entry point with proper error handling."""
    try:
        return start_app()
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1
