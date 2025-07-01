# gui/gui_main.py
"""Main entry point for the KI Vision Tools application."""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from config import Config

def start_app():
    """Initialize and start the application."""
    # Ensure proper QApplication initialization with sys.argv
    app = QApplication(sys.argv)
    
    # Dark mode aus Config anwenden
    if Config.ui.dark_mode:
        app.setStyle("Fusion")
        palette = app.palette()
        palette.setColor(palette.Window, QColor(53, 53, 53))
        palette.setColor(palette.WindowText, Qt.GlobalColor.white)
        app.setPalette(palette)
    
    # Neue projektbasierte MainMenu verwenden
    from gui.main_menu import MainMenu
    window = MainMenu()
    
    # Nur anzeigen wenn Projekt erfolgreich geladen wurde
    if window.project_manager:
        window.show()
        return app.exec()
    else:
        # Kein Projekt geladen - App beenden
        return 0

def main():
    """Main entry point with proper error handling."""
    try:
        return start_app()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return 1