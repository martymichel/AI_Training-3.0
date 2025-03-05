"""Utility functions for model verification."""

import os
import logging
from PyQt6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)

def validate_model_path(model_path: str) -> bool:
    """Validate model file path."""
    if not os.path.exists(model_path):
        QMessageBox.warning(None, "Error", "Bitte wählen Sie eine gültige Modell-Datei aus")
        return False
    return True

def validate_test_folder(test_folder: str) -> bool:
    """Validate test dataset folder."""
    if not os.path.isdir(test_folder):
        QMessageBox.warning(None, "Error", "Bitte wählen Sie ein gültiges Testverzeichnis aus")
        return False
    return True

def get_model_status(accuracy: float) -> tuple:
    """Get model status based on accuracy."""
    if accuracy >= 98:
        return "#4CAF50", "KI-MODELL SEHR GUT"  # Green
    elif accuracy >= 95:
        return "#FF9800", "KI-MODELL AKZEPTABEL"  # Orange
    return "#F44336", "KI-MODELL UNGENÜGEND"  # Red

def open_directory(dir_path: str) -> None:
    """Open directory in file explorer."""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(dir_path)
        else:  # Linux/Mac
            os.system(f'xdg-open "{dir_path}"')
    except Exception as e:
        logger.error(f"Error opening directory: {e}")
        QMessageBox.warning(None, "Error", f"Fehler beim Öffnen des Verzeichnisses: {e}")

def format_summary(total_images: int, good_count: int, bad_count: int, misannotated_dir: str) -> str:
    """Format summary text."""
    return (f"Live Annotation abgeschlossen.\n"
            f"Gesamtbilder: {total_images}\n"
            f"Korrekt annotiert: {good_count}\n"
            f"Falsch annotiert: {bad_count}\n"
            f"Falsch annotierte Bilder im Ordner: {misannotated_dir}")

def setup_logging(misannotated_dir: str) -> None:
    """Set up logging configuration."""
    log_file = os.path.join(misannotated_dir, "live_annotation.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)