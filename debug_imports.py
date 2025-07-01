# gui/gui_main.py - Korrigierte Version
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


# ==================== ALTERNATIVE: Minimale main.py Korrektur ====================

# main.py - Schnelle Korrektur ohne gui_main.py zu ändern
"""Main entry point for the KI Vision Tools application."""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

def main():
    """Main entry point with proper error handling."""
    try:
        # QApplication initialisieren
        app = QApplication(sys.argv)
        
        # Neue projektbasierte MainMenu direkt importieren
        from gui.main_menu import MainMenu
        window = MainMenu()
        
        # Nur anzeigen wenn Projekt erfolgreich geladen wurde
        if hasattr(window, 'project_manager') and window.project_manager:
            window.show()
            return app.exec()
        else:
            print("Kein Projekt ausgewählt - App wird beendet")
            return 0
            
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())


# ==================== IMPORT-PROBLEM LÖSUNG ====================

# Fügen Sie diese Zeile am Anfang der gui/main_menu.py hinzu:
import sys
import os

# Projekt-Root zum Python-Path hinzufügen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dann erst die Projekt-Manager Imports
try:
    from project_manager import ProjectManager, ProjectManagerDialog, WorkflowStatusWidget, WorkflowStep
except ImportError as e:
    print(f"Fehler beim Importieren des Project Managers: {e}")
    print("Bitte stellen Sie sicher, dass project_manager.py im Hauptverzeichnis liegt")
    sys.exit(1)


# ==================== DEBUGGING HILFE ====================

# Temporäres Debug-Script: debug_imports.py
"""
Erstellen Sie diese Datei im Hauptverzeichnis zum Testen der Imports
"""

import sys
import os

print("Python Version:", sys.version)
print("Aktuelles Verzeichnis:", os.getcwd())
print("Python Path:", sys.path)

# Test: Können wir project_manager importieren?
try:
    from project_manager import ProjectManager
    print("✅ project_manager erfolgreich importiert")
except ImportError as e:
    print(f"❌ Import-Fehler project_manager: {e}")

# Test: Können wir GUI Module importieren?
try:
    from gui.main_menu import MainMenu
    print("✅ gui.main_menu erfolgreich importiert")
except ImportError as e:
    print(f"❌ Import-Fehler gui.main_menu: {e}")

# Test: PyQt6 verfügbar?
try:
    from PyQt6.QtWidgets import QApplication
    print("✅ PyQt6 erfolgreich importiert")
except ImportError as e:
    print(f"❌ Import-Fehler PyQt6: {e}")

print("\nDateien im Hauptverzeichnis:")
for file in os.listdir('.'):
    if file.endswith('.py'):
        print(f"  📄 {file}")

print("\nDateien im gui/ Verzeichnis:")
gui_path = os.path.join('.', 'gui')
if os.path.exists(gui_path):
    for file in os.listdir(gui_path):
        if file.endswith('.py'):
            print(f"  📄 gui/{file}")
else:
    print("  ❌ gui/ Verzeichnis nicht gefunden")


# ==================== SCHRITT-FÜR-SCHRITT LÖSUNG ====================

"""
1. SCHRITT: Prüfen Sie die Dateistruktur
   Ihr Hauptverzeichnis sollte enthalten:
   - main.py
   - project_manager.py  ← WICHTIG: Diese Datei muss vorhanden sein
   - gui/
     ├── main_menu.py
     ├── __init__.py
     └── ... andere GUI-Dateien

2. SCHRITT: project_manager.py erstellen
   Kopieren Sie den kompletten Inhalt aus dem ersten Artifact in diese Datei

3. SCHRITT: main.py ersetzen
   Ersetzen Sie Ihre main.py mit der minimalen Version oben

4. SCHRITT: gui/main_menu.py anpassen
   Fügen Sie die Import-Korrektur am Anfang der Datei hinzu

5. SCHRITT: Testen
   python debug_imports.py
   Sollte alle ✅ anzeigen

6. SCHRITT: App starten
   python main.py
   Sollte jetzt den Projekt-Manager Dialog öffnen
"""