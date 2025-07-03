#!/usr/bin/env python3
"""
Debug-Script zur Diagnose von Import-Problemen
"""

import sys
import traceback

def test_imports():
    """Teste alle kritischen Imports systematisch."""
    
    print("🔍 Teste Python-Basis-Imports...")
    try:
        from typing import Dict, List, Optional, Tuple
        print("✅ typing imports OK")
    except ImportError as e:
        print(f"❌ typing import failed: {e}")
        return False
    
    print("\n🔍 Teste PyQt6-Imports...")
    try:
        from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow
        from PyQt6.QtCore import Qt, QThread, pyqtSignal
        from PyQt6.QtGui import QFont
        print("✅ PyQt6 imports OK")
    except ImportError as e:
        print(f"❌ PyQt6 import failed: {e}")
        print("   → Installiere PyQt6: pip install PyQt6")
        return False
    
    print("\n🔍 Teste Standard-Library-Imports...")
    try:
        import json
        import yaml
        import logging
        import shutil
        from datetime import datetime
        from pathlib import Path
        from dataclasses import dataclass, asdict
        from enum import Enum
        print("✅ Standard library imports OK")
    except ImportError as e:
        print(f"❌ Standard library import failed: {e}")
        return False
    
    print("\n🔍 Teste Project-Manager-Import...")
    try:
        from project_manager import WorkflowStep, ProjectConfig, ProjectManager
        print("✅ project_manager core imports OK")
    except ImportError as e:
        print(f"❌ project_manager import failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False
    except Exception as e:
        print(f"❌ project_manager error: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False
    
    print("\n🔍 Teste GUI-Imports...")
    try:
        from gui.main_menu import MainMenu
        print("✅ gui.main_menu import OK")
    except ImportError as e:
        print(f"❌ gui.main_menu import failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False
    except Exception as e:
        print(f"❌ gui.main_menu error: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False
    
    print("\n✅ Alle kritischen Imports erfolgreich!")
    return True

def main():
    print("🧪 AI Vision Tools - Import-Diagnose")
    print("=" * 50)
    
    if test_imports():
        print("\n🚀 Starte Applikation...")
        try:
            from PyQt6.QtWidgets import QApplication
            from gui.main_menu import MainMenu
            
            app = QApplication(sys.argv)
            window = MainMenu()
            
            if hasattr(window, 'project_manager') and window.project_manager:
                window.show()
                print("✅ Applikation erfolgreich gestartet!")
                return app.exec()
            else:
                print("❌ Kein Projekt geladen - App beendet")
                return 0
                
        except Exception as e:
            print(f"❌ Fehler beim Starten der Applikation: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return 1
    else:
        print("\n❌ Import-Probleme erkannt. Bitte behebe die Fehler vor dem Start.")
        return 1

if __name__ == "__main__":
    sys.exit(main())