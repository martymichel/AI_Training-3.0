"""Main entry point for the KI Vision Tools application."""

import sys
from PyQt6.QtWidgets import QApplication

def main():
    """Main entry point with proper error handling."""
    try:
        app = QApplication(sys.argv)
        
        # Prüfen ob project_manager verfügbar ist
        try:
            from project_manager import ProjectManager
            use_project_system = True
        except ImportError:
            print("Project Manager nicht verfügbar - verwende klassische Version")
            use_project_system = False
        finally:
            pass  # Placeholder to fix indentation error
        
        # MainMenu importieren (funktioniert mit beiden Versionen)
        from gui.main_menu import MainMenu
        window = MainMenu()
        window.show()
        return app.exec()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())