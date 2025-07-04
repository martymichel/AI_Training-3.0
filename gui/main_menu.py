# ==================== UPDATE FÜR MAIN_MENU.PY ====================

"""
Ersetze die bestehende MainMenu-Klasse in gui/main_menu.py mit dieser erweiterten Version
"""
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

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSizePolicy, QSpacerItem, QStyle, QMenuBar, QMenu
)

from gui.augmentation_preview import load_sample_image
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QPixmap, QIcon, QAction
from PyQt6.QtWidgets import QDialog, QMessageBox, QGridLayout
APP_STYLE = """
QMainWindow {
    background-color: #2b2b2b;
    color: #f0f0f0;
}
QMenuBar {
    background-color: #333;
}
QMenuBar::item {
    padding: 4px 12px;
}
QMenuBar::item:selected {
    background-color: #444;
}
QPushButton {
    background-color: #444;
    color: #f0f0f0;
    padding: 12px;
    border-radius: 6px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #555;
}
"""


class MainMenu(QMainWindow):
    """Erweiterte Main Menu Klasse mit Projekt-Management"""
    
    def __init__(self):
        super().__init__()
        self.project_manager = None
        self.windows = {}
        
        # Projekt-Manager beim Start öffnen
        self.init_project()
        
        if self.project_manager:
            self.init_ui()
        else:
            sys.exit()  # Beenden wenn kein Projekt gewählt wurde
    
    def init_project(self):
        """Initialisiert Projekt-Management"""
        dialog = ProjectManagerDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            project_path = dialog.get_selected_project()
            if project_path:
                self.load_project(project_path)
        else:
            # Kein Projekt gewählt - App beenden
            return
    
    def load_project(self, project_path: str):
        """Lädt ein Projekt"""
        try:
            self.project_manager = ProjectManager(project_path)
            print(f"Projekt geladen: {self.project_manager.config.project_name}")
        except Exception as e:
            QMessageBox.critical(
                self, "Fehler", 
                f"Projekt konnte nicht geladen werden:\n{str(e)}"
            )
    
    def init_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        self.setWindowTitle(f"AI Vision Tools - {self.project_manager.config.project_name}")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setStyleSheet(APP_STYLE)
        
        # Menu Bar hinzufügen
        self.create_menu_bar()
        
        # Zentrales Widget mit vertikalem Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(50, 30, 50, 30)
        
        # Logo und Header
        self.create_header(layout)
        
        # Workflow-Status Widget
        self.workflow_widget = WorkflowStatusWidget(self.project_manager)
        self.workflow_widget.step_clicked.connect(self.open_workflow_step)
        layout.addWidget(self.workflow_widget)
        
        layout.addSpacing(20)
        
        # Tool-Buttons (traditionelle Ansicht)
        self.create_tool_buttons(layout)
        layout.addStretch()
        
        # Footer
        self.create_footer(layout)
    
    def create_menu_bar(self):
        """Erstellt Menu Bar"""
        menubar = self.menuBar()
        
        # Projekt Menu
        project_menu = menubar.addMenu('Projekt')
        
        switch_action = QAction('Projekt wechseln...', self)
        switch_action.triggered.connect(self.switch_project)
        project_menu.addAction(switch_action)
        
        continual_action = QAction('Continual Learning...', self)
        continual_action.triggered.connect(self.open_continual_learning)
        project_menu.addAction(continual_action)
        
        project_menu.addSeparator()
        
        exit_action = QAction('Beenden', self)
        exit_action.triggered.connect(self.close)
        project_menu.addAction(exit_action)
        
        # Hilfe Menu
        help_menu = menubar.addMenu('Hilfe')
        
        about_action = QAction('Über AI Vision Tools', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_header(self, layout):
        """Erstellt den Kopfbereich"""
        title = QLabel("AI Vision Tools")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 28, QFont.Weight.Bold))

        subtitle = QLabel("by Michel Marty")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont("Arial", 14))

        project_info = QLabel(f"Aktuelles Projekt: {self.project_manager.config.project_name}")
        project_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        project_info.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        project_info.setStyleSheet("margin-bottom: 10px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(project_info)
    
    def create_tool_buttons(self, layout):
        """Erstellt die Tool-Buttons"""
        description = QLabel("Oder wählen Sie direkt ein Tool aus:")
        description.setFont(QFont("Arial", 14))
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)
        buttons = [
            ("Kamera", self.open_camera),
            ("Labeling", self.open_labeling),
            ("Augmentation", self.open_augmentation),
            ("Dataset Viewer", self.open_dataset_viewer),
            ("Dataset Splitter", self.open_splitter),
            ("Training", self.open_training),
            ("Verifikation", self.open_verification),
            ("Live Detection", self.open_detection),
            ("Dashboard", self.open_dashboard),
        ]
        grid = QGridLayout()
        grid.setSpacing(10)
        for i, (text, cb) in enumerate(buttons):
            btn = self.create_button(text, cb)
            row = i // 3
            col = i % 3
            grid.addWidget(btn, row, col)
        layout.addLayout(grid)

    
    def create_footer(self, layout):
        footer = QLabel("Application by Michel Marty for Flex Precision Plastic Solutions AG Switzerland © 2025")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer)

    def create_button(self, text, callback):
        btn = QPushButton(text)
        btn.setMinimumSize(180, 80)
        btn.clicked.connect(callback)
        return btn
    
    # ==================== WORKFLOW-STEP HANDLING ====================
    
    def open_workflow_step(self, step_value: str):
        """Öffnet entsprechendes Tool basierend auf Workflow-Schritt"""
        step = WorkflowStep(step_value)
        
        # Validierung vor dem Öffnen
        can_execute, message = self.project_manager.validate_workflow_step(step)
        if not can_execute:
            QMessageBox.warning(self, "Schritt nicht verfügbar", message)
            return
        
        # Entsprechendes Tool öffnen
        tool_mapping = {
            WorkflowStep.CAMERA: self.open_camera,
            WorkflowStep.LABELING: self.open_labeling,
            WorkflowStep.AUGMENTATION: self.open_augmentation,
            WorkflowStep.SPLITTING: self.open_splitter,
            WorkflowStep.TRAINING: self.open_training,
            WorkflowStep.VERIFICATION: self.open_verification,
            WorkflowStep.LIVE_DETECTION: self.open_detection
        }
        
        tool_mapping[step]()
    
    # ==================== TOOL-OPENING METHODS ====================
    
    def open_camera(self):
        """Öffnet Kamera-App mit Projekt-Kontext"""
        if 'camera' not in self.windows:
            from gui.camera_app import CameraApp
            app = CameraApp()
            app.project_manager = self.project_manager
            app.output_dir = str(self.project_manager.get_raw_images_dir())
            app.dir_label.setText(f"Output Directory: {app.output_dir}")
            self.windows['camera'] = app
        
        self.windows['camera'].show()
        self.project_manager.mark_step_completed(WorkflowStep.CAMERA)
        self.workflow_widget.update_status()
    
    def open_labeling(self):
        """Öffnet Labeling-App mit Projekt-Kontext"""
        if 'labeling' not in self.windows:
            from gui.image_labeling import ImageLabelingApp
            app = ImageLabelingApp()
            app.project_manager = self.project_manager
            app.source_dir = str(self.project_manager.get_raw_images_dir())
            app.dest_dir = str(self.project_manager.get_labeled_dir())
            
            # UI Labels aktualisieren
            app.lbl_source_dir.setText(f"Quellverzeichnis: {app.source_dir}")
            app.lbl_dest_dir.setText(f"Zielverzeichnis: {app.dest_dir}")
            
            # Klassen aus Projekt laden
            classes = self.project_manager.get_classes()
            colors = self.project_manager.get_class_colors()
            
            app.classes = []
            for class_id in sorted(classes.keys()):
                class_name = classes[class_id]
                color = colors.get(class_id, "#FF0000")
                from PyQt6.QtGui import QColor
                app.classes.append((class_name, QColor(color)))
            
            app.update_class_list()
            app.load_images()
            
            self.windows['labeling'] = app
        
        self.windows['labeling'].show()
    
    def open_augmentation(self):
        """Öffnet Augmentation-App mit Projekt-Kontext"""
        if 'augmentation' not in self.windows:
            from gui.augmentation_app import ImageAugmentationApp
            app = ImageAugmentationApp()
            app.project_manager = self.project_manager
            app.source_path = str(self.project_manager.get_labeled_dir())
            app.dest_path = str(self.project_manager.get_augmented_dir())
            
            # UI Labels aktualisieren
            app.source_label.setText(f"Quellverzeichnis: {app.source_path}")
            app.dest_label.setText(f"Zielverzeichnis: {app.dest_path}")

            # Gespeicherte Settings laden
            saved_settings = self.project_manager.get_augmentation_settings()
            if saved_settings:
                app.settings.update(saved_settings)

            # Count information and preview
            app.update_expected_count()
            load_sample_image(app)

            self.windows['augmentation'] = app
        
        self.windows['augmentation'].show()
    
    def open_dataset_viewer(self):
        """Öffnet Dataset-Viewer"""
        if 'dataset_viewer' not in self.windows:
            from gui.dataset_viewer import DatasetViewerApp
            app = DatasetViewerApp()
            # Auto-load projekt data
            labeled_dir = self.project_manager.get_labeled_dir()
            if labeled_dir.exists():
                app.dataset_path = str(labeled_dir)
                app.path_label.setText(f"Dataset: {labeled_dir}")
                app.analyze_dataset()
            self.windows['dataset_viewer'] = app
        
        self.windows['dataset_viewer'].show()
    
    def open_splitter(self):
        """Öffnet Dataset-Splitter mit Projekt-Kontext"""
        if 'splitter' not in self.windows:
            from gui.dataset_splitter import DatasetSplitterApp
            app = DatasetSplitterApp()
            app.project_manager = self.project_manager
            
            # Source: Augmented oder Labeled (fallback)
            aug_dir = self.project_manager.get_augmented_dir()
            labeled_dir = self.project_manager.get_labeled_dir()
            
            aug_files = list(aug_dir.glob("*.jpg")) + list(aug_dir.glob("*.png"))
            source_dir = str(aug_dir) if aug_files else str(labeled_dir)
            
            app.source_path.setText(source_dir)
            app.output_path.setText(str(self.project_manager.get_split_dir()))
            
            # Auto-analyze wenn möglich
            if source_dir:
                app.analyze_classes(source_dir)
                
                # Klassen-Namen vorbelegen
                classes = self.project_manager.get_classes()
                for class_id, class_name in classes.items():
                    if class_id in app.class_inputs:
                        app.class_inputs[class_id].setText(class_name)
            
            self.windows['splitter'] = app
        
        self.windows['splitter'].show()
    
    def open_training(self):
        """Öffnet Training-Window mit Projekt-Kontext"""
        if 'training' not in self.windows:
            from gui.training.settings_window import TrainSettingsWindow
            app = TrainSettingsWindow()
            app.project_manager = self.project_manager
            
            # Automatische Pfad-Setzung
            app.project_input.setText(str(self.project_manager.get_models_dir().parent))
            app.name_input.setText("training")
            app.data_input.setText(str(self.project_manager.get_yaml_file()))
            
            # Gespeicherte Settings laden
            saved_settings = self.project_manager.get_training_settings()
            if saved_settings:
                for key, value in saved_settings.items():
                    if hasattr(app, f"{key}_input"):
                        widget = getattr(app, f"{key}_input")
                        if hasattr(widget, 'setValue'):
                            widget.setValue(value)
                        elif hasattr(widget, 'setText'):
                            widget.setText(str(value))
                        elif hasattr(widget, 'setChecked'):
                            widget.setChecked(bool(value))
            
            self.windows['training'] = app
        
        self.windows['training'].show()
    
    def open_verification(self):
        """Öffnet Verification-App mit Projekt-Kontext"""
        if 'verification' not in self.windows:
            from gui.verification_app import LiveAnnotationApp
            app = LiveAnnotationApp()
            app.project_manager = self.project_manager
            
            # Automatische Pfad-Setzung
            model_path = self.project_manager.get_current_model_path()
            if not model_path:
                model_path = self.project_manager.get_latest_model_path()
            
            if model_path and model_path.exists():
                app.model_line_edit.setText(str(model_path))
            
            # Test-Verzeichnis setzen
            test_dir = self.project_manager.get_split_dir() / "test" / "images"
            if not test_dir.exists() or not list(test_dir.glob("*.jpg")):
                test_dir = self.project_manager.get_labeled_dir()
            
            app.folder_line_edit.setText(str(test_dir))
            
            self.windows['verification'] = app
        
        self.windows['verification'].show()
    
    def open_detection(self):
        """Öffnet Live-Detection mit Projekt-Kontext"""
        if 'detection' not in self.windows:
            from gui.live_detection import LiveDetectionApp
            app = LiveDetectionApp()
            app.project_manager = self.project_manager
            
            # Automatische Pfad-Setzung
            model_path = self.project_manager.get_current_model_path()
            if not model_path:
                model_path = self.project_manager.get_latest_model_path()
            
            if model_path and model_path.exists():
                app.model_path = str(model_path)
                app.model_path_label.setText(model_path.name)
            
            yaml_path = self.project_manager.get_yaml_file()
            if yaml_path.exists():
                try:
                    import yaml
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        yaml_data = yaml.safe_load(f)
                        app.class_names = yaml_data.get('names', {})
                        app.num_classes = len(app.class_names)
                        app.yaml_path_label.setText(yaml_path.name)
                except Exception as e:
                    print(f"Error loading YAML: {e}")
            
            app.check_ready()
            self.windows['detection'] = app
        
        self.windows['detection'].show()
    
    def open_dashboard(self):
        """Öffnet Training-Dashboard"""
        if 'dashboard' not in self.windows:
            from gui.gui_dashboard import DashboardWindow
            # Dashboard auf Projekt-Verzeichnis zeigen lassen
            models_dir = self.project_manager.get_models_dir()
            app = DashboardWindow(str(models_dir.parent), "training")
            self.windows['dashboard'] = app
        
        self.windows['dashboard'].show()
    
    # ==================== MENU ACTIONS ====================
    
    def switch_project(self):
        """Wechselt zu anderem Projekt"""
        reply = QMessageBox.question(
            self, "Projekt wechseln",
            "Möchten Sie zu einem anderen Projekt wechseln?\n"
            "Alle geöffneten Fenster werden geschlossen.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Alle Fenster schließen
            for window in self.windows.values():
                if hasattr(window, 'close'):
                    window.close()
            self.windows.clear()
            
            # Neues Projekt laden
            self.init_project()
            if self.project_manager:
                # UI neu aufbauen
                self.init_ui()
                # Workflow-Status aktualisieren
                self.workflow_widget.update_status()
            else:
                self.close()
    
    def open_continual_learning(self):
        """Öffnet Continual Learning Dialog"""
        from project_manager import ContinualTrainingDialog
        dialog = ContinualTrainingDialog(self.project_manager, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Workflow-Status aktualisieren nach erfolgreichem Training
            self.workflow_widget.update_status()
            QMessageBox.information(
                self, "Training aktualisiert",
                "Das Modell wurde erfolgreich nachtrainiert und ist jetzt aktiv."
            )
    
    def show_about(self):
        """Zeigt Über-Dialog"""
        QMessageBox.about(
            self, "Über AI Vision Tools",
            f"AI Vision Tools v2.0\n"
            f"Projektbasiertes System für industrielle Computer Vision\n\n"
            f"Aktuelles Projekt: {self.project_manager.config.project_name}\n"
            f"Erstellt: {self.project_manager.config.created_date[:10]}\n"
            f"Letzte Änderung: {self.project_manager.config.last_modified[:10]}\n\n"
            f"Entwickelt von Michel Marty\n"
            f"für Flex Precision Plastic Solutions AG\n"
            f"© 2025"
        )
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Alle Fenster schließen
        for window in self.windows.values():
            if hasattr(window, 'close'):
                window.close()
        event.accept()