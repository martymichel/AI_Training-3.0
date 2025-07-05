"""
Moderne, überarbeitete Main Menu Applikation
Sauberes, professionelles Design ohne Emojis
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
    QFrame, QScrollArea, QGridLayout, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QFont, QPixmap, QAction, QPainter, QPen, QColor, QBrush
from PyQt6.QtWidgets import QDialog, QMessageBox, QGraphicsDropShadowEffect

class MainMenu(QFrame):
    """Moderne Workflow-Karte mit sauberem Design"""
    
    clicked = pyqtSignal(str)
    
    def __init__(self, title, description, step_key, status="disabled"):
        super().__init__()
        self.title = title
        self.description = description
        self.step_key = step_key
        self.status = status
        self.is_hovered = False
        
        self.init_ui()
        self.set_style()
    
    def init_ui(self):
        self.setFixedSize(280, 160)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        
        # Shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Status indicator
        status_widget = QWidget()
        status_widget.setFixedHeight(6)
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_bar = QFrame()
        self.status_bar.setFixedHeight(4)
        self.status_bar.setFrameStyle(QFrame.Shape.NoFrame)
        status_layout.addWidget(self.status_bar)
        status_layout.addStretch()
        
        # Title
        title_label = QLabel(self.title)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setWeight(QFont.Weight.Medium)
        title_label.setFont(title_font)
        title_label.setWordWrap(True)
        
        # Description
        desc_label = QLabel(self.description)
        desc_font = QFont()
        desc_font.setPointSize(11)
        desc_label.setFont(desc_font)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        layout.addWidget(status_widget)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()
        
        # Click handling
        self.setCursor(Qt.CursorShape.PointingHandCursor if self.status != "disabled" else Qt.CursorShape.ForbiddenCursor)
    
    def set_style(self):
        """Setzt das Styling basierend auf dem Status"""
        if self.status == "ready":
            bg_color = "#ffffff"
            border_color = "#22c55e"
            status_color = "#22c55e"
            text_color = "#111827"
        elif self.status == "completed":
            bg_color = "#ffffff" 
            border_color = "#3b82f6"
            status_color = "#3b82f6"
            text_color = "#111827"
        else:  # disabled
            bg_color = "#f9fafb"
            border_color = "#e5e7eb"
            status_color = "#d1d5db"
            text_color = "#6b7280"
        
        self.setStyleSheet(f"""
            MainMenu {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 12px;
            }}
            MainMenu:hover {{
                border-color: {"#16a34a" if self.status == "ready" else "#2563eb" if self.status == "completed" else border_color};
            }}
            QLabel {{
                color: {text_color};
                background: transparent;
                border: none;
            }}
        """)
        
        self.status_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {status_color};
                border-radius: 2px;
            }}
        """)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.status in ["ready", "completed"]:
                self.clicked.emit(self.step_key)
            elif self.status == "disabled":
                self.show_disabled_message()
    
    def show_disabled_message(self):
        """Zeigt Nachricht für deaktivierte Karten"""
        step_messages = {
            "camera": "Die Kamera ist immer verfügbar zum Sammeln von Bilddaten.",
            "labeling": "Labeling ist verfügbar, sobald Bilder vorhanden sind.\n\nBitte nehmen Sie zuerst Bilder auf.",
            "augmentation": "Augmentation ist verfügbar, sobald gelabelte Daten vorhanden sind.\n\nBitte führen Sie zuerst das Labeling durch.",
            "splitting": "Dataset Splitting ist verfügbar, sobald Daten vorhanden sind.\n\nBitte führen Sie zuerst Labeling durch.",
            "training": "Training ist verfügbar, sobald ein Dataset vorhanden ist.\n\nBitte führen Sie zuerst Dataset Splitting durch.",
            "verification": "Verifikation ist verfügbar, sobald ein Modell trainiert wurde.\n\nBitte führen Sie zuerst das Training durch.",
            "detection": "Live Detection ist verfügbar, sobald ein Modell verifiziert wurde.\n\nBitte führen Sie zuerst Training und Verifikation durch."
        }
        
        message = step_messages.get(self.step_key, "Diese Funktion ist noch nicht verfügbar.")
        
        QMessageBox.information(
            self, 
            f"{self.title} - Noch nicht verfügbar", 
            message
        )
    
    def update_status(self, new_status):
        """Aktualisiert den Status der Karte"""
        self.status = new_status
        self.set_style()
        self.setCursor(Qt.CursorShape.PointingHandCursor if self.status != "disabled" else Qt.CursorShape.ForbiddenCursor)

class WorkflowSection(QFrame):
    """Moderne Workflow-Sektion"""
    
    def __init__(self, title, description):
        super().__init__()
        self.title = title
        self.description = description
        self.cards = []
        
        self.init_ui()
    
    def init_ui(self):
        self.setFrameStyle(QFrame.Shape.NoFrame)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Header
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)
        
        # Title
        title_label = QLabel(self.title)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setWeight(QFont.Weight.Bold)
        title_label.setFont(title_font)
        
        # Description
        desc_label = QLabel(self.description)
        desc_font = QFont()
        desc_font.setPointSize(12)
        desc_label.setFont(desc_font)
        desc_label.setStyleSheet("color: #6b7280;")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(desc_label)
        
        # Cards container
        cards_widget = QWidget()
        self.cards_layout = QHBoxLayout(cards_widget)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(20)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        layout.addWidget(header_widget)
        layout.addWidget(cards_widget)
        
        # Styling
        self.setStyleSheet("""
            WorkflowSection {
                background-color: white;
                border-radius: 16px;
                border: 1px solid #e5e7eb;
            }
            QLabel {
                background: transparent;
                border: none;
                color: #111827;
            }
        """)
        
        # Shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 10))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
    
    def add_card(self, card):
        self.cards.append(card)
        self.cards_layout.addWidget(card)
    
    def add_stretch(self):
        self.cards_layout.addStretch()

class ModernMainMenu(QMainWindow):
    """Moderne Hauptmenü-Applikation"""
    
    def __init__(self):
        super().__init__()
        self.project_manager = None
        self.windows = {}
        self.workflow_sections = []
        
        # Projekt-Manager beim Start öffnen
        self.init_project()
        
        if self.project_manager:
            self.init_ui()
        else:
            sys.exit()
    
    def init_project(self):
        """Initialisiert Projekt-Management"""
        dialog = ProjectManagerDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            project_path = dialog.get_selected_project()
            if project_path:
                self.load_project(project_path)
        else:
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
        """Initialisiert die moderne Benutzeroberfläche"""
        self.setWindowTitle(f"AI Vision Tools - {self.project_manager.config.project_name}")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
        # Hauptstil
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f3f4f6;
            }
            QMenuBar {
                background-color: white;
                border-bottom: 1px solid #e5e7eb;
                padding: 8px 0;
                font-size: 13px;
            }
            QMenuBar::item {
                padding: 8px 16px;
                color: #374151;
                font-weight: 500;
            }
            QMenuBar::item:selected {
                background-color: #f3f4f6;
                color: #111827;
            }
            QMenu {
                background-color: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 8px 0;
            }
            QMenu::item {
                padding: 8px 16px;
                color: #374151;
            }
            QMenu::item:selected {
                background-color: #f3f4f6;
                color: #111827;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f3f4f6;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #d1d5db;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9ca3af;
            }
        """)
        
        # Menu Bar
        self.create_menu_bar()
        
        # Hauptlayout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Projekt-Header
        self.create_project_header(main_layout)
        
        # Scroll Area für Workflow
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 20, 0, 40)
        scroll_layout.setSpacing(30)
        
        # Workflow-Sektionen
        self.create_workflow_sections(scroll_layout)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        
        # Footer
        self.create_footer(main_layout)
    
    def create_menu_bar(self):
        """Erstellt moderne Menu Bar"""
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
        
        # Extras Menu
        extras_menu = menubar.addMenu('Extras')
        
        dataset_action = QAction('Dataset Viewer', self)
        dataset_action.triggered.connect(self.open_dataset_viewer)
        extras_menu.addAction(dataset_action)
        
        dashboard_action = QAction('Training Dashboard', self)
        dashboard_action.triggered.connect(self.open_dashboard)
        extras_menu.addAction(dashboard_action)
        
        # Hilfe Menu
        help_menu = menubar.addMenu('Hilfe')
        
        about_action = QAction('Über AI Vision Tools', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_project_header(self, layout):
        """Erstellt modernen Projekt-Header"""
        header_widget = QWidget()
        header_widget.setFixedHeight(120)
        
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(40, 30, 40, 30)
        header_layout.setSpacing(8)
        
        # Projekt-Name
        project_name = QLabel(self.project_manager.config.project_name)
        project_font = QFont()
        project_font.setPointSize(28)
        project_font.setWeight(QFont.Weight.Bold)
        project_name.setFont(project_font)
        project_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Projekt-Details
        created_date = self.project_manager.config.created_date[:10]
        modified_date = self.project_manager.config.last_modified[:10]
        
        details = QLabel(f"Erstellt: {created_date} • Letzte Änderung: {modified_date}")
        details_font = QFont()
        details_font.setPointSize(12)
        details.setFont(details_font)
        details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(project_name)
        header_layout.addWidget(details)
        
        # Styling
        header_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-bottom: 1px solid #e5e7eb;
            }
            QLabel {
                color: #111827;
                background: transparent;
                border: none;
            }
            QLabel:last-child {
                color: #6b7280;
            }
        """)
        
        layout.addWidget(header_widget)
    
    def create_workflow_sections(self, layout):
        """Erstellt die modernen Workflow-Sektionen"""
        
        # 1. Datenerfassung
        data_section = WorkflowSection(
            "Datenerfassung",
            "Sammeln und Organisieren von Bilddaten für das Training"
        )
        
        camera_card = MainMenu(
            "Kamera",
            "Bilder aufnehmen mit integrierter Kamera oder Webcam",
            "camera",
            self.get_card_status(WorkflowStep.CAMERA)
        )
        camera_card.clicked.connect(self.open_camera)
        
        data_section.add_card(camera_card)
        data_section.add_stretch()

        # 2. Datenbearbeitung
        processing_section = WorkflowSection(
            "Datenbearbeitung", 
            "Annotieren und Erweitern der Datenbasis"
        )
        
        labeling_card = MainMenu(
            "Labeling",
            "Manuelle Annotation von Objekten in Bildern",
            "labeling",
            self.get_card_status(WorkflowStep.LABELING)
        )
        labeling_card.clicked.connect(self.open_labeling)
        
        augmentation_card = MainMenu(
            "Augmentation",
            "Erweitern des Datensatzes durch Bildtransformationen",
            "augmentation",
            self.get_card_status(WorkflowStep.AUGMENTATION)
        )
        augmentation_card.clicked.connect(self.open_augmentation)
        
        splitting_card = MainMenu(
            "Dataset Splitting",
            "Aufteilen in Training-, Validierungs- und Test-Sets",
            "splitting",
            self.get_card_status(WorkflowStep.SPLITTING)
        )
        splitting_card.clicked.connect(self.open_splitter)
        
        processing_section.add_card(labeling_card)
        processing_section.add_card(augmentation_card)
        processing_section.add_card(splitting_card)
        
        # 3. Modellentwicklung
        model_section = WorkflowSection(
            "Modellentwicklung",
            "Training und Optimierung des KI-Modells"
        )
        
        training_card = MainMenu(
            "Training",
            "Trainieren des YOLO-Modells mit den vorbereiteten Daten",
            "training",
            self.get_card_status(WorkflowStep.TRAINING)
        )
        training_card.clicked.connect(self.open_training)
        
        verification_card = MainMenu(
            "Verifikation",
            "Bewertung und Validierung der Modell-Performance",
            "verification",
            self.get_card_status(WorkflowStep.VERIFICATION)
        )
        verification_card.clicked.connect(self.open_verification)
        
        model_section.add_card(training_card)
        model_section.add_card(verification_card)
        model_section.add_stretch()
        
        # 4. Anwendung
        application_section = WorkflowSection(
            "Anwendung",
            "Einsatz des trainierten Modells in der Praxis"
        )
        
        detection_card = MainMenu(
            "Live Detection",
            "Echtzeit-Objekterkennung mit dem trainierten Modell",
            "detection",
            self.get_card_status(WorkflowStep.LIVE_DETECTION)
        )
        detection_card.clicked.connect(self.open_detection)
        
        application_section.add_card(detection_card)
        application_section.add_stretch()

        # Sektionen hinzufügen
        layout.addWidget(data_section)
        layout.addWidget(processing_section)
        layout.addWidget(model_section)
        layout.addWidget(application_section)
        
        # Referenzen speichern
        self.workflow_sections = [data_section, processing_section, model_section, application_section]
    
    def get_card_status(self, step):
        """Ermittelt Status einer Karte"""
        if not self.project_manager:
            return "disabled"
        
        can_execute, _ = self.project_manager.validate_workflow_step(step)
        is_completed = self.project_manager.is_step_completed(step)
        
        if is_completed:
            return "completed"
        elif can_execute:
            return "ready"
        else:
            return "disabled"
    
    def create_footer(self, layout):
        """Erstellt modernen Footer"""
        footer_widget = QWidget()
        footer_widget.setFixedHeight(60)
        
        footer_layout = QVBoxLayout(footer_widget)
        footer_layout.setContentsMargins(40, 20, 40, 20)
        
        footer_label = QLabel("AI Vision Tools • Entwickelt von Michel Marty für Flex Precision Plastic Solutions AG © 2025")
        footer_font = QFont()
        footer_font.setPointSize(10)
        footer_label.setFont(footer_font)
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        footer_layout.addWidget(footer_label)
        
        footer_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-top: 1px solid #e5e7eb;
            }
            QLabel {
                color: #6b7280;
                background: transparent;
                border: none;
            }
        """)
        
        layout.addWidget(footer_widget)
    
    def update_workflow_status(self):
        """Aktualisiert Status aller Workflow-Karten"""
        for section in self.workflow_sections:
            for card in section.cards:
                step_map = {
                    'camera': WorkflowStep.CAMERA,
                    'labeling': WorkflowStep.LABELING,
                    'augmentation': WorkflowStep.AUGMENTATION,
                    'splitting': WorkflowStep.SPLITTING,
                    'training': WorkflowStep.TRAINING,
                    'verification': WorkflowStep.VERIFICATION,
                    'detection': WorkflowStep.LIVE_DETECTION
                }
                
                if card.step_key in step_map:
                    step = step_map[card.step_key]
                    new_status = self.get_card_status(step)
                    card.update_status(new_status)
    
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
        self.update_workflow_status()
    
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
            try:
                from gui.augmentation_preview import load_sample_image
                load_sample_image(app)
            except ImportError:
                pass

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
                self.update_workflow_status()
            else:
                self.close()
    
    def open_continual_learning(self):
        """Öffnet Continual Learning Dialog"""
        from project_manager import ContinualTrainingDialog
        dialog = ContinualTrainingDialog(self.project_manager, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Workflow-Status aktualisieren nach erfolgreichem Training
            self.update_workflow_status()
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

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = ModernMainMenu()
    window.show()
    sys.exit(app.exec())