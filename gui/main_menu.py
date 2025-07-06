"""
Moderne, überarbeitete Main Menu Applikation
Sauberes, professionelles Design ohne Emojis
"""
import sys
import os
import subprocess
from pathlib import Path

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

class ModernCard(QFrame):
    """Moderne Workflow-Karte mit sauberem Design"""
    
    clicked = pyqtSignal(str)
    
    def __init__(self, title, description, icon_name, step_key, status="disabled"):
        super().__init__()
        self.title = title
        self.description = description
        self.icon_name = icon_name
        self.step_key = step_key
        self.status = status
        self.is_hovered = False
        self.icon_label = None
        
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
        
        # Icon laden und anzeigen
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(60, 60)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("background: transparent; border: none;")
        self.load_icon()
        
        # Title
        title_label = QLabel(self.title)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setWeight(QFont.Weight.Medium)
        title_label.setFont(title_font)
        title_label.setWordWrap(True)
        title_label.setStyleSheet("background: transparent;")
        
        # Description
        desc_label = QLabel(self.description)
        desc_font = QFont()
        desc_font.setPointSize(11)
        desc_label.setFont(desc_font)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        desc_label.setStyleSheet("background: transparent;")
        
        # Layout mit Icon-Platzierung
        layout.addWidget(status_widget)
        
        # Horizontales Layout für Titel und Icon
        title_icon_layout = QHBoxLayout()
        title_icon_layout.setContentsMargins(0, 0, 0, 0)
        title_icon_layout.addWidget(title_label)
        title_icon_layout.addStretch()
        title_icon_layout.addWidget(self.icon_label)
        
        title_icon_widget = QWidget()
        title_icon_widget.setLayout(title_icon_layout)
        title_icon_widget.setStyleSheet("background: transparent;")
        
        layout.addWidget(title_icon_widget)
        layout.addWidget(desc_label)
        layout.addStretch()
        
        # Click handling
        self.setCursor(Qt.CursorShape.PointingHandCursor if self.status != "disabled" else Qt.CursorShape.ForbiddenCursor)
    
    def load_icon(self):
        """Lädt das Icon als QPixmap"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        icon_path = os.path.join(project_root, "icons", f"{self.icon_name}.png")
        
        if os.path.exists(icon_path):
            print(f"DEBUG: Icon gefunden: {icon_path}")
            pixmap = QPixmap(icon_path)
            
            if not pixmap.isNull():
                # Icon skalieren auf 50x50 Pixel
                scaled_pixmap = pixmap.scaled(
                    50, 50,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Icon entsprechend dem Status einfärben/transparent machen
                if self.status == "disabled":
                    # Icon halbtransparent machen für disabled Status
                    painter = QPainter(scaled_pixmap)
                    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
                    painter.fillRect(scaled_pixmap.rect(), QColor(0, 0, 0, 128))
                    painter.end()
                
                self.icon_label.setPixmap(scaled_pixmap)
                print(f"DEBUG: Icon erfolgreich geladen für {self.icon_name}")
            else:
                print(f"WARNING: Pixmap konnte nicht erstellt werden für {icon_path}")
                self.set_fallback_icon()
        else:
            print(f"WARNING: Icon-Datei nicht gefunden: {icon_path}")
            self.set_fallback_icon()
    
    def set_fallback_icon(self):
        """Setzt ein Fallback-Icon wenn das originale nicht gefunden wird"""
        fallback_icons = {
            "camera": "📷",
            "labeling": "🏷️",
            "augmentation": "🔄",
            "splitting": "📊",
            "training": "🧠",
            "verification": "✅",
            "detection": "👁️"
        }
        
        emoji = fallback_icons.get(self.icon_name, "📦")
        self.icon_label.setText(emoji)
        self.icon_label.setStyleSheet("""
            background: transparent;
            border: none;
            font-size: 32px;
            color: #6c757d;
        """)
    
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
            ModernCard {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 12px;
            }}
            ModernCard:hover {{
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
        
        # Icon neu laden wenn sich der Status ändert
        if self.icon_label:
            self.load_icon()
    
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
        # Icon neu laden für Status-spezifische Darstellung
        if self.icon_label:
            self.load_icon()

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

class MainMenu(QMainWindow):
    """Moderne Hauptmenü-Applikation"""
    
    def __init__(self):
        super().__init__()
        self.project_manager = None
        self.windows = {}
        self.workflow_sections = []
        self.camera_process = None
        self.detection_process = None
        
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
            "1. Datenerfassung",
            "Sammeln und Organisieren von Bilddaten für das Training"
        )
        
        camera_card = ModernCard(
            "1.1 Bild Erfassung",
            "Bilder aufnehmen mit integrierter Kamera oder Webcam",
            "camera",
            "camera",
            self.get_card_status(WorkflowStep.CAMERA)
        )
        camera_card.clicked.connect(self.open_camera)
        
        data_section.add_card(camera_card)
        data_section.add_stretch()

        # 2. Datenbearbeitung
        processing_section = WorkflowSection(
            "2. Datenbearbeitung", 
            "Annotieren und Erweitern der Datenbasis"
        )
        
        labeling_card = ModernCard(
            "2.1 Labeling",
            "Objekten in Bildern markieren (Labeling)",
            "labeling",
            "labeling",
            self.get_card_status(WorkflowStep.LABELING)
        )
        labeling_card.clicked.connect(self.open_labeling)
        
        augmentation_card = ModernCard(
            "2.2 Augmentation",
            "Erweitern des Datensatzes durch Bildtransformationen",
            "augmentation",
            "augmentation",
            self.get_card_status(WorkflowStep.AUGMENTATION)
        )
        augmentation_card.clicked.connect(self.open_augmentation)

        label_checker_card = ModernCard(
            "2.3 Label Check",
            "Labels überprüfen und bereinigen",
            "labeling",
            "label_checker",
            self.get_card_status(WorkflowStep.SPLITTING)
        )
        label_checker_card.clicked.connect(self.open_label_checker)
        
        splitting_card = ModernCard(
            "2.4 Dataset Splitting",
            "Aufteilen in Training-, Validierungs- und Test-Sets",
            "splitting",
            "splitting",
            self.get_card_status(WorkflowStep.SPLITTING)
        )
        splitting_card.clicked.connect(self.open_splitter)
        
        processing_section.add_card(labeling_card)
        processing_section.add_card(augmentation_card)
        processing_section.add_card(label_checker_card)
        processing_section.add_card(splitting_card)
        
        # 3. Modellentwicklung
        model_section = WorkflowSection(
            "3. Modellentwicklung",
            "Training und Optimierung des KI-Modells"
        )
        
        training_card = ModernCard(
            "3.1 Training",
            "Trainieren des YOLO-Modells mit den vorbereiteten Daten",
            "training",
            "training",
            self.get_card_status(WorkflowStep.TRAINING)
        )
        training_card.clicked.connect(self.open_training)
        
        verification_card = ModernCard(
            "3.2 Verifikation",
            "Bewertung und Validierung der Modell-Performance",
            "verification",
            "verification",
            self.get_card_status(WorkflowStep.VERIFICATION)
        )
        verification_card.clicked.connect(self.open_verification)
        
        model_section.add_card(training_card)
        model_section.add_card(verification_card)
        model_section.add_stretch()
        
        # 4. Anwendung
        application_section = WorkflowSection(
            "4. Anwendung Live testen",
            "Einsatz des trainierten Modells in der Praxis mit IDS nxt Kamera"
        )
        
        detection_card = ModernCard(
            "4.1 Live Detection",
            "Echtzeit-Objekterkennung mit dem trainierten Modell",
            "detection",
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
                    'label_checker': WorkflowStep.SPLITTING,
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
        """Öffnet die neue Kamera-Anwendung"""
        if self.camera_process is None or self.camera_process.poll() is not None:
            settings_dir = str(self.project_manager.project_root)
            self.camera_process = subprocess.Popen([
                sys.executable,
                "-m",
                "gui.camera_app",
                settings_dir,
            ])
        else:
            QMessageBox.information(self, "Info", "Die Kamera-Anwendung läuft bereits.")

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

    def open_label_checker(self):
        """Öffnet Label-Checker mit Projekt-Kontext"""
        if 'label_checker' not in self.windows:
            from gui.label_checker_app import FastYOLOChecker
            app = FastYOLOChecker()
            app.project_manager = self.project_manager

            dataset_dir = self.project_manager.get_augmented_dir()
            if not list(dataset_dir.glob("*.jpg")) and not list(dataset_dir.glob("*.png")):
                dataset_dir = self.project_manager.get_labeled_dir()

            if dataset_dir.exists():
                app.dataset_path = str(dataset_dir)
                app.load_dataset()

            self.windows['label_checker'] = app

        self.windows['label_checker'].show()        
    
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
        """Startet die Tkinter-basierte Live-Detection"""
        if self.detection_process is None or self.detection_process.poll() is not None:
            settings_dir = str(self.project_manager.project_root)
            self.detection_process = subprocess.Popen([
                sys.executable,
                "-m",
                "gui.camera_app",
                settings_dir,
                "--show-detection",
            ])
        else:
            QMessageBox.information(self, "Info", "Die Live-Detection läuft bereits.")

        self.project_manager.mark_step_completed(WorkflowStep.LIVE_DETECTION)
        self.update_workflow_status()
    
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
    window = MainMenu()
    window.show()
    sys.exit(app.exec())