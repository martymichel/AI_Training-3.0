from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QSpacerItem, QStyle
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QPixmap, QIcon
from gui.gui_settings import TrainSettingsWindow
from gui.augmentation_app import ImageAugmentationApp
from gui.dataset_viewer import DatasetViewerApp
from gui.verification_app import LiveAnnotationApp
from gui.dataset_splitter import DatasetSplitterApp
from gui.image_labeling import ImageLabelingApp
from gui.camera_app import CameraApp
from gui.live_detection import LiveDetectionApp
from gui.gui_dashboard import DashboardWindow

class MainMenu(QMainWindow):
    """Main menu window providing access to different applications."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Vision Tools")
        # Maximiertes Startfenster, unabhängig von der Bildschirmgröße
        self.setWindowState(Qt.WindowState.WindowMaximized)
        # Hintergrundfarbe des Hauptfensters
        self.setStyleSheet("background-color: #292b2f; color: white;")

        # Zentrales Widget mit vertikalem Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(200, 60, 200, 60)  # Abstand zu den Seitenrändern

        # Logo oben links
        logo_layout = QHBoxLayout()
        self.logo_label = QLabel()
        pixmap = QPixmap("img/logo.png")
        self.logo_label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        logo_layout.addWidget(self.logo_label)
        logo_layout.addStretch()
        layout.addLayout(logo_layout)
        
        # Header
        title = QLabel("AI Vision Tools")
        title_font = QFont("Arial", 32, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("by Michel Marty")
        subtitle.setFont(QFont("Arial", 18))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        description = QLabel("Willkommen bei den AI Vision Tools. Wähle eine Anwendung aus:")
        description.setFont(QFont("Arial", 14))
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        # Abstand zwischen "Willkommen-Text" und den Buttons
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addItem(spacer)
        
        # Button-Stil (zentriert, Hintergrundfarbe, Schriftart, Schriftgröße, Schriftfarbe, abgerundete Ecken)
        button_font = QFont("Arial", 14)
        button_style = """
            QPushButton {
                background-color: #165a69;
                color: white;
                padding: 20px;
                border-radius: 8px;
                font-weight: bold;
                min-width: 300px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #7ABF5A;
            }
        """

        # Buttons mit Icons
        layout.addWidget(self.create_button("1. Kamera Livestream / Bilder aufnehmen", self.open_camera,
            "Öffnet den Kamera-Livestream, um Bilder für das Training eines KI-Modells aufzunehmen.", QStyle.StandardPixmap.SP_DriveCDIcon))

        layout.addWidget(self.create_button("2. Bilder Labeln / Bounding Boxen markieren", self.open_labeling,
            "Ermöglicht das Annotieren von Bildern mit Bounding Boxen, ein essenzieller Schritt für YOLO-Modelle.", QStyle.StandardPixmap.SP_FileDialogDetailedView))

        # Augmentierung & Dataset Viewer nebeneinander
        row1 = QHBoxLayout()
        row1.addWidget(self.create_button("3. Bilder augmentieren", self.open_augmentation,
            "Erzeugt neue Bildvariationen durch Transformationen wie Drehen, Spiegeln und Skalieren.", QStyle.StandardPixmap.SP_ComputerIcon))
        row1.addWidget(self.create_button("Labels prüfen / Dataset Viewer", self.open_dataset_viewer,
            "Visualisiert und überprüft Bounding Boxen im Datensatz vor dem Training.", QStyle.StandardPixmap.SP_FileDialogInfoView))
        layout.addLayout(row1)


        layout.addWidget(self.create_button("4. Dataset Splitter (Train/Validation/Test)", self.open_splitter,
            "Teilt den Datensatz in Trainings-, Validierungs- und Testdaten auf.", QStyle.StandardPixmap.SP_DirIcon))


        # YOLO Trainer & Dashboard nebeneinander
        row2 = QHBoxLayout()
        row2.addWidget(self.create_button("5. KI-Trainer / Modell-Training", self.open_yolo_trainer,
            "Trainiert ein YOLO-Modell basierend auf den annotierten Bildern.", QStyle.StandardPixmap.SP_ArrowRight))
        row2.addWidget(self.create_button("Training-Dashboard", self.open_dashboard,
            "Zeigt Trainingsfortschritt und Modellmetriken.", QStyle.StandardPixmap.SP_ComputerIcon))
        layout.addLayout(row2)


        # Verifikation & Hyperparameter-Tuning nebeneinander
        row3 = QHBoxLayout()
        row3.addWidget(self.create_button("6. Verifikation mit Test-Bildern", self.open_verification,
            "Überprüft das trainierte Modell mit einem separaten Test-Datensatz.", QStyle.StandardPixmap.SP_FileDialogContentsView))
        row3.addWidget(self.create_button("Hyperparamteter tuning", self.open_verification,
            "Berechnet für jede Objektklasse und IoU die optimalen Schwellenwerte (dauert lange!).", QStyle.StandardPixmap.SP_FileDialogContentsView))        
        layout.addLayout(row3)

        # Live Detection
        layout.addWidget(self.create_button("7. Live Objekterkennung / Kamerastream", self.open_detection,
            "Erkennt Objekte in Echtzeit aus einem Kamerastream.", QStyle.StandardPixmap.SP_MediaPlay))

        
        layout.addStretch()
        self.windows = {}

        # Footer ganz unten am Fenster zentriert
        # Text: "Application by Michel Marty for flex precision plastic solutions AG switzerland (copyright symbol) 2025"
        footer = QLabel("Application by Michel Marty for Flex Precision Plastic Solutions AG Switzerland © 2025")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(footer)

    def create_button(self, text, callback, tooltip, icon, button_font=QFont("Arial", 14)):
        btn = QPushButton(text)
        btn.setFont(button_font)
        button_style = """
            QPushButton {
                background-color: #165a69;
                color: white;
                padding: 20px;
                border-radius: 8px;
                font-weight: bold;
                min-width: 300px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #7ABF5A;
            }
        """
        btn.setStyleSheet(button_style)
        btn.setToolTip(tooltip)
        btn.setIcon(self.style().standardIcon(icon))  # StandardPixmap korrekt laden
        btn.setIconSize(QSize(32, 32))
        btn.clicked.connect(callback)
        return btn    

    
    def open_camera(self):
        if 'camera' not in self.windows:
            self.windows['camera'] = CameraApp()
        self.windows['camera'].show()
    
    def open_labeling(self):
        if 'labeling' not in self.windows:
            self.windows['labeling'] = ImageLabelingApp()
        self.windows['labeling'].show()
    
    def open_augmentation(self):
        if 'augmentation' not in self.windows:
            self.windows['augmentation'] = ImageAugmentationApp()
        self.windows['augmentation'].show()
    
    def open_dataset_viewer(self):
        if 'dataset_viewer' not in self.windows:
            self.windows['dataset_viewer'] = DatasetViewerApp()
        self.windows['dataset_viewer'].show()
    
    def open_splitter(self):
        if 'splitter' not in self.windows:
            self.windows['splitter'] = DatasetSplitterApp()
        self.windows['splitter'].show()
    
    def open_yolo_trainer(self):
        if 'yolo_trainer' not in self.windows:
            self.windows['yolo_trainer'] = TrainSettingsWindow()
        self.windows['yolo_trainer'].show()
    
    def open_dashboard(self):
        if 'dashboard' not in self.windows:
            self.windows['dashboard'] = DashboardWindow()
        self.windows['dashboard'].show()
    
    def open_verification(self):
        if 'verification' not in self.windows:
            self.windows['verification'] = LiveAnnotationApp()
        self.windows['verification'].show()

    """ 
    # Diese App ist noch nicht implementiert    
    def open_verification(self):
        if 'verification' not in self.windows:
            self.windows['verification'] = LiveAnnotationApp()
        self.windows['verification'].show()
    """   
    
    def open_detection(self):
        if 'detection' not in self.windows:
            self.windows['detection'] = LiveDetectionApp()
        self.windows['detection'].show()
