"""Main menu window for launching different applications."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from gui.gui_settings import TrainSettingsWindow
from gui.augmentation_app import ImageAugmentationApp
from gui.dataset_viewer import DatasetViewerApp
from gui.verification_app import LiveAnnotationApp

class MainMenu(QMainWindow):
    """Main menu window providing access to different applications."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KI Vision Tools")
        self.setGeometry(100, 100, 800, 500)
        
        # Zentral-Widget und Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Überschrift
        title = QLabel("KI Vision Tools")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Beschreibung
        description = QLabel(
            "Willkommen bei den KI Vision Tools. "
            "Wählen Sie eine Anwendung aus:"
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_font = QFont()
        description_font.setPointSize(12)
        description.setFont(description_font)
        layout.addWidget(description)

        # Button-Schriftart
        button_font = QFont()
        button_font.setPointSize(12)

        # Image Augmentation Button
        self.augmentation_button = QPushButton("Bild Augmentierung")
        self.augmentation_button.setMinimumHeight(60)
        self.augmentation_button.clicked.connect(self.open_augmentation)
        self.augmentation_button.setFont(button_font)
        layout.addWidget(self.augmentation_button)
        
        # Dataset Viewer Button
        self.dataset_viewer_button = QPushButton("Dataset Viewer")
        self.dataset_viewer_button.setMinimumHeight(60)
        self.dataset_viewer_button.clicked.connect(self.open_dataset_viewer)
        self.dataset_viewer_button.setFont(button_font)
        layout.addWidget(self.dataset_viewer_button)        
        
        # YOLO Trainer Button
        self.yolo_button = QPushButton("YOLO Trainer")
        self.yolo_button.setMinimumHeight(60)
        self.yolo_button.clicked.connect(self.open_yolo_trainer)
        self.yolo_button.setFont(button_font)
        layout.addWidget(self.yolo_button)

        # Modell-Verifikation Button
        self.verification_button = QPushButton("Modell-Verifikation")
        self.verification_button.setMinimumHeight(60)
        self.verification_button.clicked.connect(self.open_verification)
        self.verification_button.setFont(button_font)        
        layout.addWidget(self.verification_button)
        
        # Platzhalter für weitere Apps
        layout.addStretch()
        
        # Speichern der Fenster-Referenzen
        self.windows = {}
    
    def open_yolo_trainer(self):
        """Open the YOLO trainer window."""
        if 'yolo_trainer' not in self.windows:
            self.windows['yolo_trainer'] = TrainSettingsWindow()
        self.windows['yolo_trainer'].show()
        self.windows['yolo_trainer'].activateWindow()

    def open_augmentation(self):
        """Open the image augmentation window."""
        if 'augmentation' not in self.windows:
            self.windows['augmentation'] = ImageAugmentationApp()
        self.windows['augmentation'].show()
        self.windows['augmentation'].activateWindow()

    def open_dataset_viewer(self):
        """Open the dataset viewer window."""
        if 'dataset_viewer' not in self.windows:
            self.windows['dataset_viewer'] = DatasetViewerApp()
        self.windows['dataset_viewer'].show()
        self.windows['dataset_viewer'].activateWindow()

    def open_verification(self):
        """Open the model verification window."""
        if 'verification' not in self.windows:
            self.windows['verification'] = LiveAnnotationApp()
        self.windows['verification'].show()
        self.windows['verification'].activateWindow()