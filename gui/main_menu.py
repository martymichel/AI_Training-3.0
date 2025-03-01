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
from gui.dataset_splitter import DatasetSplitterApp
from gui.image_labeling import ImageLabelingApp
from gui.camera_app import CameraApp
from gui.live_detection import LiveDetectionApp
class MainMenu(QMainWindow):
    """Main menu window providing access to different applications."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Vision Tools")
        self.setGeometry(100, 100, 800, 500)
        
        # Zentral-Widget und Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Überschrift
        title = QLabel("AI Vision Tools")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Unterüberschrift
        subtitle = QLabel(
            "by Michel Marty"
        )
        subtitle_font = QFont()
        subtitle_font.setPointSize(16)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        # Beschreibung
        description = QLabel(
            "Willkommen bei den AI Vision Tools. "
            "Wähle eine Anwendung aus:"
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

        # Camera App Button
        self.camera_button = QPushButton("1. Kamera Livestream / Bilder aufnehmen")
        self.camera_button.setMinimumHeight(60)
        self.camera_button.clicked.connect(self.open_camera)
        self.camera_button.setFont(button_font)
        layout.addWidget(self.camera_button)        

        # Image Labeling Button
        self.labeling_button = QPushButton("2. Bilder Labeln / Bounding Boxen markieren")
        self.labeling_button.setMinimumHeight(60)
        self.labeling_button.clicked.connect(self.open_labeling)
        self.labeling_button.setFont(button_font)
        layout.addWidget(self.labeling_button)

        # Image Augmentation Button
        self.augmentation_button = QPushButton("3. Bilder vervielfältigen (Augmentierung)")
        self.augmentation_button.setMinimumHeight(60)
        self.augmentation_button.clicked.connect(self.open_augmentation)
        self.augmentation_button.setFont(button_font)
        layout.addWidget(self.augmentation_button)
        
        # Dataset Viewer Button
        self.dataset_viewer_button = QPushButton("4. Bounding Boxen prüfen / Dataset Viewer")
        self.dataset_viewer_button.setMinimumHeight(60)
        self.dataset_viewer_button.clicked.connect(self.open_dataset_viewer)
        self.dataset_viewer_button.setFont(button_font)
        layout.addWidget(self.dataset_viewer_button)        
        
        # Dataset Splitter Button
        self.splitter_button = QPushButton("5. Dataset Splitter (Train/Validation/Test)")
        self.splitter_button.setMinimumHeight(60)
        self.splitter_button.clicked.connect(self.open_splitter)
        self.splitter_button.setFont(button_font)
        layout.addWidget(self.splitter_button)

        # YOLO Trainer Button
        self.yolo_button = QPushButton("6. YOLO Trainer / Modell-Training")
        self.yolo_button.setMinimumHeight(60)
        self.yolo_button.clicked.connect(self.open_yolo_trainer)
        self.yolo_button.setFont(button_font)
        layout.addWidget(self.yolo_button)

        # Modell-Verifikation Button
        self.verification_button = QPushButton("7. Modell-Verifikation / Annotation mit Test-Dataset")
        self.verification_button.setMinimumHeight(60)
        self.verification_button.clicked.connect(self.open_verification)
        self.verification_button.setFont(button_font)        
        layout.addWidget(self.verification_button)

        # Live Detection Button
        self.detection_button = QPushButton("8. Live Objekterkennung / Annotation mit Kamerastream")
        self.detection_button.setMinimumHeight(60)
        self.detection_button.clicked.connect(self.open_detection)
        self.detection_button.setFont(button_font)
        layout.addWidget(self.detection_button)
        
        # Platzhalter für weitere Apps
        layout.addStretch()
        
        # Speichern der Fenster-Referenzen
        self.windows = {}

    def open_camera(self):
        """Open the camera application window."""
        if 'camera' not in self.windows:
            self.windows['camera'] = CameraApp()
        self.windows['camera'].show()
        self.windows['camera'].activateWindow()
    
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

    def open_splitter(self):
        """Open the dataset splitter window."""
        if 'splitter' not in self.windows:
            self.windows['splitter'] = DatasetSplitterApp()
        self.windows['splitter'].show()
        self.windows['splitter'].activateWindow()

    def open_labeling(self):
        """Open the image labeling window."""
        if 'labeling' not in self.windows:
            from gui.image_labeling import ImageLabelingApp
            self.windows['labeling'] = ImageLabelingApp()
        self.windows['labeling'].show()
        self.windows['labeling'].activateWindow()

    def open_detection(self):
        """Open the live detection application window."""
        if 'detection' not in self.windows:
            self.windows['detection'] = LiveDetectionApp()
        self.windows['detection'].show()
        self.windows['detection'].activateWindow()        