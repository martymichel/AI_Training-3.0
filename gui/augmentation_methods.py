"""Method management for the augmentation application."""

import logging
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QSpinBox, QCheckBox, QMessageBox
)

logger = logging.getLogger(__name__)

def get_method_key(method_name):
    """Convert German method name to English key."""
    method_map = {
        "Verschiebung": "Shift",
        "Rotation": "Rotate",
        "Zoom": "Zoom",
        "Helligkeit": "Brightness",
        "Unschärfe": "Blur"
    }
    return method_map.get(method_name, method_name)

def create_method_controls(app):
    """Create controls for all augmentation methods."""
    method_group_style = """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 5px;
            padding: 8px;
            background: white;
        }
        QCheckBox {
            spacing: 3px;
            color: #333;
            padding: 5px;
            font-weight: bold;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 2px;
            border: 1px solid #ddd;
            background: white;
        }
        QCheckBox::indicator:checked {
            background: #2196F3;
            border-color: #2196F3;
        }
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 3px;
            padding: 5px;
            font-weight: bold;
            margin-left: 5px;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QSpinBox {
            padding: 3px;
            border: 1px solid #ddd;
            border-radius: 3px;
            background: white;
            min-width: 50px;
        }
        QSpinBox:hover {
            border-color: #2196F3;
        }
        /* Make the up/down arrows wider */
        QSpinBox::up-button, QSpinBox::down-button {
            width: 20px; /* Wider buttons */
        }
    """
    app.method_group.setStyleSheet(method_group_style)
    
    # Create method boxes
    for method in app.methods:
        # Create method container
        method_box = QGroupBox()
        method_layout = QVBoxLayout(method_box)
        method_layout.setContentsMargins(4, 4, 4, 4)
        method_layout.setSpacing(2)

        # Header layout with checkbox and info button
        header_layout = QHBoxLayout()
        checkbox = QCheckBox(method)
        checkbox.stateChanged.connect(app.update_expected_count)
        checkbox.stateChanged.connect(app.update_preview)
        app.method_checkboxes.append(checkbox)

        # Info button
        info_btn = QPushButton("ℹ️")
        info_btn.setFixedSize(24, 24)
        info_btn.clicked.connect(lambda checked, m=method: show_method_info(app, m))

        header_layout.addWidget(checkbox)
        header_layout.addWidget(info_btn)
        header_layout.addStretch()
        method_layout.addLayout(header_layout)

        # Level controls
        level_layout = QHBoxLayout()
        level_layout.setContentsMargins(5, 0, 0, 0)  # Indent levels
        level_layout.setSpacing(3)
        level_label1 = QLabel("Stufe 1:")
        level_label2 = QLabel("Stufe 2:")

        level_spinbox1 = QSpinBox()
        level_spinbox2 = QSpinBox()
        level_spinbox1.setRange(1, 100)
        level_spinbox2.setRange(1, 100)
        level_spinbox1.setValue(2)
        level_spinbox2.setValue(10)
        
        # Connect spinboxes to preview update
        level_spinbox1.valueChanged.connect(app.update_preview)
        level_spinbox2.valueChanged.connect(app.update_preview)

        level_layout.addWidget(level_label1)
        level_layout.addWidget(level_spinbox1)
        level_layout.addWidget(level_label2)
        level_layout.addWidget(level_spinbox2)
        level_layout.addStretch()
        method_layout.addLayout(level_layout)

        # Add container to main layout
        app.method_layout.addWidget(method_box)
        
        # Style the method box
        method_box.setStyleSheet("""
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 3px;
                background: white;
            }
            QGroupBox:hover {
                border-color: #2196F3;
            }
        """)

        # Store references for later use
        method_key = get_method_key(method)
        app.method_levels[method_key] = (checkbox, level_spinbox1, level_spinbox2)

    # Add flip controls in a separate group box
    flip_box = QGroupBox()
    flip_layout = QVBoxLayout(flip_box)
    flip_layout.setContentsMargins(4, 4, 4, 4)
    flip_layout.setSpacing(2)
    
    # Horizontal flip
    app.horizontal_flip = QCheckBox("Horizontal Spiegeln")
    app.horizontal_flip.stateChanged.connect(app.update_expected_count)
    app.horizontal_flip.stateChanged.connect(app.update_preview)
    flip_layout.addWidget(app.horizontal_flip)
    
    # Vertical flip
    app.vertical_flip = QCheckBox("Vertikal Spiegeln")
    app.vertical_flip.stateChanged.connect(app.update_expected_count)
    app.vertical_flip.stateChanged.connect(app.update_preview)
    flip_layout.addWidget(app.vertical_flip)
    
    # Style flip box
    flip_box.setStyleSheet("""
        QGroupBox {
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 3px;
            padding: 5px;
            background: white;
        }
        QGroupBox:hover {
            border-color: #2196F3;
        }
    """)
    
    app.method_layout.addWidget(flip_box)

def show_method_info(app, method):
    """Show detailed information about augmentation method parameters."""
    info = {
        "Verschiebung": """Verschiebung des Bildes in X- und Y-Richtung
            
            Level 1: Minimaler Verschiebungsfaktor
            Level 2: Maximaler Verschiebungsfaktor
            
            Beispiel:
            Level 1 = 10, Level 2 = 30 bedeutet:
            - Zufällige Verschiebung zwischen 10% und 30% der Bildgröße in beide Richtungen
            - Die Richtung (positiv/negativ) wird zufällig gewählt
            
            Implementierung:
            1. Berechne Verschiebungsfaktoren zwischen Level 1 und 2
            2. Wähle zufällig positive oder negative Richtung
            3. Wende Verschiebung auf das Bild an
        """,
        "Rotation": """Rotation des Bildes um seinen Mittelpunkt
            
            Level 1: Minimaler Rotationswinkel (% von 360°)
            Level 2: Maximaler Rotationswinkel (% von 360°)
            
            Beispiel:
            Level 1 = 10, Level 2 = 45 bedeutet:
            - Zufällige Rotation zwischen 36° und 162°
            - Berechnung: Level × 360° / 100
            
            Implementierung:
            1. Berechne Winkelbereich aus Levels (Level × 360° / 100)
            2. Wähle zufälligen Winkel aus diesem Bereich
            3. Rotiere Bild und Boxen um Mittelpunkt
        """,
        "Zoom": """Skalierung des Bildes (Vergrößerung)
            
            Level 1: Minimale Vergrößerung (%)
            Level 2: Maximale Vergrößerung (%)
            
            Beispiel:
            Level 1 = 10, Level 2 = 50 bedeutet:
            - Zufällige Vergrößerung zwischen 110% und 150%
            - Berechnung: 1.0 + (Level/100)
            
            Implementierung:
            1. Berechne Skalierungsfaktoren (1.0 + Level/100)
            2. Wähle zufälligen Faktor zwischen Level 1 und 2
            3. Skaliere Bild und passe Boxen an
        """,
        "Helligkeit": """Anpassung der Bildhelligkeit
            
            Level 1: Minimale Helligkeitsänderung (%)
            Level 2: Maximale Helligkeitsänderung (%)
            
            Beispiel:
            Level 1 = 10, Level 2 = 30 bedeutet:
            - Zufällige Helligkeitsänderung zwischen 10% und 30%
            - Die Richtung (heller/dunkler) wird zufällig gewählt
            
            Implementierung:
            1. Wähle zufälligen Faktor zwischen Level 1 und 2
            2. Wähle zufällig positive oder negative Änderung
            3. Passe Bildhelligkeit entsprechend an
        """,
        "Unschärfe": """Gaussian Blur (Weichzeichnen) des Bildes
            
            Level 1: Minimale Unschärfe (%)
            Level 2: Maximale Unschärfe (%)
            
            Beispiel:
            Level 1 = 10, Level 2 = 30 bedeutet:
            - Zufällige Unschärfe mit Radius zwischen 0.1 und 0.3 Pixel
            - Berechnung: Level/100 Pixel Radius
            
            Implementierung:
            1. Berechne Blur-Radius aus Level (Level/100)
            2. Wende Gaussian Blur mit diesem Radius an
        """
    }
    
    # Show info in a styled message box
    msg = QMessageBox()
    msg.setWindowTitle(f"Info: {method}")
    msg.setText(info.get(method, "Keine Informationen verfügbar."))
    msg.setStyleSheet("""
        QMessageBox {
            background-color: white;
        }
        QMessageBox QLabel {
            color: #333;
            font-size: 14px;
            min-width: 700px;
        }
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
    """)
    msg.exec()