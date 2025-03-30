"""UI creation functions for the training settings panel."""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, 
    QScrollArea, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from config import Config
from gui.training.parameter_info import ParameterInfoButton

def create_settings_ui(window):
    """Create and return the settings UI components."""
    # Scroll area for settings
    settings_scroll = QScrollArea()
    settings_scroll.setWidgetResizable(True)
    settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    settings_scroll.setStyleSheet("""
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QScrollBar:vertical {
            border: none;
            background: #f0f2f5;
            width: 8px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background: #cbd5e0;
            min-height: 30px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical:hover {
            background: #a0aec0;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)
    
    settings_content = QWidget()
    settings_content_layout = QVBoxLayout(settings_content)
    settings_content_layout.setSpacing(10)
    settings_scroll.setWidget(settings_content)
    
    # Add title to settings panel - REDUCED SIZE
    settings_title = QLabel("Training Configuration")
    settings_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
    settings_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    settings_title.setMaximumHeight(30)  # Reduced height
    settings_title.setStyleSheet("margin-bottom: 5px; color: #2d3748;")
    settings_content_layout.addWidget(settings_title)
    
    # Create the basic settings
    basic_group = create_basic_settings(window)
    settings_content_layout.addWidget(basic_group)
    
    # Create advanced settings
    advanced_group = create_advanced_settings(window)
    settings_content_layout.addWidget(advanced_group)
    
    # Add scroll area to settings panel
    window.settings_layout.addWidget(settings_scroll)
    
    return settings_content

def create_basic_settings(window):
    """Create basic settings UI group."""
    # Project and experiment section
    group_frame = QGroupBox("Project Settings")
    group_layout = QGridLayout(group_frame)
    group_layout.setVerticalSpacing(6)  # Reduced spacing
    
    row = 0
    # Project directory
    group_layout.addWidget(QLabel("Projekt-Verzeichnis:"), row, 0)
    window.project_input = QLineEdit()
    window.project_input.setPlaceholderText("z.B. yolo_training_results")
    window.project_input.setText(Config.training.project_dir)
    group_layout.addWidget(window.project_input, row, 1)
    
    window.project_browse = QPushButton("...")
    window.project_browse.setMaximumWidth(40)
    window.project_browse.clicked.connect(lambda: browse_project(window))
    group_layout.addWidget(window.project_browse, row, 2)
    
    info_button = ParameterInfoButton(
        "Das Projekt-Verzeichnis bestimmt, wo die Trainingsergebnisse gespeichert werden.\n\n"
        "Dies ist der übergeordnete Ordner, in dem für jedes Experiment ein separater "
        "Unterordner angelegt wird."
    )
    group_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Experiment name
    group_layout.addWidget(QLabel("Experiment-Name:"), row, 0)
    window.name_input = QLineEdit()
    window.name_input.setPlaceholderText("z.B. experiment_v1")
    group_layout.addWidget(window.name_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Der Experiment-Name identifiziert das aktuelle Training und bestimmt "
        "den Namen des Unterordners im Projekt-Verzeichnis.\n\n"
        "Verwenden Sie aussagekräftige Namen, um verschiedene Trainingsläufe "
        "zu unterscheiden (z.B. 'small_model_v1', 'large_dataset_test')."
    )
    group_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Data path
    group_layout.addWidget(QLabel("Datenpfad (YAML):"), row, 0)
    window.data_input = QLineEdit()
    group_layout.addWidget(window.data_input, row, 1)
    
    window.data_browse = QPushButton("...")
    window.data_browse.setMaximumWidth(40)
    window.data_browse.clicked.connect(lambda: browse_data(window))
    group_layout.addWidget(window.data_browse, row, 2)
    
    info_button = ParameterInfoButton(
        "SUCHE IM SPLITTED-VERZEICHNIS, das erstellt wurde, beim Teilen des Datensatzes.\n\n"
        "Die YAML-Datei beschreibt den Datensatz mit:\n"
        "- Pfaden zu Trainings- und Validierungsdaten\n"
        "- Klassennamen und IDs\n"
        "- Anzahl der Klassen\n\n"
        "Format-Beispiel:\n"
        "```\n"
        "path: ../datasets/coco128\n"
        "train: images/train2017\n"
        "val: images/train2017\n"
        "names:\n"
        "  0: person\n"
        "  1: bicycle\n"
        "  ...\n"
        "```"
    )
    group_layout.addWidget(info_button, row, 3)
    
    # Basic training parameters
    training_group = QGroupBox("Basic Training Parameters")
    training_layout = QGridLayout(training_group)
    training_layout.setVerticalSpacing(6)  # Reduced spacing
    
    row = 0
    # Epochs
    training_layout.addWidget(QLabel("Anzahl Epochen:"), row, 0)
    window.epochs_input = QSpinBox()
    window.epochs_input.setRange(5, 500)
    window.epochs_input.setValue(Config.training.epochs)
    training_layout.addWidget(window.epochs_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Anzahl der vollständigen Durchläufe durch den Trainingsdatensatz.\n\n"
        "Empfehlungen:\n"
        "- Kleinere Datensätze (<1000 Bilder): 100-300 Epochen\n"
        "- Mittlere Datensätze (1000-10000 Bilder): 50-100 Epochen\n"
        "- Große Datensätze (>10000 Bilder): 30-50 Epochen\n\n"
        "Zu viele Epochen können zu Überanpassung führen, bei der das Modell die "
        "Trainingsdaten auswendig lernt, statt zu generalisieren."
    )
    training_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Image size
    training_layout.addWidget(QLabel("Bildgröße:"), row, 0)
    window.imgsz_input = QSpinBox()
    window.imgsz_input.setRange(640, 1280)
    window.imgsz_input.setValue(Config.training.image_size)
    window.imgsz_input.setSingleStep(32)
    training_layout.addWidget(window.imgsz_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Die Größe, auf die alle Trainingsbilder skaliert werden (in Pixeln).\n\n"
        "Empfehlungen:\n"
        "- 640: Standardwert, gute Balance aus Geschwindigkeit und Genauigkeit\n"
        "- 832: Bessere Erkennung kleiner Fehler in Spritzgussteilen\n"
        "- 1024-1280: Höchste Genauigkeit, aber langsamer und benötigt mehr GPU-Speicher\n\n"
        "Wichtig: Die Bildgröße sollte durch 32 teilbar sein (z.B. 640, 672, 704, usw.)."
    )
    training_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Batch
    training_layout.addWidget(QLabel("Batch:"), row, 0)
    window.batch_input = QDoubleSpinBox()
    window.batch_input.setRange(0.0, 1.0)
    window.batch_input.setDecimals(2)
    window.batch_input.setValue(Config.training.batch)
    window.batch_input.setSingleStep(0.05)
    training_layout.addWidget(window.batch_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Die Batch-Größe bestimmt, wie viele Bilder gleichzeitig verarbeitet werden.\n\n"
        "In YOLOv8 wird die Batch-Größe automatisch bestimmt, basierend auf dem verfügbaren "
        "GPU-Speicher. Der Wert zwischen 0 und 1 gibt an, welcher Anteil des maximal möglichen "
        "Batch verwendet werden soll.\n\n"
        "Empfehlungen:\n"
        "- 0.5: Conservative (stabiler, aber langsamer)\n"
        "- 0.7: Balanced (gute Balance)\n"
        "- 0.9: Aggressive (schneller, aber kann zu Out-of-Memory führen)\n\n"
        "Bei Out-of-Memory-Fehlern reduzieren Sie diesen Wert."
    )
    training_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Learning rate
    training_layout.addWidget(QLabel("Lernrate (lr0):"), row, 0)
    window.lr_input = QDoubleSpinBox()
    window.lr_input.setRange(0.0001, 0.1)
    window.lr_input.setDecimals(4)
    window.lr_input.setValue(Config.training.lr0)
    window.lr_input.setSingleStep(0.0005)
    training_layout.addWidget(window.lr_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Die anfängliche Lernrate bestimmt, wie stark die Modellgewichte bei jedem "
        "Trainingsschritt angepasst werden.\n\n"
        "Empfehlungen:\n"
        "- 0.001-0.01: Typischer Bereich für AdamW Optimizer\n"
        "- 0.01: Schnelleres Training, kann aber zu Instabilität führen\n"
        "- 0.001: Stabileres Training, aber langsamer\n\n"
        "Eine zu hohe Lernrate kann dazu führen, dass das Training divergiert und die "
        "Performance abnimmt. Eine zu niedrige Lernrate kann das Training sehr langsam machen."
    )
    training_layout.addWidget(info_button, row, 3)
    
    # Create a container widget for both groups
    container = QWidget()
    container_layout = QVBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setSpacing(8)  # Reduced spacing between group boxes
    container_layout.addWidget(group_frame)
    container_layout.addWidget(training_group)
    
    return container

def create_advanced_settings(window):
    """Create advanced settings UI group."""
    # Advanced parameters
    advanced_frame = QGroupBox("Advanced Settings")
    advanced_layout = QGridLayout(advanced_frame)
    advanced_layout.setVerticalSpacing(6)  # Reduced spacing
    
    row = 0
    # Resume training
    advanced_layout.addWidget(QLabel("Training fortsetzen:"), row, 0)
    window.resume_input = QCheckBox()
    # Style the checkbox to be more visible on light background
    window.resume_input.setStyleSheet("""
        QCheckBox {
            color: #333333;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border: 2px solid #1976D2;
            border-radius: 4px;
            background-color: white;
        }
        QCheckBox::indicator:checked {
            background-color: #1976D2;
            border-color: #1976D2;
            image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0iZmVhdGhlciBmZWF0aGVyLWNoZWNrIj48cG9seWxpbmUgcG9pbnRzPSIyMCA2IDkgMTcgNCAxMiI+PC9wb2x5bGluZT48L3N2Zz4=);
        }
    """)
    window.resume_input.setChecked(Config.training.resume)
    advanced_layout.addWidget(window.resume_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Wenn aktiviert, wird das Training vom letzten gespeicherten Checkpoint fortgesetzt.\n\n"
        "Nützlich wenn:\n"
        "- Ein Training unterbrochen wurde und fortgesetzt werden soll\n"
        "- Zusätzliche Epochen nach einem abgeschlossenen Training gewünscht sind\n\n"
        "Das Modell wird vom letzten Checkpoint (last.pt) geladen und das Training wird "
        "von dort aus fortgesetzt."
    )
    advanced_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Multi-scale training
    advanced_layout.addWidget(QLabel("Multi-Scale Training:"), row, 0)
    window.multi_scale_input = QCheckBox()
    # Apply same checkbox style
    window.multi_scale_input.setStyleSheet(window.resume_input.styleSheet())
    window.multi_scale_input.setChecked(Config.training.multi_scale)
    advanced_layout.addWidget(window.multi_scale_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Wenn aktiviert, werden während des Trainings Bilder mit verschiedenen Größen verwendet.\n\n"
        "Vorteile:\n"
        "- Verbessert die Modellgeneralisierung auf verschiedene Bildgrößen\n"
        "- Erhöht die Robustheit gegenüber verschiedenen Fehlerpositionen und -größen\n\n"
        "Nachteile:\n"
        "- Langsamer als Training mit fester Bildgröße\n"
        "- Kann mehr GPU-Speicher benötigen\n\n"
        "Besonders nützlich für die Inspektion von Spritzgussteilen, wo Fehler in verschiedenen Größen auftreten können."
    )
    advanced_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Cosine LR scheduling
    advanced_layout.addWidget(QLabel("Cosine Learning Rate:"), row, 0)
    window.cos_lr_input = QCheckBox()
    # Apply same checkbox style
    window.cos_lr_input.setStyleSheet(window.resume_input.styleSheet())
    window.cos_lr_input.setChecked(Config.training.cos_lr)
    advanced_layout.addWidget(window.cos_lr_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Wenn aktiviert, wird die Lernrate nach einem Cosinus-Schema reduziert.\n\n"
        "Cosine LR Scheduling:\n"
        "- Beginnt mit der angegebenen Lernrate\n"
        "- Reduziert die Lernrate langsamer zu Beginn und schneller gegen Ende\n"
        "- Folgt einer Cosinus-Kurve, endet nahe Null\n\n"
        "Vorteile:\n"
        "- Oft bessere Konvergenz als lineare Reduktion\n"
        "- Hilft, ein optimales Minimum zu finden\n\n"
        "In den meisten Fällen empfohlen, besonders für die präzise Erkennung von Fehlern in Spritzgussteilen."
    )
    advanced_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Close mosaic
    advanced_layout.addWidget(QLabel("Close Mosaic Epochs:"), row, 0)
    window.close_mosaic_input = QSpinBox()
    window.close_mosaic_input.setRange(0, 15)
    window.close_mosaic_input.setValue(Config.training.close_mosaic)
    advanced_layout.addWidget(window.close_mosaic_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Anzahl der letzten Epochen, in denen Mosaic-Augmentation deaktiviert wird.\n\n"
        "Mosaic-Augmentation kombiniert 4 Trainingsbilder zu einem neuen Bild, was die "
        "Erkennung von Objekten in unterschiedlichen Kontexten verbessert.\n\n"
        "Empfehlungen:\n"
        "- 0: Mosaic wird während des gesamten Trainings verwendet\n"
        "- 5-10: In den letzten Epochen wird das Modell mit nicht-transformierten Bildern feinabgestimmt\n\n"
        "Das Deaktivieren von Mosaic in den letzten Epochen kann die Genauigkeit verbessern, "
        "da das Modell mit realistischen, nicht-transformierten Bildern feinabgestimmt wird."
    )
    advanced_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Momentum
    advanced_layout.addWidget(QLabel("Momentum:"), row, 0)
    window.momentum_input = QDoubleSpinBox()
    window.momentum_input.setRange(0.8, 0.999)
    window.momentum_input.setDecimals(3)
    window.momentum_input.setValue(Config.training.momentum)
    window.momentum_input.setSingleStep(0.01)
    advanced_layout.addWidget(window.momentum_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Momentum ist ein Parameter, der dem Optimizer hilft, lokale Minima zu überwinden.\n\n"
        "Bei AdamW (Standardoptimizer) entspricht dieser Wert dem Beta1-Parameter, der "
        "bestimmt, wie stark frühere Gradienten die aktuelle Richtung beeinflussen.\n\n"
        "Empfehlungen:\n"
        "- 0.9: Standardwert, funktioniert gut für die meisten Fälle\n"
        "- 0.95: Kann bei rauschigen Gradienten helfen, ist aber weniger reaktionsschnell\n"
        "- 0.8: Reagiert schneller auf neue Gradienten, kann aber weniger stabil sein\n\n"
        "Bei der Erkennung von Fehlern in Spritzgussteilen kann ein höheres Momentum (0.95) "
        "hilfreich sein, wenn die Fehlerbilder variabel oder verrauscht sind."
    )
    advanced_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Warmup epochs
    advanced_layout.addWidget(QLabel("Warmup Epochs:"), row, 0)
    window.warmup_epochs_input = QSpinBox()
    window.warmup_epochs_input.setRange(0, 10)
    window.warmup_epochs_input.setValue(Config.training.warmup_epochs)
    advanced_layout.addWidget(window.warmup_epochs_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Anzahl der Epochen, in denen die Lernrate langsam auf den vollen Wert erhöht wird.\n\n"
        "Während der Warmup-Phase wird die Lernrate graduell von einem kleinen Wert auf den "
        "konfigurieren lr0-Wert erhöht.\n\n"
        "Vorteile:\n"
        "- Stabilisiert frühe Trainingsphase\n"
        "- Verhindert divergierende Gewichtsaktualisierungen zu Beginn\n\n"
        "Empfehlungen:\n"
        "- 0: Kein Warmup (volle Lernrate von Anfang an)\n"
        "- 3: Typischer Wert für die meisten Trainings\n"
        "- 5-10: Für komplizierte Datensätze oder instabiles Training"
    )
    advanced_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Warmup momentum
    advanced_layout.addWidget(QLabel("Warmup Momentum:"), row, 0)
    window.warmup_momentum_input = QDoubleSpinBox()
    window.warmup_momentum_input.setRange(0.0, 1.0)
    window.warmup_momentum_input.setDecimals(2)
    window.warmup_momentum_input.setValue(Config.training.warmup_momentum)
    advanced_layout.addWidget(window.warmup_momentum_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Anfangswert für Momentum während der Warmup-Phase.\n\n"
        "Ähnlich zur Lernrate wird auch das Momentum während des Warmups graduell "
        "von diesem Wert auf den vollen Momentum-Wert erhöht.\n\n"
        "Empfehlungen:\n"
        "- 0.8: Typischer Wert, bietet gute Balance\n"
        "- 0.5: Niedrigerer Wert für sanfteren Start\n"
        "- 0.9: Höherer Wert, wenn weniger Warmup-Effekt gewünscht ist\n\n"
        "In den meisten Fällen kann der Standardwert beibehalten werden."
    )
    advanced_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Box loss gain
    advanced_layout.addWidget(QLabel("Box Loss Gain:"), row, 0)
    window.box_input = QSpinBox()
    window.box_input.setRange(3, 10)
    window.box_input.setValue(Config.training.box)
    advanced_layout.addWidget(window.box_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Gewichtungsfaktor für den Bounding-Box-Lokalisierungs-Loss.\n\n"
        "Dieser Parameter bestimmt, wie stark Fehler in der Positionierung der Bounding Boxes "
        "im Vergleich zu anderen Loss-Komponenten gewichtet werden.\n\n"
        "Empfehlungen:\n"
        "- 7.5: Standardwert, gute Balance\n"
        "- 5: Weniger Gewicht auf Lokalisierungsgenauigkeit (wenn die genaue Position weniger wichtig ist)\n"
        "- 10: Mehr Gewicht auf Lokalisierungsgenauigkeit (für Anwendungen, die präzise Boxen erfordern)\n\n"
        "Für die Fehlerinspektion in Spritzgussteilen oft sinnvoll, einen höheren Wert (8-10) zu wählen, "
        "um die genaue Abgrenzung von Defekten zu verbessern."
    )
    advanced_layout.addWidget(info_button, row, 3)
    
    row += 1
    # Dropout
    advanced_layout.addWidget(QLabel("Dropout:"), row, 0)
    window.dropout_input = QDoubleSpinBox()
    window.dropout_input.setRange(0.0, 0.5)
    window.dropout_input.setDecimals(2)
    window.dropout_input.setValue(Config.training.dropout)
    window.dropout_input.setSingleStep(0.05)
    advanced_layout.addWidget(window.dropout_input, row, 1)
    
    info_button = ParameterInfoButton(
        "Dropout-Rate zur Verhinderung von Überanpassung während des Trainings.\n\n"
        "Dropout deaktiviert zufällig einen Teil der Neuronen während des Trainings, "
        "was dem Modell hilft, robustere Features zu lernen und nicht von einzelnen "
        "Neuronen abhängig zu werden.\n\n"
        "Empfehlungen:\n"
        "- 0.0: Kein Dropout (für kleine Datensätze oder wenn Überanpassung kein Problem ist)\n"
        "- 0.1: Leichter Dropout-Effekt (Standardwert für die meisten Fälle)\n"
        "- 0.3-0.5: Stärkerer Dropout für Datensätze mit Überanpassungsrisiko\n\n"
        "In der industriellen Fehlerinspektion kann ein moderater Dropout-Wert (0.1-0.2) helfen, "
        "die Robustheit gegenüber verschiedenen Fehlerdarstellungen zu verbessern."
    )
    advanced_layout.addWidget(info_button, row, 3)
    
    return advanced_frame

def browse_project(window):
    """Open file dialog to select project directory."""
    directory = QFileDialog.getExistingDirectory(
        window, "Projektverzeichnis wählen", 
        window.project_input.text()
    )
    if directory:
        window.project_input.setText(directory)

def browse_data(window):
    """Open file dialog to select YAML dataset file."""
    file_path, _ = QFileDialog.getOpenFileName(
        window, "Datenpfad auswählen", "", "YAML Dateien (*.yaml)"
    )
    if file_path:
        window.data_input.setText(file_path)