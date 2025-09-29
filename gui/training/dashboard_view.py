"""Dashboard visualization for the training window."""

import matplotlib
matplotlib.use('QtAgg')  # Use Qt backend

# Configure matplotlib logging - SUPPRESS DEBUG MESSAGES
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Set to WARNING to suppress DEBUG messages

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
import logging

from PyQt6.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QScrollArea,
    QLabel,
    QWidget,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QTextEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


# Import metrics info
from gui.training.metrics_info import create_metrics_info_tab

# Configure logging
logger = logging.getLogger("dashboard")
logger.setLevel(logging.INFO)

# Detailed metric explanations
METRIC_INFO = {
    "Box Loss": """<b>Box Loss</b> misst, wie genau das Modell die Fehlerposition in Spritzgussteilen bestimmt.

<b>Für Industrielle Anwendungen:</b>
- <b>Sehr gut:</b> < 0.05 – Extrem präzise Lokalisierung von Defekten
- <b>Gut:</b> 0.05 - 0.1 – Geringe Abweichungen bei der Defektlokalisierung
- <b>Akzeptabel:</b> 0.1 - 0.3 – Leichte Ungenauigkeiten in der Positionierung
- <b>Verbesserungsbedürftig:</b> > 0.3 – Die Defektlokalisierung ist unzureichend

<b>Praktische Bedeutung:</b>
Niedriger Box Loss bedeutet, dass Fehler wie Grate, Lunker oder Risse in Spritzgussteilen präzise lokalisiert werden, was für die automatisierte Qualitätssicherung entscheidend ist.""",

    "Class Loss": """<b>Class Loss</b> bewertet, wie gut das Modell zwischen verschiedenen Fehlerklassen in Spritzgussteilen unterscheiden kann.

<b>Für Industrielle Anwendungen:</b>
- <b>Sehr gut:</b> < 0.02 – Fast perfekte Unterscheidung zwischen Fehlertypen
- <b>Gut:</b> 0.02 - 0.05 – Hohe Genauigkeit bei der Fehlerklassifikation
- <b>Akzeptabel:</b> 0.05 - 0.15 – Einige Verwechslungen zwischen ähnlichen Fehlerarten
- <b>Verbesserungsbedürftig:</b> > 0.15 – Häufige Verwechslungen zwischen Fehlertypen

<b>Praktische Bedeutung:</b>
Niedriger Class Loss bedeutet, dass das Modell zuverlässig zwischen verschiedenen Defektarten (z.B. Risse, Lunker, Grate, Farbfehler) unterscheiden kann, was für gezielte Prozessoptimierung wichtig ist.""",

    "Precision & Recall": """<b>Precision & Recall</b> zeigen die Balance zwischen Zuverlässigkeit und Vollständigkeit der Fehlererkennung.

<b>Precision:</b> Anteil der korrekt erkannten Fehler unter allen erkannten "Fehlern"
<b>Recall:</b> Anteil der erkannten Fehler im Verhältnis zu allen tatsächlichen Fehlern

<b>Für Industrielle Anwendungen:</b>
- <b>Sehr gut:</b> > 95% – Kaum Fehlalarme, fast alle Defekte werden erkannt
- <b>Gut:</b> 90% - 95% – Wenige Fehlalarme, die meisten Defekte werden erkannt
- <b>Akzeptabel:</b> 80% - 90% – Moderate Fehlalarmrate, einige Defekte können übersehen werden
- <b>Verbesserungsbedürftig:</b> < 80% – Zu viele Fehlalarme oder übersehene Defekte

<b>Praktische Bedeutung:</b>
Hohe Precision reduziert Ausschuss von fälschlicherweise aussortiertem Material, hoher Recall gewährleistet, dass fehlerhafte Teile nicht an Kunden gelangen.""",

    "mAP Scores": """<b>mAP Scores</b> (Mean Average Precision) bewerten die Gesamtleistung des Modells bei der Fehlererkennung über alle Klassen.

<b>mAP50:</b> Erkennung bei 50% Überlappung
<b>mAP50-95:</b> Durchschnitt über verschiedene Überlappungsgrade (50%-95%)

<b>Für Industrielle Anwendungen:</b>
- <b>Sehr gut:</b> mAP50 > 95%, mAP50-95 > 80% – Zuverlässige und präzise Erkennung
- <b>Gut:</b> mAP50 90%-95%, mAP50-95 70%-80% – Gute Balance zwischen Erkennung und Präzision
- <b>Akzeptabel:</b> mAP50 80%-90%, mAP50-95 60%-70% – Brauchbar für einfachere Anwendungen
- <b>Verbesserungsbedürftig:</b> mAP50 < 80%, mAP50-95 < 60% – Nicht zuverlässig genug

<b>Praktische Bedeutung:</b>
Hohe mAP-Werte bedeuten, dass das Modell verschiedene Fehlertypen in Spritzgussteilen zuverlässig erkennt und präzise lokalisiert.""",

    "DFL Loss": """<b>DFL Loss</b> (Distribution Focal Loss) bewertet die Präzision der Bounding-Box-Grenzen.

<b>Für Industrielle Anwendungen:</b>
- <b>Sehr gut:</b> < 0.03 – Extrem präzise Grenzen um Fehlerregionen
- <b>Gut:</b> 0.03 - 0.08 – Genaue Abgrenzung von Defekten
- <b>Akzeptabel:</b> 0.08 - 0.15 – Leichte Ungenauigkeiten in der Abgrenzung
- <b>Verbesserungsbedürftig:</b> > 0.15 – Ungenaue Defektgrenzen

<b>Praktische Bedeutung:</b>
Niedriger DFL Loss ist besonders wichtig für die präzise Vermessung von Defekten in Spritzgussteilen und für die genaue Bestimmung, ob ein Fehler innerhalb der Toleranzgrenzen liegt.""",

    "Learning Rate": """<b>Learning Rate</b> ist die Schrittweite bei der Anpassung der Modellgewichte während des Trainings.

<b>Für Industrielle Anwendungen:</b>
- <b>Optimal:</b> 0.001 - 0.005 – Gute Balance aus Trainingsgeschwindigkeit und Stabilität
- <b>Zu hoch:</b> > 0.01 – Kann zu instabilem Training führen
- <b>Zu niedrig:</b> < 0.0005 – Training dauert unnötig lange

<b>Praktische Bedeutung:</b>
Die optimale Learning Rate verkürzt die Trainingszeit bis zur Produktionsreife des Modells und beeinflusst entscheidend die Endgenauigkeit bei der Fehlererkennung in Spritzgussteilen."""
}

def create_dashboard_tabs(window):
    """Create dashboard visualization tabs."""
    # Tab widget for dashboard and log
    tabs = QTabWidget()
    
    # Dashboard tab
    dashboard_tab = QWidget()
    dashboard_tab_layout = QVBoxLayout(dashboard_tab)
    
    # Create matplotlib figure for plots
    figure = plt.figure(figsize=(8, 10))
    canvas = FigureCanvas(figure)
    canvas.setMinimumHeight(500)
    
    # Make the canvas have a white background
    canvas.setStyleSheet("background-color: white;")
    
    dashboard_tab_layout.addWidget(canvas)
    
    # Initialize plots
    setup_plots(figure, canvas)
    
    # Log tab
    log_tab = QWidget()
    log_tab_layout = QVBoxLayout(log_tab)
    
    # Log text area - QTextEdit for full copy/paste functionality
    log_text = QTextEdit()
    log_text.setReadOnly(True)
    log_text.setPlainText("Training log will appear here...")
    log_text.setStyleSheet("""
        QTextEdit {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            color: #333333;
            font-size: 12px;
        }
    """)
    
    # Add instructions for copy/paste
    instructions = QLabel("💡 You can select and copy text from this log (Ctrl+A to select all, Ctrl+C to copy)")
    instructions.setStyleSheet("""
        QLabel {
            color: #666;
            font-size: 11px;
            padding: 5px;
            background-color: #e8f4f8;
            border-radius: 3px;
            border: 1px solid #bee5eb;
        }
    """)
    log_tab_layout.addWidget(instructions)
    
    log_tab_layout.addWidget(log_text)
    
    # Create metrics info tab
    metrics_tab = create_metrics_info_tab()
    
    # Add tabs to tab widget
    tabs.addTab(dashboard_tab, "Dashboard")
    tabs.addTab(log_tab, "Training Log")
    tabs.addTab(metrics_tab, "Metrics Info")
    
    # Connect canvas events for info popups
    canvas.mpl_connect('pick_event', lambda event: on_plot_info_click(event, window))
    
    return tabs, figure, canvas, log_text

def on_plot_info_click(event, window):
    """Display info when plot info icon is clicked."""
    if hasattr(event.artist, 'metric_key'):
        metric_key = event.artist.metric_key
        info_text = METRIC_INFO.get(metric_key, "No additional information available.")

        show_info_dialog(metric_key, info_text, window)

def show_info_dialog(metric_key: str, info_text: str, parent=None):
    """Show a scrollable dialog with metric information."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Information: {metric_key}")
    dialog.setMinimumSize(600, 400)

    layout = QVBoxLayout(dialog)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    content = QLabel()
    content.setObjectName("infoLabel")
    content.setTextFormat(Qt.TextFormat.RichText)
    content.setWordWrap(True)
    content.setText(info_text)
    scroll.setWidget(content)

    layout.addWidget(scroll)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
    buttons.accepted.connect(dialog.accept)
    layout.addWidget(buttons)

    dialog.setLayout(layout)
    dialog.exec()

def setup_plots(figure, canvas):
    """Initialize the matplotlib plots."""
    # Clear previous plots
    figure.clear()
    
    # Create subplots
    axes = figure.subplots(3, 2)
    
    # Configure subplots
    for i in range(3):
        for j in range(2):
            axes[i, j].grid(True)
            axes[i, j].set_xlabel("Epoch")
    
    # Set titles
    axes[0, 0].set_title("Box Loss")
    axes[0, 1].set_title("Class Loss")
    axes[1, 0].set_title("Precision & Recall")
    axes[1, 1].set_title("mAP Scores")
    axes[2, 0].set_title("DFL Loss")
    axes[2, 1].set_title("Learning Rate")
    
    # Set y-labels
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Score")
    axes[1, 1].set_ylabel("Score")
    axes[2, 0].set_ylabel("Loss")
    axes[2, 1].set_ylabel("LR")
    
    # Add info buttons to each subplot
    for i in range(3):
        for j in range(2):
            title = axes[i, j].get_title()
            info = axes[i, j].annotate("ℹ", xy=(0.95, 0.95), 
                                     xycoords="axes fraction",
                                     fontsize=14, 
                                     color="blue", 
                                     weight="bold",
                                     bbox=dict(boxstyle="round,pad=0.3", 
                                               fc="yellow", 
                                               ec="black", 
                                               lw=1),
                                     picker=True,
                                     zorder=100)
            info.metric_key = title
    
    # Adjust layout
    figure.tight_layout()
    canvas.draw()

def update_dashboard_plots(window, df):
    """Update the dashboard with new data."""
    try:
        # Make sure we have some data
        if len(df) < 1:
            return
            
        # Clear previous plots
        axes = window.figure.axes
        for ax in axes:
            ax.clear()
            ax.grid(True)
            
        # Extract epoch data
        epochs = df["epoch"]
        max_epoch = epochs.max()
        
        # Set up x-axis limits
        for ax in axes:
            ax.set_xlim(0, max(101, int(max_epoch) + 5))
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Box Loss plot
        ax = axes[0]
        if "train/box_loss" in df.columns:
            ax.plot(epochs, df["train/box_loss"], label="Train", color='blue')
            # Annotate minimum point
            min_idx = df["train/box_loss"].idxmin()
            min_epoch = df["epoch"].iloc[min_idx]
            min_value = df["train/box_loss"].min()
            ax.annotate(f'Best: {min_value:.4f}\nEpoch: {min_epoch}', 
                        xy=(min_epoch, min_value), 
                        xytext=(min_epoch + 2, min_value + 0.1),
                        arrowprops=dict(arrowstyle="->", color='blue'),
                        color='blue', fontsize=9)
        if "val/box_loss" in df.columns:
            ax.plot(epochs, df["val/box_loss"], label="Val", color='orange')
            # Annotate minimum point
            min_idx = df["val/box_loss"].idxmin()
            min_epoch = df["epoch"].iloc[min_idx]
            min_value = df["val/box_loss"].min()
            ax.annotate(f'Best: {min_value:.4f}\nEpoch: {min_epoch}', 
                        xy=(min_epoch, min_value), 
                        xytext=(min_epoch + 2, min_value + 0.05),
                        arrowprops=dict(arrowstyle="->", color='orange'),
                        color='orange', fontsize=9)
        ax.set_title("Box Loss")
        ax.set_ylabel("Loss")
        ax.legend()
        
        # Add info button
        info = ax.annotate("ℹ", xy=(0.95, 0.95), 
                         xycoords="axes fraction",
                         fontsize=14, 
                         color="blue", 
                         weight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  fc="yellow", 
                                  ec="black", 
                                  lw=1),
                         picker=True,
                         zorder=100)
        info.metric_key = "Box Loss"
        
        # Class Loss plot
        ax = axes[1]
        if "train/cls_loss" in df.columns:
            ax.plot(epochs, df["train/cls_loss"], label="Train", color='blue')
            # Annotate minimum point
            min_idx = df["train/cls_loss"].idxmin()
            min_epoch = df["epoch"].iloc[min_idx]
            min_value = df["train/cls_loss"].min()
            ax.annotate(f'Best: {min_value:.4f}\nEpoch: {min_epoch}', 
                        xy=(min_epoch, min_value), 
                        xytext=(min_epoch + 2, min_value + 0.05),
                        arrowprops=dict(arrowstyle="->", color='blue'),
                        color='blue', fontsize=9)
        if "val/cls_loss" in df.columns:
            ax.plot(epochs, df["val/cls_loss"], label="Val", color='orange')
            # Annotate minimum point
            min_idx = df["val/cls_loss"].idxmin()
            min_epoch = df["epoch"].iloc[min_idx]
            min_value = df["val/cls_loss"].min()
            ax.annotate(f'Best: {min_value:.4f}\nEpoch: {min_epoch}', 
                        xy=(min_epoch, min_value), 
                        xytext=(min_epoch + 2, min_value + 0.02),
                        arrowprops=dict(arrowstyle="->", color='orange'),
                        color='orange', fontsize=9)
        ax.set_title("Class Loss")
        ax.set_ylabel("Loss")
        ax.legend()
        
        # Add info button
        info = ax.annotate("ℹ", xy=(0.95, 0.95), 
                         xycoords="axes fraction",
                         fontsize=14, 
                         color="blue", 
                         weight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  fc="yellow", 
                                  ec="black", 
                                  lw=1),
                         picker=True,
                         zorder=100)
        info.metric_key = "Class Loss"
        
        # Precision & Recall plot
        ax = axes[2]
        if "metrics/precision(B)" in df.columns:
            ax.plot(epochs, df["metrics/precision(B)"], label="Precision", color='green')
            # Annotate maximum point
            max_idx = df["metrics/precision(B)"].idxmax()
            max_epoch = df["epoch"].iloc[max_idx]
            max_value = df["metrics/precision(B)"].max()
            ax.annotate(f'Best: {max_value:.4f}\nEpoch: {max_epoch}', 
                        xy=(max_epoch, max_value), 
                        xytext=(max_epoch + 2, max_value - 0.05),
                        arrowprops=dict(arrowstyle="->", color='green'),
                        color='green', fontsize=9)
        if "metrics/recall(B)" in df.columns:
            ax.plot(epochs, df["metrics/recall(B)"], label="Recall", color='red')
            # Annotate maximum point
            max_idx = df["metrics/recall(B)"].idxmax()
            max_epoch = df["epoch"].iloc[max_idx]
            max_value = df["metrics/recall(B)"].max()
            ax.annotate(f'Best: {max_value:.4f}\nEpoch: {max_epoch}', 
                        xy=(max_epoch, max_value), 
                        xytext=(max_epoch + 2, max_value - 0.05),
                        arrowprops=dict(arrowstyle="->", color='red'),
                        color='red', fontsize=9)
        ax.set_title("Precision & Recall")
        ax.set_ylabel("Score")
        ax.legend()
        
        # Add info button
        info = ax.annotate("ℹ", xy=(0.95, 0.95), 
                         xycoords="axes fraction",
                         fontsize=14, 
                         color="blue", 
                         weight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  fc="yellow", 
                                  ec="black", 
                                  lw=1),
                         picker=True,
                         zorder=100)
        info.metric_key = "Precision & Recall"
        
        # mAP Scores plot
        ax = axes[3]
        if "metrics/mAP50(B)" in df.columns:
            ax.plot(epochs, df["metrics/mAP50(B)"], label="mAP50", color='purple')
            # Annotate maximum point
            max_idx = df["metrics/mAP50(B)"].idxmax()
            max_epoch = df["epoch"].iloc[max_idx]
            max_value = df["metrics/mAP50(B)"].max()
            ax.annotate(f'Best: {max_value:.4f}\nEpoch: {max_epoch}', 
                        xy=(max_epoch, max_value), 
                        xytext=(max_epoch + 2, max_value - 0.05),
                        arrowprops=dict(arrowstyle="->", color='purple'),
                        color='purple', fontsize=9)
        if "metrics/mAP50-95(B)" in df.columns:
            ax.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP50-95", color='brown')
            # Annotate maximum point
            max_idx = df["metrics/mAP50-95(B)"].idxmax()
            max_epoch = df["epoch"].iloc[max_idx]
            max_value = df["metrics/mAP50-95(B)"].max()
            ax.annotate(f'Best: {max_value:.4f}\nEpoch: {max_epoch}', 
                        xy=(max_epoch, max_value), 
                        xytext=(max_epoch + 2, max_value - 0.03),
                        arrowprops=dict(arrowstyle="->", color='brown'),
                        color='brown', fontsize=9)
        ax.set_title("mAP Scores")
        ax.set_ylabel("Score")
        ax.legend()
        
        # Add info button
        info = ax.annotate("ℹ", xy=(0.95, 0.95), 
                         xycoords="axes fraction",
                         fontsize=14, 
                         color="blue", 
                         weight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  fc="yellow", 
                                  ec="black", 
                                  lw=1),
                         picker=True,
                         zorder=100)
        info.metric_key = "mAP Scores"
        
        # DFL Loss plot
        ax = axes[4]
        if "train/dfl_loss" in df.columns:
            ax.plot(epochs, df["train/dfl_loss"], label="Train", color='blue')
            # Annotate minimum point
            min_idx = df["train/dfl_loss"].idxmin()
            min_epoch = df["epoch"].iloc[min_idx]
            min_value = df["train/dfl_loss"].min()
            ax.annotate(f'Best: {min_value:.4f}\nEpoch: {min_epoch}', 
                        xy=(min_epoch, min_value), 
                        xytext=(min_epoch + 2, min_value + 0.05),
                        arrowprops=dict(arrowstyle="->", color='blue'),
                        color='blue', fontsize=9)
        if "val/dfl_loss" in df.columns:
            ax.plot(epochs, df["val/dfl_loss"], label="Val", color='orange')
            # Annotate minimum point
            min_idx = df["val/dfl_loss"].idxmin()
            min_epoch = df["epoch"].iloc[min_idx]
            min_value = df["val/dfl_loss"].min()
            ax.annotate(f'Best: {min_value:.4f}\nEpoch: {min_epoch}', 
                        xy=(min_epoch, min_value), 
                        xytext=(min_epoch + 2, min_value + 0.02),
                        arrowprops=dict(arrowstyle="->", color='orange'),
                        color='orange', fontsize=9)
        ax.set_title("DFL Loss")
        ax.set_ylabel("Loss")
        ax.legend()
        
        # Add info button
        info = ax.annotate("ℹ", xy=(0.95, 0.95), 
                         xycoords="axes fraction",
                         fontsize=14, 
                         color="blue", 
                         weight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  fc="yellow", 
                                  ec="black", 
                                  lw=1),
                         picker=True,
                         zorder=100)
        info.metric_key = "DFL Loss"
        
        # Learning Rate plot
        ax = axes[5]
        if "lr/pg0" in df.columns:
            ax.plot(epochs, df["lr/pg0"], label="Learning Rate", color='magenta')
        ax.set_title("Learning Rate")
        ax.set_ylabel("LR")
        ax.legend()
        
        # Add info button
        info = ax.annotate("ℹ", xy=(0.95, 0.95), 
                         xycoords="axes fraction",
                         fontsize=14, 
                         color="blue", 
                         weight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  fc="yellow", 
                                  ec="black", 
                                  lw=1),
                         picker=True,
                         zorder=100)
        info.metric_key = "Learning Rate"
        
        # Switch to dashboard tab
        window.tabs.setCurrentIndex(0)
        
        # Update plots
        window.figure.tight_layout()
        window.canvas.draw()
        window.canvas.flush_events()
        
    except Exception as e:
        import traceback
        logger.error(f"Error updating dashboard: {traceback.format_exc()}")