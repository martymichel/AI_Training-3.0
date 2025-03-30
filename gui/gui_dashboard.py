"""Dashboard module for YOLO training visualization."""

import os
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configure matplotlib logging - SUPPRESS DEBUG MESSAGES
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QLabel, QMessageBox, QHBoxLayout,
    QPushButton, QFileDialog
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator

# Disable all matplotlib logging
plt.set_loglevel('critical')


class DashboardWorker(QThread):
    """
    Überwacht periodisch die results.csv im angegebenen Projekt/Experiment-Verzeichnis.
    Sendet das geladene DataFrame nur, wenn sich die Datei (mtime) geändert hat.
    """
    update_signal = pyqtSignal(pd.DataFrame)

    def __init__(self, project, experiment, update_interval=10):
        super().__init__()
        self.project = project
        self.experiment = experiment
        self.update_interval = update_interval  # in Sekunden
        self.running = True
        self.last_mod_time = 0

    def find_results_csv(self):
        """Sucht im Projekt/Experiment-Verzeichnis nach 'results.csv'."""
        base_path = os.path.join(self.project, self.experiment)
        for root, dirs, files in os.walk(base_path):
            if "results.csv" in files:
                return os.path.join(root, "results.csv")
        return None

    def run(self):
        while self.running:
            try:
                csv_path = self.find_results_csv()
                if not csv_path:
                    time.sleep(self.update_interval)
                    continue

                current_mod_time = os.path.getmtime(csv_path)
                if current_mod_time == self.last_mod_time:
                    time.sleep(self.update_interval)
                    continue
                self.last_mod_time = current_mod_time

                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()

                if len(df) < 1:
                    time.sleep(self.update_interval)
                    continue

                self.update_signal.emit(df)
                time.sleep(self.update_interval)
            except Exception as e:
                time.sleep(self.update_interval)

    def stop(self):
        self.running = False
        self.quit()
        self.wait(1000)  # Wait up to 1 second

    def __del__(self):
        self.stop()


class DashboardWindow(QMainWindow):
    """
    Dashboard-Fenster zur Visualisierung der Trainingsmetriken.
    
    Jeder Subplot enthält ein kleines Info‑Symbol ("ℹ"), auf das man klicken kann, um
    alle wichtigen Informationen zur jeweiligen Metrik angezeigt zu bekommen.
    
    Die X-Achse wird fix auf die vorgegebene Gesamtzahl an Epochen skaliert.
    """
    def __init__(self, project="yolo_training_results", experiment="experiment", total_epochs=100):
        super().__init__()
        self.project = None
        self.experiment = None
        self.dashboard_worker = None
        self.total_epochs = total_epochs
        self.setWindowTitle("YOLO Trainings-Dashboard")
        self.setGeometry(200, 200, 1400, 800)
        
        # Set up matplotlib
        try:
            if "seaborn-whitegrid" in plt.style.available:
                plt.style.use("seaborn-whitegrid")
            else:
                plt.style.use("seaborn-darkgrid")
        except Exception:
            plt.style.use("default")

        # Create figure and canvas
        self.figure, self.axes = plt.subplots(3, 2, figsize=(16, 12))
        self.canvas = FigureCanvas(self.figure)
        
        # Liste für Info-Anmerkungen (Info-Icons)
        self.info_annotations = []

        # Infos zu den Metriken
        self.metric_info_dict = {
            "Box Loss": """Box Loss (Train & Val) – Diese Kennzahl misst, wie gut das Modell die Position der erkannten Objekte innerhalb eines Bildes bestimmt.
            Ein niedriger Wert bedeutet, dass die vorhergesagten Begrenzungsrahmen (Bounding Boxes) nah an den tatsächlichen Positionen liegen.
            
            Bewertung:
            - Sehr gut: < 0.05 – Das Modell lokalisiert Objekte extrem genau.
            - Gut: 0.05 - 0.1 – Geringe Abweichungen bei der Objektlokalisierung.
            - Akzeptabel: 0.1 - 0.3 – Leichte Ungenauigkeiten, kann aber noch funktionieren.
            - Schlecht: > 0.3 – Die Boxen weichen zu stark von den realen Positionen ab und könnten Fehler verursachen.""",

            "Class Loss": """Class Loss (Train & Val) – Diese Metrik bewertet, wie gut das Modell zwischen verschiedenen Fehlerklassen (z. B. Gratbildung, Risse) unterscheiden kann.
            Ein niedriger Wert bedeutet, dass das Modell die Klassen zuverlässig erkennt.
            
            Bewertung:
            - Sehr gut: < 0.02 – Fast perfekte Klassifikation.
            - Gut: 0.02 - 0.05 – Hohe Genauigkeit, aber leichte Fehler möglich.
            - Akzeptabel: 0.05 - 0.15 – Einige falsche Klassifikationen, Modell ist verbesserungsfähig.
            - Schlecht: > 0.15 – Häufige Verwechslungen zwischen Fehlerarten, führt zu hoher Falschklassifikationsrate.""",

            "Precision & Recall": """Precision & Recall – Diese beiden Werte zeigen an, wie gut das Modell Fehler erkennt, ohne zu viele falsche Alarme zu erzeugen.
            
            - **Precision (Präzision)**: Gibt an, wie viele der als fehlerhaft erkannten Teile tatsächlich fehlerhaft sind. Hohe Präzision bedeutet wenige Falsch-Positive (Teile werden fälschlicherweise als fehlerhaft erkannt).
            - **Recall (Empfindlichkeit)**: Zeigt, wie viele der real fehlerhaften Teile auch tatsächlich als fehlerhaft erkannt wurden. Ein hoher Wert bedeutet, dass das Modell kaum Fehler übersieht.
            
            Bewertung:
            - Sehr gut: Precision & Recall > 95% – Fast perfekte Erkennung.
            - Gut: 90 - 95% – Kaum Fehler übersehen, wenige Falsch-Positive.
            - Akzeptabel: 80 - 90% – Funktioniert gut, aber kann vereinzelt Fehler übersehen oder Fehlalarme erzeugen.
            - Schlecht: < 80% – Hohe Rate an übersehenen Fehlern oder falschen Alarmen, nicht für die Produktion geeignet.""",

            "mAP Scores": """mAP Scores – Diese Kennzahl gibt an, wie gut das Modell Fehler in verschiedenen Bereichen des Bildes erkennt. Sie wird als Durchschnitt aller Vorhersagegenauigkeiten über verschiedene Schwellenwerte berechnet.
            
            - **mAP50** (Mittlere Average Precision bei 50% Übereinstimmung): Gibt an, wie gut das Modell Fehler findet, wenn es eine 50%ige Übereinstimmung mit der tatsächlichen Fehlerposition gibt.
            - **mAP50-95** (Mittlere Average Precision über verschiedene Schwellenwerte von 50% bis 95%): Bewertet, wie gut das Modell unter strengen Genauigkeitsanforderungen arbeitet.
            
            Bewertung:
            - Sehr gut: mAP50 > 95% und mAP50-95 > 80% – Sehr zuverlässige Fehlererkennung.
            - Gut: mAP50 90 - 95% und mAP50-95 70 - 80% – Funktioniert gut, leichte Verbesserungen möglich.
            - Akzeptabel: mAP50 80 - 90% und mAP50-95 60 - 70% – Gute Grundgenauigkeit, aber feinere Optimierung erforderlich.
            - Schlecht: mAP50 < 80% oder mAP50-95 < 60% – Hohe Fehlerrate, nicht praxistauglich.""",

            "DFL Loss": """DFL Loss (Train & Val) – Distribution Focal Loss bewertet die Unsicherheit des Modells bei der Positionsvorhersage von Fehlern.
            Ein hoher Wert zeigt, dass das Modell sich nicht sicher ist, wo genau sich der Fehler befindet.
            
            Bewertung:
            - Sehr gut: < 0.03 – Hohe Sicherheit bei der Fehlerlokalisierung.
            - Gut: 0.03 - 0.08 – Leichte Unsicherheiten, aber akzeptabel.
            - Akzeptabel: 0.08 - 0.15 – Erkennbar unsicher, könnte Fehlerpositionen ungenau bestimmen.
            - Schlecht: > 0.15 – Modell ist unsicher, Fehlerlokalisierung unzuverlässig.""",

            "Learning Rate": """Learning Rate – Dieser Wert zeigt, wie stark das Modell seine Parameter bei jedem Trainingsschritt anpasst.
            Eine zu hohe Lernrate führt dazu, dass das Modell nicht stabil lernt, eine zu niedrige Lernrate führt zu langsamen oder unvollständigem Lernen.
            
            Bewertung:
            - Sehr gut: 0.001 - 0.005 – Optimale Lernrate für stabile Anpassung.
            - Gut: 0.005 - 0.01 – Etwas aggressiver, aber meist stabil.
            - Akzeptabel: 0.01 - 0.02 – Risiko für instabiles Training, aber kann in manchen Fällen funktionieren.
            - Schlecht: > 0.02 oder < 0.0005 – Entweder zu schnell (instabiles Lernen) oder zu langsam (sehr langes Training)."""
        }


        # Zentrales Widget und Layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        self.setCentralWidget(central_widget)

        # Project selection
        select_layout = QHBoxLayout()
        self.project_label = QLabel("Kein Projekt ausgewählt")
        select_btn = QPushButton("Projekt auswählen...")
        select_btn.clicked.connect(self.select_project)
        select_layout.addWidget(self.project_label)
        select_layout.addWidget(select_btn)
        layout.addLayout(select_layout)

        # Add canvas to layout
        layout.addWidget(self.canvas)

        # Status label at the bottom
        self.info_label = QLabel("Bitte wählen Sie ein Projekt aus")
        layout.addWidget(self.info_label)

        # Connect canvas pick event
        self.canvas.mpl_connect("pick_event", self._on_info_pick)

    def select_project(self):
        """Open file dialog to select project directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Projektverzeichnis wählen"
        )
        if directory:
            # Look for results.csv in the directory
            for root, dirs, files in os.walk(directory):
                if "results.csv" in files:
                    self.project = os.path.dirname(root)  # Parent of the directory containing results.csv
                    self.experiment = os.path.basename(root)  # Directory containing results.csv
                    self.project_label.setText(f"Projekt: {os.path.basename(self.project)}/{self.experiment}")
                    
                    # Start monitoring
                    if self.dashboard_worker:
                        self.dashboard_worker.stop()
                    self.dashboard_worker = DashboardWorker(self.project, self.experiment)
                    self.dashboard_worker.update_signal.connect(self.update_dashboard)
                    self.dashboard_worker.start()
                    
                    self.info_label.setText("Überwache Training...")
                    return
                    
            QMessageBox.warning(
                self,
                "Keine Trainingsdaten gefunden",
                "Keine results.csv im ausgewählten Verzeichnis gefunden."
            )


    def update_dashboard(self, df):
        if not self.isVisible():
            return
        QTimer.singleShot(0, lambda: self._safe_update(df))

    def _safe_update(self, df):
        try:
            if "epoch" not in df.columns:
                return

            epochs = df["epoch"]

            def setup_axis(ax):
                ax.set_xlim(0, max(101, int(df['epoch'].max()) + 1))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Entferne alte Info-Icons
            for annot in self.info_annotations:
                annot.remove()
            self.info_annotations = []

            # [0,0]: Box Loss
            ax = self.axes[0, 0]
            ax.cla()
            if "train/box_loss" in df.columns:
                ax.plot(epochs, df["train/box_loss"], label="Train Box Loss", color='blue')
                min_loss_idx = df["train/box_loss"].idxmin()
                best_epoch = df["epoch"].iloc[min_loss_idx]
                best_value = df["train/box_loss"].min()
                ax.annotate(f'Best: {best_value:.4f}\nEpoch: {best_epoch}', xy=(best_epoch, best_value), 
                            xytext=(best_epoch, best_value + 0.1),
                            arrowprops=dict(arrowstyle='->', color='blue'),
                            color='blue', fontsize=9, ha='center')
            else:
                ax.text(0.5, 0.5, "Train Box Loss fehlt", ha="center", va="center", transform=ax.transAxes)
            if "val/box_loss" in df.columns:
                ax.plot(epochs, df["val/box_loss"], label="Val Box Loss", color='orange')
                min_val_loss_idx = df["val/box_loss"].idxmin()
                best_val_epoch = df["epoch"].iloc[min_val_loss_idx]
                best_val_value = df["val/box_loss"].min()
                ax.annotate(f'Best: {best_val_value:.4f}\nEpoch: {best_val_epoch}', xy=(best_val_epoch, best_val_value), 
                            xytext=(best_val_epoch, best_val_value + 0.1),
                            arrowprops=dict(arrowstyle='->', color='orange'),
                            color='orange', fontsize=9, ha='center')
            ax.set_title("Box Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
            
            # Info-Icon hinzufügen
            annot = ax.annotate("ℹ", xy=(0.95, 0.95), xycoords="axes fraction",
                                fontsize=12, color="blue", weight="bold",
                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1),
                                picker=True)
            annot.metric_key = "Box Loss"
            self.info_annotations.append(annot)

            # [0,1]: Class Loss
            ax = self.axes[0, 1]
            ax.cla()
            if "train/cls_loss" in df.columns:
                ax.plot(epochs, df["train/cls_loss"], label="Train Class Loss", color='blue')
                min_cls_loss_idx = df["train/cls_loss"].idxmin()
                best_cls_epoch = df["epoch"].iloc[min_cls_loss_idx]
                best_cls_value = df["train/cls_loss"].min()
                ax.annotate(f'Best: {best_cls_value:.4f}\nEpoch: {best_cls_epoch}', xy=(best_cls_epoch, best_cls_value), 
                            xytext=(best_cls_epoch, best_cls_value + 0.1),
                            arrowprops=dict(arrowstyle='->', color='blue'),
                            color='blue', fontsize=9, ha='center')
            else:
                ax.text(0.5, 0.5, "Train Class Loss fehlt", ha="center", va="center", transform=ax.transAxes)
            if "val/cls_loss" in df.columns:
                ax.plot(epochs, df["val/cls_loss"], label="Val Class Loss", color='orange')
                min_val_cls_loss_idx = df["val/cls_loss"].idxmin()
                best_val_cls_epoch = df["epoch"].iloc[min_val_cls_loss_idx]
                best_val_cls_value = df["val/cls_loss"].min()
                ax.annotate(f'Best: {best_val_cls_value:.4f}\nEpoch: {best_val_cls_epoch}', xy=(best_val_cls_epoch, best_val_cls_value), 
                            xytext=(best_val_cls_epoch, best_val_cls_value + 0.1),
                            arrowprops=dict(arrowstyle='->', color='orange'),
                            color='orange', fontsize=9, ha='center')
            ax.set_title("Class Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
            
            # Info-Icon hinzufügen
            annot = ax.annotate("ℹ", xy=(0.95, 0.95), xycoords="axes fraction",
                                fontsize=12, color="blue", weight="bold",
                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1),
                                picker=True)
            annot.metric_key = "Class Loss"
            self.info_annotations.append(annot)

            # [1,0]: Precision & Recall
            ax = self.axes[1, 0]
            ax.cla()
            if "metrics/precision(B)" in df.columns:
                ax.plot(epochs, df["metrics/precision(B)"], label="Precision", color='green')
                max_precision_idx = df["metrics/precision(B)"].idxmax()
                best_precision_epoch = df["epoch"].iloc[max_precision_idx]
                best_precision_value = df["metrics/precision(B)"].max()
                ax.annotate(f'Best: {best_precision_value:.2f}\nEpoch: {best_precision_epoch}', xy=(best_precision_epoch, best_precision_value), 
                            xytext=(best_precision_epoch, best_precision_value + 0.02),
                            arrowprops=dict(arrowstyle='->', color='green'),
                            color='green', fontsize=9, ha='center')
            else:
                ax.text(0.5, 0.5, "Precision fehlt", ha="center", va="center", transform=ax.transAxes)
            if "metrics/recall(B)" in df.columns:
                ax.plot(epochs, df["metrics/recall(B)"], label="Recall", color='red')
                max_recall_idx = df["metrics/recall(B)"].idxmax()
                best_recall_epoch = df["epoch"].iloc[max_recall_idx]
                best_recall_value = df["metrics/recall(B)"].max()
                ax.annotate(f'Best: {best_recall_value:.2f}\nEpoch: {best_recall_epoch}', xy=(best_recall_epoch, best_recall_value), 
                            xytext=(best_recall_epoch, best_recall_value + 0.02),
                            arrowprops=dict(arrowstyle='->', color='red'),
                            color='red', fontsize=9, ha='center')
            ax.set_title("Precision & Recall")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
            
            # Info-Icon hinzufügen
            annot = ax.annotate("ℹ", xy=(0.95, 0.95), xycoords="axes fraction",
                                fontsize=12, color="blue", weight="bold",
                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1),
                                picker=True)
            annot.metric_key = "Precision & Recall"
            self.info_annotations.append(annot)

            # [1,1]: mAP Scores
            ax = self.axes[1, 1]
            ax.cla()
            if "metrics/mAP50(B)" in df.columns:
                ax.plot(epochs, df["metrics/mAP50(B)"], label="mAP50", color='purple')
                max_mAP50_idx = df["metrics/mAP50(B)"].idxmax()
                best_mAP50_epoch = df["epoch"].iloc[max_mAP50_idx]
                best_mAP50_value = df["metrics/mAP50(B)"].max()
                ax.annotate(f'Best: {best_mAP50_value:.2f}\nEpoch: {best_mAP50_epoch}', xy=(best_mAP50_epoch, best_mAP50_value), 
                            xytext=(best_mAP50_epoch, best_mAP50_value + 0.02),
                            arrowprops=dict(arrowstyle='->', color='purple'),
                            color='purple', fontsize=9, ha='center')
            else:
                ax.text(0.5, 0.5, "mAP50 fehlt", ha="center", va="center", transform=ax.transAxes)
            if "metrics/mAP50-95(B)" in df.columns:
                ax.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP50-95", color='brown')
                max_mAP50_95_idx = df["metrics/mAP50-95(B)"].idxmax()
                best_mAP50_95_epoch = df["epoch"].iloc[max_mAP50_95_idx]
                best_mAP50_95_value = df["metrics/mAP50-95(B)"].max()
                ax.annotate(f'Best: {best_mAP50_95_value:.2f}\nEpoch: {best_mAP50_95_epoch}', xy=(best_mAP50_95_epoch, best_mAP50_95_value), 
                            xytext=(best_mAP50_95_epoch, best_mAP50_95_value + 0.02),
                            arrowprops=dict(arrowstyle='->', color='brown'),
                            color='brown', fontsize=9, ha='center')
            ax.set_title("mAP Scores")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
            
            # Info-Icon hinzufügen
            annot = ax.annotate("ℹ", xy=(0.95, 0.95), xycoords="axes fraction",
                                fontsize=12, color="blue", weight="bold",
                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1),
                                picker=True)
            annot.metric_key = "mAP Scores"
            self.info_annotations.append(annot)

            # [2,0]: DFL Loss
            ax = self.axes[2, 0]
            ax.cla()
            if "train/dfl_loss" in df.columns:
                ax.plot(epochs, df["train/dfl_loss"], label="Train DFL Loss", color='blue')
                min_dfl_loss_idx = df["train/dfl_loss"].idxmin()
                best_dfl_epoch = df["epoch"].iloc[min_dfl_loss_idx]
                best_dfl_value = df["train/dfl_loss"].min()
                ax.annotate(f'Best: {best_dfl_value:.4f}\nEpoch: {best_dfl_epoch}', xy=(best_dfl_epoch, best_dfl_value), 
                            xytext=(best_dfl_epoch, best_dfl_value + 0.1),
                            arrowprops=dict(arrowstyle='->', color='blue'),
                            color='blue', fontsize=9, ha='center')
            else:
                ax.text(0.5, 0.5, "Train DFL Loss fehlt", ha="center", va="center", transform=ax.transAxes)
            if "val/dfl_loss" in df.columns:
                ax.plot(epochs, df["val/dfl_loss"], label="Val DFL Loss", color='orange')
                min_val_dfl_loss_idx = df["val/dfl_loss"].idxmin()
                best_val_dfl_epoch = df["epoch"].iloc[min_val_dfl_loss_idx]
                best_val_dfl_value = df["val/dfl_loss"].min()
                ax.annotate(f'Best: {best_val_dfl_value:.4f}\nEpoch: {best_val_dfl_epoch}', xy=(best_val_dfl_epoch, best_val_dfl_value), 
                            xytext=(best_val_dfl_epoch, best_val_dfl_value + 0.1),
                            arrowprops=dict(arrowstyle='->', color='orange'),
                            color='orange', fontsize=9, ha='center')
            ax.set_title("DFL Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
            
            # Info-Icon hinzufügen
            annot = ax.annotate("ℹ", xy=(0.95, 0.95), xycoords="axes fraction",
                                fontsize=12, color="blue", weight="bold",
                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1),
                                picker=True)
            annot.metric_key = "DFL Loss"
            self.info_annotations.append(annot)

            # [2,1]: Learning Rate
            ax = self.axes[2, 1]
            ax.cla()
            if "lr/pg0" in df.columns:
                ax.plot(epochs, df["lr/pg0"], label="Learning Rate", color='magenta')
                # Hier könntest du den besten Learning Rate Wert finden und annotieren, wenn relevant
            else:
                ax.text(0.5, 0.5, "Learning Rate fehlt", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Learning Rate")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("LR")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
            
            # Info-Icon hinzufügen
            annot = ax.annotate("ℹ", xy=(0.95, 0.95), xycoords="axes fraction",
                                fontsize=12, color="blue", weight="bold",
                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1),
                                picker=True)
            annot.metric_key = "Learning Rate"
            self.info_annotations.append(annot)

            self.figure.tight_layout()
            self.canvas.draw()
            self.canvas.flush_events()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update dashboard: {str(e)}")

    def _on_info_pick(self, event):
        artist = event.artist
        if hasattr(artist, "metric_key"):
            metric = artist.metric_key
            info_text = self.metric_info_dict.get(metric, "Keine Informationen verfügbar.")
            QMessageBox.information(self, f"Information zu {metric}", info_text)

    def closeEvent(self, event):
        if self.dashboard_worker:
            self.dashboard_worker.stop()
            self.dashboard_worker = None
        self.hide()
        event.ignore()