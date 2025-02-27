import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import logging

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QMessageBox
from PyQt6.QtCore import QTimer, QThread, pyqtSignal

# Logger konfigurieren
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


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
        self.last_mod_time = None

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
                    logger.debug("results.csv noch nicht gefunden. Warte...")
                    time.sleep(self.update_interval)
                    continue

                current_mod_time = os.path.getmtime(csv_path)
                if self.last_mod_time is not None and current_mod_time == self.last_mod_time:
                    time.sleep(self.update_interval)
                    continue
                self.last_mod_time = current_mod_time

                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()

                if len(df) < 1:
                    logger.debug("Noch zu wenige Epochen... Warte auf erste Ergebnisse.")
                    time.sleep(self.update_interval)
                    continue

                logger.info("results.csv erfolgreich geladen.")
                self.update_signal.emit(df)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Fehler beim Dashboard-Update: {e}")
                time.sleep(self.update_interval)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class DashboardWindow(QMainWindow):
    """
    Dashboard-Fenster zur Visualisierung der Trainingsmetriken.
    
    Jeder Subplot enthält ein kleines Info‑Symbol ("ℹ"), auf das man klicken kann, um
    alle wichtigen Informationen zur jeweiligen Metrik angezeigt zu bekommen.
    
    Die X-Achse wird fix auf die vorgegebene Gesamtzahl an Epochen skaliert.
    """
    def __init__(self, project="yolo_training_results", experiment="experiment", total_epochs=100):
        super().__init__()
        self.project = project
        self.experiment = experiment
        self.total_epochs = total_epochs
        self.setWindowTitle("YOLO Trainings-Dashboard")
        self.setGeometry(200, 200, 1400, 800)

        # Infos zu den Metriken
        self.metric_info_dict = {
            "Box Loss": "Box Loss (Train & Val) – Darstellung der Verlustwerte für Boxen. Niedrigere Werte sind besser.",
            "Class Loss": "Class Loss (Train & Val) – Verlustwerte für die Klassifizierung. Niedrigere Werte sind besser.",
            "Precision & Recall": "Precision & Recall – Metriken zur Bewertung der Vorhersagegenauigkeit. Höhere Werte sind besser.",
            "mAP Scores": "mAP Scores – Mittlere Average Precision (mAP50 und mAP50-95) zur Beurteilung der Erkennungsleistung.",
            "DFL Loss": "DFL Loss (Train & Val) – Distribution Focal Loss als Ersatz für F1-Score. Niedrigere Werte sind besser.",
            "Learning Rate": "Learning Rate – Aktuelle Lernrate des Modells, zeigt, wie stark die Gewichte aktualisiert werden."
        }

        # Zentrales Widget und Layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.info_label = QLabel("Trainingsstart wird abgewartet...")
        layout.addWidget(self.info_label)

        try:
            if "seaborn-whitegrid" in plt.style.available:
                plt.style.use("seaborn-whitegrid")
            else:
                plt.style.use("seaborn-darkgrid")
        except Exception as e:
            logger.error(f"Fehler beim Setzen des Styles: {e}")
            plt.style.use("default")

        self.figure, self.axes = plt.subplots(3, 2, figsize=(16, 12))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Liste für Info-Anmerkungen (Info-Icons)
        self.info_annotations = []

        # Mit dem Pick-Event werden Klicks auf die Info-Icons erkannt.
        self.canvas.mpl_connect("pick_event", self._on_info_pick)

        self.dashboard_worker = DashboardWorker(self.project, self.experiment)
        self.dashboard_worker.update_signal.connect(self.update_dashboard)
        self.dashboard_worker.start()

    def update_dashboard(self, df):
        if not self.isVisible():
            return
        QTimer.singleShot(0, lambda: self._safe_update(df))

    def _safe_update(self, df):
        try:
            self.info_label.setText("")
            if "epoch" not in df.columns:
                logger.error("Spalte 'epoch' fehlt in der CSV.")
                return

            epochs = df["epoch"]

            def setup_axis(ax):
                ax.set_xlim(0, self.total_epochs)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Entferne alte Info-Icons
            for annot in self.info_annotations:
                annot.remove()
            self.info_annotations = []

            # Für jeden Subplot: Plot aktualisieren und Info-Icon hinzufügen.

            # [0,0]: Box Loss
            ax = self.axes[0, 0]
            ax.cla()
            if "train/box_loss" in df.columns:
                ax.plot(epochs, df["train/box_loss"], label="Train Box Loss", color='blue')
            else:
                ax.text(0.5, 0.5, "Train Box Loss fehlt", ha="center", va="center", transform=ax.transAxes)
            if "val/box_loss" in df.columns:
                ax.plot(epochs, df["val/box_loss"], label="Val Box Loss", color='orange')
            ax.set_title("Box Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
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
            else:
                ax.text(0.5, 0.5, "Train Class Loss fehlt", ha="center", va="center", transform=ax.transAxes)
            if "val/cls_loss" in df.columns:
                ax.plot(epochs, df["val/cls_loss"], label="Val Class Loss", color='orange')
            ax.set_title("Class Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
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
            else:
                ax.text(0.5, 0.5, "Precision fehlt", ha="center", va="center", transform=ax.transAxes)
            if "metrics/recall(B)" in df.columns:
                ax.plot(epochs, df["metrics/recall(B)"], label="Recall", color='red')
            ax.set_title("Precision & Recall")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
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
            else:
                ax.text(0.5, 0.5, "mAP50 fehlt", ha="center", va="center", transform=ax.transAxes)
            if "metrics/mAP50-95(B)" in df.columns:
                ax.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP50-95", color='brown')
            ax.set_title("mAP Scores")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
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
            else:
                ax.text(0.5, 0.5, "Train DFL Loss fehlt", ha="center", va="center", transform=ax.transAxes)
            if "val/dfl_loss" in df.columns:
                ax.plot(epochs, df["val/dfl_loss"], label="Val DFL Loss", color='orange')
            ax.set_title("DFL Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
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
            else:
                ax.text(0.5, 0.5, "Learning Rate fehlt", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Learning Rate")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("LR")
            ax.legend()
            ax.grid(True)
            setup_axis(ax)
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
            logger.error(f"Fehler beim Aktualisieren der Dashboard-UI: {e}")

    def _on_info_pick(self, event):
        artist = event.artist
        if hasattr(artist, "metric_key"):
            metric = artist.metric_key
            info_text = self.metric_info_dict.get(metric, "Keine Informationen verfügbar.")
            QMessageBox.information(self, f"Information zu {metric}", info_text)

    def closeEvent(self, event):
        self.hide()
        event.ignore()
