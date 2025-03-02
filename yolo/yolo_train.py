"""YOLO training module with threading support."""

from ultralytics import YOLO
from threading import Thread
import os
import pandas as pd
import time
import logging
from config import Config

# Logger konfigurieren
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def start_training_threaded(data_path, epochs, imgsz, batch, lr0, optimizer, augment, project, name, progress_callback=None, log_callback=None):
    """Start YOLO training in a separate thread with progress monitoring.
    
    Args:
        data_path (str): Path to YAML dataset file
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch (int): Batch size
        lr0 (float): Initial learning rate
        optimizer (str): Optimizer name (AdamW, Adam, SGD)
        augment (bool): Use data augmentation
        project (str): Project directory
        name (str): Experiment name
        progress_callback (callable): Callback function to update progress
        log_callback (callable): Callback function to log messages
    """
    def train():
        try:
            logger.info("Starte Training mit folgenden Parametern:")
            logger.info(f"Epochs: {epochs}, Image Size: {imgsz}, Batch: {batch}")
            logger.info(f"Learning Rate: {lr0}, Optimizer: {optimizer}")
            
            model = YOLO("yolo11n.pt")

            # YOLO Training starten
            model.train(
                data=data_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr0,
                workers=os.cpu_count()-1,  # Dynamisch verfügbare Worker ermitteln
                optimizer=optimizer,
                augment=augment,
                device=0,  # Nutze GPU 0
                project=project,
                name=name
            )

            logger.info("Training erfolgreich abgeschlossen")
            if progress_callback:
                progress_callback(100)

        except Exception as e:
            logger.error(f"Fehler während des Trainings: {str(e)}")
            if progress_callback:
                progress_callback(0, str(e))

    def monitor_progress():
        """Monitor training progress by reading results.csv."""
        csv_path = os.path.join(project, name, "results.csv")

        while not os.path.exists(csv_path):
            logger.debug("Warte auf results.csv...")
            time.sleep(5)

        while True:
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()

                current_epoch = len(df)
                progress = (current_epoch / epochs) * 100
                logger.debug(f"Trainingsfortschritt: {progress:.1f}%")

                if progress_callback:
                    progress_callback(progress)

                if progress >= 100:
                    logger.info("Monitoring beendet - Training abgeschlossen")
                    break

                time.sleep(10)

            except Exception as e:
                logger.error(f"Fehler beim Fortschritt-Tracking: {str(e)}")
                time.sleep(10)

    # Training in eigenem Thread starten
    training_thread = Thread(target=train)
    training_thread.start()

    # Fortschrittsüberwachung in eigenem Thread starten
    monitor_thread = Thread(target=monitor_progress)
    monitor_thread.start()
