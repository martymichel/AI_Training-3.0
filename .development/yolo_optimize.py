import pandas as pd
import optuna
import torch
from ultralytics import YOLO
import threading

# --- Globale Einstellungen ---
Runs = 1
lock = threading.Lock()  # Für sicheren Zugriff auf das Modell
torch.set_num_threads(14)  # Nutze 14 CPU-Kerne für PyTorch

# --- Pfade ---
model_path = r"G:\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\M_inner_B_Reklamation_3AA01995\Train1\weights\best.pt"
data_path = r"G:\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\M_inner_B_Reklamation_3AA01995\4_Splitted_Train\data.yaml"

# --- Initialisiere Modell ---
model = None

def load_model():
    """Lädt das Modell nur einmal in den Speicher."""
    global model
    with lock:
        if model is None:
            model = YOLO(model_path)


def objective(trial):
    """Optuna-Ziel-Funktion zur Optimierung von Confidence & IoU-Werten."""
    load_model()

    # Lade Klassen dynamisch aus dem Modell
    classes = list(model.names.values())

    # Confidence-Werte pro Klasse optimieren
    conf_values = {cls: trial.suggest_float(f'conf_{cls}', 0.55, 0.95, step=0.05) for cls in classes}
    
    # IoU-Wert optimieren
    iou = trial.suggest_float('iou', 0.1, 0.7, step=0.1)

    total_fitness = 0

    for cls_id, cls_name in model.names.items():
        conf = conf_values[cls_name]  # Konfidenz für diese Klasse

        print("\n" + "-" * 50 + "\n") # Printe einen Querstrich, um die Konsolen-Ausgabe zwischen den Epochen zu trennen
        print(f"🔍 Klasse '{cls_name}': Conf={conf:.2f}, IoU={iou:.2f}") # Printe aktuelle Parameter

        # Führe Validierung mit optimierten Parametern durch
        try:
            result = model.val(
                data=data_path,
                batch=32,  # Wählt optimale Batch-Größe
                imgsz=640,  # Bildgröße bleibt 640x640
                device='cuda:0',  # GPU nutzen
                workers=8,  # Nutzt 8 Worker für parallelen Bild-Load
                conf=conf,  # Klassenbasierte Confidence-Werte
                iou=iou,
                cache='disk',  # Disk-Caching für deterministische Ergebnisse
                half=True,  # FP16 für schnellere Berechnung
                verbose=False  # Kein Spam in der Konsole
            )

            # Berechne Fitness pro Klasse und addiere sie
            class_fitness = result.results_dict.get('fitness', 0)
            total_fitness += class_fitness

        except Exception as e:
            print(f"⚠️ Fehler bei Klasse '{cls_name}': {e}")
            total_fitness += 0  # Falls ein Fehler auftritt, ignoriere diese Klasse

    return total_fitness / max(1, len(classes))  # Durchschnittswert der Fitness

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=Runs)

    best_params = study.best_params
    best_fitness = round(study.best_value, 4)

    # Werte auf 2 Nachkommastellen runden
    formatted_params = {k: round(v, 2) for k, v in best_params.items()}

    # DataFrame zur besseren Darstellung in der Konsole
    df = pd.DataFrame(formatted_params.items(), columns=["Parameter", "Wert"])

    # Konsolenausgabe
    print("\n🎯 Optimierte Parameter:\n")
    print(df.to_string(index=False))
    print(f"\n📈 Beste Fitness: {best_fitness}")
    # Erklärung, was die Fitness bedeutet
    print("\n🔍 Fitness: Durchschnittliche Genauigkeit der Klassifizierung über alle Klassen.")
