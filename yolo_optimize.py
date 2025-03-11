import optuna
import os
import torch
from ultralytics import YOLO

# Modell nur einmal laden!
model_path = r"C:\Users\miche\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\M_inner_B_Reklamation_3AA01995\Train1\weights\best.pt"
data_path = r"C:\Users\miche\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\M_inner_B_Reklamation_3AA01995\4_Splitted_Train\data - ASUS.yaml"

model = YOLO(model_path)

# Versuche, das Fusing zu deaktivieren, falls nötig
try:
    model.fuse()  # Falls es keinen Fehler gibt, ist Fusing aktiv.
    print("Fusion erfolgreich durchgeführt.")
except AttributeError:
    print("Fusing nicht möglich, vermutlich bereits deaktiviert.")

def objective(trial):
    conf = trial.suggest_float('conf', 0.5, 0.95, step=0.05)
    iou = trial.suggest_float('iou', 0.1, 0.8, step=0.1)

    result = model.val(
        data=data_path,
        device='0',
        conf=conf,
        iou=iou,
        verbose=False
    )

    return result.results_dict['fitness']

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')

    # Parallelisierung auf mehrere Threads!
    study.optimize(objective, n_trials=30, n_jobs=4)

    print("Beste Parameter:", study.best_params)
    print("Beste Fitness:", study.best_value)