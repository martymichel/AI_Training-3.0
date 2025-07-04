# %% [markdown]
# # AUGMENTATION
# 
# Augmentation bedeutet, dass wir den Bild-Datensatz erweitern, indem wir jedes Bild auf vielfältige Weise verändern. Das ist wichtig, weil wir so mehr Daten haben, um das Modell zu trainieren.

# %%
# 12.11.2024 --> USE PYTHON 3.10.6

# %pip install albumentations --quiet

import os
import cv2
import sys
import glob
import random
import shutil
import numpy as np
from glob import glob
from itertools import product
from albumentations import (
    Compose, Rotate, RandomScale, OpticalDistortion, HorizontalFlip,
    HueSaturationValue, Blur, RandomBrightnessContrast
)
from albumentations.pytorch import ToTensorV2


# %%
# === Zentrale Konfiguration importieren ===
# Annahme: config.py liegt im Projektroot, Skript im Ordner "scripts" oder auf gleicher Ebene.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import LABELED_DIR, AUGMENTED_DIR

IMAGES_DIR = os.path.join(LABELED_DIR, "images")
LABELS_DIR = os.path.join(LABELED_DIR, "labels")
OUT_IMAGES_DIR = os.path.join(AUGMENTED_DIR, "images")
OUT_LABELS_DIR = os.path.join(AUGMENTED_DIR, "labels")

# Ordner anlegen, falls nicht vorhanden
os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_LABELS_DIR, exist_ok=True)

# %% [markdown]
# ---
# 
# ## Konfiguration
# 
# ### Augmentierungen aktivieren/deaktivieren und Stufen festlegen
# 
# - Die Stufen sind die Werte, die in der Augmentierung verwendet werden.
# - Beispiel: [0, 22, 45] bedeutet, dass die Augmentierung mit 0°, 22° und 45° angewendet wird.
# - Wähle True/False für die Aktivierung der Augmentierung und die Stufen entsprechend den gewünschten Werten.
# - Je mehr Arten von Augmentierungen du verwendest, desto mehr Bilder werden generiert.
# - Aber sei vorsichtig, denn es kann sein, dass die Augmentierungen nicht immer sinnvoll sind. Und es werden viele Bilder generiert.
# - Beispiel: Wenn du 3 Augmentierungen mit jeweils 3 Stufen hast, werden aus einem Bild neu 3^3 = 27 Bilder generiert.

# %%
# Einstellungen

AUG_CONFIG = {
    "rotate":      {"active": True,  "levels": [0, 22, 45]},       # Grad (°) Ränder werden aufgefüllt
    "scale":       {"active": True,  "levels": [0.8, 1.0, 1.2]}, # Faktor des Zooms (1.0 = 100%) Ränder werden aufgefüllt
    "distort":     {"active": False,  "levels": [0, 0.2, 0.4]}, # Verzerrung (Fischaugen Objektiv und Gegenteil davon)
    "hflip":       {"active": True,  "levels": [0, 1, 2]},       # 0 = nicht, 1 = flip, 2 = probabilistisch
    "hsv":         {"active": False,  "levels": [0, 15, 30]},     # Farbverschiebung hsv (h = Hue, s = Saturation, v = Value)
    "blur":        {"active": False,  "levels": [0, 1, 2]},       # 0 = nicht, 1 = leicht, 2 = stark verwaschene Kanten
    "brightness":  {"active": False,  "levels": [-0.15, 0, 0.15]}    # Kontrast-Shift
}


# %% [markdown]
# ---
# 
# ### Funktionen definieren

# %%
def parse_yolo_label(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        split = line.strip().split()
        labels.append([int(split[0])] + [float(x) for x in split[1:]])
    return labels

def save_yolo_label(label_path, labels):
    with open(label_path, "w") as f:
        for l in labels:
            lstr = [str(l[0])] + [f"{x:.6f}" for x in l[1:]]
            f.write(" ".join(lstr) + "\n")

# Generiere alle Stufenkombinationen
active_augs = [k for k, v in AUG_CONFIG.items() if v["active"]]
levels = [AUG_CONFIG[k]["levels"] for k in active_augs]
combinations = list(product(*levels))

def get_aug(aug_setting):
    # aug_setting: Tuple mit allen Stufen, z. B. (7, 1.1, 0.05, ...)
    transforms = []
    setting = dict(zip(active_augs, aug_setting))
    
    if "rotate" in setting and setting["rotate"] != 0:
        transforms.append(Rotate(limit=(setting["rotate"], setting["rotate"]), p=1.0))
    if "scale" in setting and setting["scale"] != 1.0:
        transforms.append(RandomScale(scale_limit=(setting["scale"]-1, setting["scale"]-1), p=1.0))
    if "distort" in setting and setting["distort"] != 0:
        transforms.append(OpticalDistortion(distort_limit=setting["distort"], p=1.0))
    if "hflip" in setting:
        if setting["hflip"] == 1:
            transforms.append(HorizontalFlip(p=1.0))
        elif setting["hflip"] == 2:
            transforms.append(HorizontalFlip(p=0.5))
    if "hsv" in setting and setting["hsv"] != 0:
        transforms.append(HueSaturationValue(hue_shift_limit=setting["hsv"], sat_shift_limit=setting["hsv"], val_shift_limit=0, p=1.0))
    if "blur" in setting and setting["blur"] != 0:
        transforms.append(Blur(blur_limit=(setting["blur"], setting["blur"]*2+1), p=1.0))
    if "brightness" in setting and setting["brightness"] != 0:
        transforms.append(RandomBrightnessContrast(brightness_limit=setting["brightness"], contrast_limit=setting["brightness"], p=1.0))
    
    return Compose(transforms, bbox_params={'format': 'yolo', 'label_fields': ['category_ids']})


# %% [markdown]
# ### Prüfen, wie viele Bilder entstehen

# %%
# Diese Zelle berechnet die Anzahl der zu erzeugenden Bilder basierend auf der konfigurierten Augmentierung

def calculate_augmentation_count(detailed_explanation=True):
    # Ermittle die Anzahl der Originalbilder (nur Bilddateien erfassen)
    original_images = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + \
                      glob.glob(os.path.join(IMAGES_DIR, "*.png")) + \
                        glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
    num_original_images = len(original_images)
    
    if num_original_images == 0:
        print(f"WARNUNG: Keine Bilder im Verzeichnis '{IMAGES_DIR}' gefunden!")
        return
    
    # Ermittle die aktiven Augmentierungen und deren Levels
    active_augs = [k for k, v in AUG_CONFIG.items() if v["active"]]
    levels = [AUG_CONFIG[k]["levels"] for k in active_augs]
    
    # Für jede aktive Augmentierung: Die effektiven Levels anzeigen
    effective_levels = {}
    aug_combinations = []
    
    for aug_name, aug_levels in zip(active_augs, levels):
        # Standardwerte, die keine Änderung bewirken
        no_change_value = None
        if aug_name == "rotate": no_change_value = 0
        elif aug_name == "scale": no_change_value = 1.0
        elif aug_name == "distort": no_change_value = 0
        elif aug_name == "hflip": no_change_value = 0
        elif aug_name == "hsv": no_change_value = 0
        elif aug_name == "blur": no_change_value = 0
        elif aug_name == "brightness": no_change_value = 0
        
        # Zähle effektive Levels (die tatsächlich eine Änderung bewirken)
        effective_count = sum(1 for level in aug_levels if level != no_change_value)
        effective_levels[aug_name] = effective_count
        aug_combinations.append(len(aug_levels))
    
    # Gesamtzahl der Kombinationen berechnen
    total_combinations = 1
    for count in aug_combinations:
        total_combinations *= count
    
    # Gesamtzahl der zu erzeugenden Bilder = (Originalbilder × (alle Kombinationen))
    total_augmented_images = num_original_images * (total_combinations - 1)
    
    # Ausgabe
    print(f"Anzahl der Originalbilder: {num_original_images}")
    print(f"Aktive Augmentierungen:")
    
    if detailed_explanation:
        total_product = 1  # Für die schrittweise Erklärung
        for aug_name, aug_levels in zip(active_augs, levels):
            level_count = len(aug_levels)
            level_str = ", ".join(str(level) for level in aug_levels)
            print(f"  - {aug_name}: {level_count} Level ({level_str})")
            total_product *= level_count
    else:
        for aug_name, aug_levels in zip(active_augs, levels):
            level_count = len(aug_levels)
            level_str = ", ".join(str(level) for level in aug_levels)
            print(f"  - {aug_name}: {level_count} Level ({level_str})")
    
    print(f"\nBerechnung der Kombinationen:")
    print(f"  Total = {' × '.join(str(len(levels)) for levels in levels)} = {total_combinations}")
    print(f"  Davon 1 Kombination ohne Änderungen (Original)")
    print(f"  Effektive Kombinationen mit Augmentierung: {total_combinations - 1}")
    
    print(f"\nZu erzeugende augmentierte Bilder: {num_original_images} × {total_combinations - 1} = {total_augmented_images}")
    print(f"Gesamtzahl aller Bilder nach Augmentierung: {num_original_images + total_augmented_images}")
    print("Diese Zahl stimmt nur, wenn jede Bilddatei auch eine Labeldatei hat.")
    
    # Warnung bei hoher Anzahl
    if total_augmented_images > 5000:
        print("\n⚠️ WARNUNG: Sehr hohe Anzahl an Bildern! ⚠️")
        print("Überlege, ob du die Anzahl der Augmentierungen reduzieren möchtest.")
        
        # Vorschlag für Reduzierung
        if len(active_augs) > 2:
            example_reduced = 1
            for i, count in enumerate(aug_combinations[:2]):
                example_reduced *= count
            example_reduced -= 1
            example_reduced *= num_original_images
            print(f"\nBeispiel: Durch Verwendung von nur {active_augs[0]} und {active_augs[1]} ")
            print(f"würden nur {example_reduced} augmentierte Bilder erzeugt.")
    
    # Schätzung der Dateigröße
    if num_original_images > 0:
        try:
            avg_size = sum(os.path.getsize(img) for img in original_images) / num_original_images
            total_size_mb = (num_original_images + total_augmented_images) * avg_size / (1024*1024)
            print(f"\nGeschätzte Gesamtgröße aller Bilder: {total_size_mb:.2f} MB")
            
            if total_size_mb > 1000:
                print(f"  ≈ {total_size_mb/1000:.2f} GB")
        except:
            pass

# Führe die Berechnung aus
calculate_augmentation_count(detailed_explanation=True)

# %% [markdown]
# ---
# 
# ### Funktion zur Augmentierung und Speicherung der neuen Bilder

# %%
def parse_yolo_label(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        split = line.strip().split()
        labels.append([int(split[0])] + [float(x) for x in split[1:]])
    return labels

def save_yolo_label(label_path, labels):
    with open(label_path, "w") as f:
        for l in labels:
            lstr = [str(l[0])] + [f"{x:.6f}" for x in l[1:]]
            f.write(" ".join(lstr) + "\n")

active_augs = [k for k, v in AUG_CONFIG.items() if v["active"]]
levels = [AUG_CONFIG[k]["levels"] for k in active_augs]
combinations = list(product(*levels))

def get_aug(aug_setting):
    transforms = []
    setting = dict(zip(active_augs, aug_setting))
    if "rotate" in setting and setting["rotate"] != 0:
        transforms.append(Rotate(limit=(setting["rotate"], setting["rotate"]), p=1.0, border_mode=cv2.BORDER_REFLECT_101))
    if "scale" in setting and setting["scale"] != 1.0:
        transforms.append(RandomScale(scale_limit=(setting["scale"]-1, setting["scale"]-1), p=1.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101))
    if "distort" in setting and setting["distort"] != 0:
        transforms.append(OpticalDistortion(distort_limit=setting["distort"], p=1.0, border_mode=cv2.BORDER_REFLECT_101))
    if "hflip" in setting:
        if setting["hflip"] == 1:
            transforms.append(HorizontalFlip(p=1.0))
        elif setting["hflip"] == 2:
            transforms.append(HorizontalFlip(p=0.5))
    if "hsv" in setting and setting["hsv"] != 0:
        transforms.append(HueSaturationValue(hue_shift_limit=setting["hsv"], sat_shift_limit=setting["hsv"], val_shift_limit=0, p=1.0))
    if "blur" in setting and setting["blur"] != 0:
        transforms.append(Blur(blur_limit=(setting["blur"], setting["blur"]*2+1), p=1.0))
    if "brightness" in setting and setting["brightness"] != 0:
        transforms.append(RandomBrightnessContrast(brightness_limit=setting["brightness"], contrast_limit=setting["brightness"], p=1.0))
    return Compose(transforms, bbox_params={'format': 'yolo', 'label_fields': ['category_ids']})

def calculate_augmentation_count(detailed_explanation=True):
    original_images = glob(os.path.join(IMAGES_DIR, "*.jpg")) + \
                      glob(os.path.join(IMAGES_DIR, "*.png")) + \
                      glob(os.path.join(IMAGES_DIR, "*.jpeg"))
    num_original_images = len(original_images)
    if num_original_images == 0:
        print(f"WARNUNG: Keine Bilder im Verzeichnis '{IMAGES_DIR}' gefunden!")
        return
    active_augs = [k for k, v in AUG_CONFIG.items() if v["active"]]
    levels = [AUG_CONFIG[k]["levels"] for k in active_augs]
    aug_combinations = [len(lev) for lev in levels]
    total_combinations = 1
    for count in aug_combinations:
        total_combinations *= count
    total_augmented_images = num_original_images * (total_combinations - 1)
    print(f"Anzahl der Originalbilder: {num_original_images}")
    print(f"Aktive Augmentierungen:")
    if detailed_explanation:
        for aug_name, aug_levels in zip(active_augs, levels):
            print(f"  - {aug_name}: {len(aug_levels)} Level ({', '.join(map(str, aug_levels))})")
    print(f"\nBerechnung der Kombinationen:")
    print(f"  Total = {' × '.join(str(len(levels)) for levels in levels)} = {total_combinations}")
    print(f"  Davon 1 Kombination ohne Änderungen (Original)")
    print(f"  Effektive Kombinationen mit Augmentierung: {total_combinations - 1}")
    print(f"\nZu erzeugende augmentierte Bilder: {num_original_images} × {total_combinations - 1} = {total_augmented_images}")
    print(f"Gesamtzahl aller Bilder nach Augmentierung: {num_original_images + total_augmented_images}")
    if total_augmented_images > 5000:
        print("\n⚠️ WARNUNG: Sehr hohe Anzahl an Bildern! ⚠️")

def augment_images():
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        img_path = os.path.join(IMAGES_DIR, img_file)
        label_path = os.path.join(LABELS_DIR, base_name + '.txt')
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_file}, skipping")
            continue
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = parse_yolo_label(label_path)
        bboxes = [l[1:] for l in labels]
        class_ids = [l[0] for l in labels]
        for i, aug_setting in enumerate(combinations):
            aug = get_aug(aug_setting)
            suffix = "_aug_" + "_".join([f"{k}{v}" for k, v in zip(active_augs, aug_setting)])
            try:
                augmented = aug(image=image, bboxes=bboxes, category_ids=class_ids)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_ids = augmented['category_ids']
                aug_img_path = os.path.join(OUT_IMAGES_DIR, f"{base_name}{suffix}{ext}")
                cv2.imwrite(aug_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                aug_labels = [[cls_id] + list(box) for cls_id, box in zip(aug_class_ids, aug_bboxes)]
                aug_label_path = os.path.join(OUT_LABELS_DIR, f"{base_name}{suffix}.txt")
                save_yolo_label(aug_label_path, aug_labels)
            except Exception as e:
                print(f"Error augmenting {img_file} with setting {aug_setting}: {str(e)}")

def main():
    print(f"Starting augmentation with {len(combinations)} configurations...")
    calculate_augmentation_count(detailed_explanation=True)
    augment_images()
    print(f"Augmentation complete. Files saved to {OUT_IMAGES_DIR} and {OUT_LABELS_DIR}")


# %% [markdown]
# ---
# 
# # Ausführen: Start der Bilder-Generierung
# 
# **Rechenintensiv!!!**

# %%
if __name__ == "__main__":
    main()



