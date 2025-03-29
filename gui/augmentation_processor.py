"""Process execution for the augmentation application."""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QApplication
from PyQt6.QtGui import QImage, QPixmap
from utils.augmentation_utils import augment_image_with_boxes
from itertools import product

logger = logging.getLogger(__name__)

def calculate_augmentation_count(app):
    """Calculate the expected number of output images accurately."""
    if not app.source_path:
        return 0, 0
            
    image_files = list(Path(app.source_path).rglob("*.jpg")) + \
                 list(Path(app.source_path).rglob("*.png"))
    if not image_files:
        return 0, 0

    # Get selected methods
    selected_methods = []
    for method in app.methods:
        method_key = get_method_key(method)
        if method_key in app.method_levels:
            checkbox, _, _ = app.method_levels[method_key]
            if checkbox.isChecked():
                selected_methods.append(method)

    if not selected_methods and not app.horizontal_flip.isChecked() and not app.vertical_flip.isChecked():
        return len(image_files), len(image_files)

    # For each method, we have 3 possibilities: no augmentation (0), level1 (1), level2 (2)
    # Start with 1 combination (original image without augmentation)
    base_combinations = 3 ** len(selected_methods)
    
    # Apply flip factors if enabled (50% of images will have each flip)
    # So we need to multiply by 1.5 for each enabled flip
    flip_factor = 1.0
    if app.horizontal_flip.isChecked():
        flip_factor *= 1.5
    if app.vertical_flip.isChecked():
        flip_factor *= 1.5
    
    # Calculate total combinations
    total_combinations = int(base_combinations * flip_factor)
    
    # Since original image (no augmentation) is included in the combinations
    # we need to subtract 1 if we don't want to include it in the augmentation count
    if total_combinations > 1:
        total_combinations -= 1  # Subtract original image
    
    total_augmentations = len(image_files) * total_combinations + len(image_files)
    return len(image_files), total_augmentations

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

def start_augmentation_process(app):
    """Start the augmentation process."""
    try:
        # Initial validation
        if not app.source_path or not app.dest_path:
            QMessageBox.warning(app, "Fehler", 
                              "Bitte wählen Sie Quell- und Zielverzeichnis aus.")
            return

        # Get selected methods
        selected_methods = []
        for method in app.methods:
            method_key = get_method_key(method)
            if method_key in app.method_levels:
                checkbox, level_spinbox1, level_spinbox2 = app.method_levels[method_key]
                if checkbox.isChecked():
                    level1 = level_spinbox1.value()
                    level2 = level_spinbox2.value()
                    if level1 >= level2:
                        QMessageBox.warning(app, "Fehler", 
                                          f"Für {method} muss Stufe 1 kleiner als Stufe 2 sein.")
                        return
                    selected_methods.append((method_key, level1, level2))

        if not selected_methods and not app.horizontal_flip.isChecked() and not app.vertical_flip.isChecked():
            QMessageBox.warning(app, "Fehler", 
                              "Bitte wählen Sie mindestens eine Augmentierungsmethode aus.")
            return

        # Find all images and labels
        image_files = list(Path(app.source_path).rglob("*.jpg")) + \
                     list(Path(app.source_path).rglob("*.png"))
        label_files = {file.stem: file for file in Path(app.source_path).rglob("*.txt")}

        if not image_files:
            QMessageBox.warning(app, "Fehler", 
                              "Keine Bilder im Quellverzeichnis gefunden.")
            return

        # Use the same calculation method for UI and popup
        original_count, total_count = calculate_augmentation_count(app)

        # Show expected augmentation count with the exact same calculation
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Augmentierung starten")
        msg.setText(
            f"Es werden {original_count} Bilder mit verschiedenen "
            f"Augmentierungen verarbeitet.\n\n"
            f"Geschätzte Anzahl resultierender Bilder: {total_count}\n"
            f"(Die tatsächliche Anzahl kann geringer sein, wenn Bilder die "
            f"Validierungskriterien nicht erfüllen)"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if msg.exec() == QMessageBox.StandardButton.Cancel:
            return

        # Switch to processing view
        app.stack.setCurrentIndex(0)

        # Reset progress
        app.progress_bar.setValue(0)
        app.progress_bar.setMaximum(len(image_files))
        
        valid_augmentations = 0
        invalid_augmentations = 0

        # Create total combinations for methods
        # Each method can have: no augmentation (0), level1 (1), level2 (2)
        method_combinations = list(product([0, 1, 2], repeat=len(selected_methods)))

        # Perform augmentation
        for i, image_file in enumerate(image_files):
            # Update progress for each source image
            app.progress_bar.setValue(i + 1)
            QApplication.processEvents()

            image = cv2.imread(str(image_file))
            label_file = label_files.get(image_file.stem)
            boxes = []

            if label_file:
                with open(label_file, 'r') as f:
                    try:
                        boxes = [list(map(float, line.strip().split())) for line in f]
                    except ValueError as e:
                        logger.error(f"Fehler beim Parsen der Labels in {label_file}: {e}")
                        continue

            # Save original image to destination if needed
            # (commented out as we typically don't want to copy originals)
            # shutil.copy2(str(image_file), os.path.join(app.dest_path, image_file.name))
            
            # Generate all combinations
            for combination in method_combinations:
                # Skip the "no augmentation" case (all zeros) to avoid copying original image
                if all(level == 0 for level in combination):
                    continue
                    
                augmented_image = image.copy()
                augmented_boxes = boxes.copy()
                valid_augmentation = True
                augmented = False  # Track if any augmentation was applied
                output_suffix = []

                # Apply method augmentations based on combination
                for method_idx, level in enumerate(combination):
                    if level == 0:
                        continue  # Skip methods with level 0 (no augmentation)
                        
                    method, level1, level2 = selected_methods[method_idx]
                    if level == 1:
                        augmented = True
                        # For level 1, use a random value between 0 and level1
                        augmented_image, augmented_boxes = augment_image_with_boxes(
                            augmented_image, augmented_boxes, method, 0, level1,
                            min_visibility=app.settings.get('min_visibility', 0.3),
                            min_size=app.settings.get('min_size', 20))
                        output_suffix.append(f"{method}_L1")
                    elif level == 2:
                        augmented = True
                        # For level 2, use a random value between level1 and level2
                        augmented_image, augmented_boxes = augment_image_with_boxes(
                            augmented_image, augmented_boxes, method, level1, level2,
                            min_visibility=app.settings.get('min_visibility', 0.3),
                            min_size=app.settings.get('min_size', 20))
                        output_suffix.append(f"{method}_L2")

                # Apply flips if enabled (50% chance each)
                if app.horizontal_flip.isChecked() and np.random.random() < 0.5:
                    augmented = True
                    augmented_image, augmented_boxes = augment_image_with_boxes(
                        augmented_image, augmented_boxes, "HorizontalFlip", 0, 0,
                        min_visibility=app.settings.get('min_visibility', 0.3),
                        min_size=app.settings.get('min_size', 20))
                    output_suffix.append("HFlip")
                    
                if app.vertical_flip.isChecked() and np.random.random() < 0.5:
                    augmented = True
                    augmented_image, augmented_boxes = augment_image_with_boxes(
                        augmented_image, augmented_boxes, "VerticalFlip", 0, 0,
                        min_visibility=app.settings.get('min_visibility', 0.3),
                        min_size=app.settings.get('min_size', 20))
                    output_suffix.append("VFlip")
                
                # Skip if no valid boxes remain after augmentation or no augmentation applied
                if (not augmented_boxes and boxes) or not augmented:  
                    invalid_augmentations += 1
                    continue

                # Show preview of current augmentation
                if app.preview_checkbox.isChecked() and augmented_image is not None:
                    # Convert to RGB for Qt
                    preview_img = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    
                    # Get preview label size
                    preview_width = min(800, app.preview_label.width())  # Limit max width
                    preview_height = min(600, app.preview_label.height()) # Limit max height
                    
                    # Calculate aspect ratio preserving scale
                    img_h, img_w = preview_img.shape[:2]
                    scale = min(preview_width/img_w, preview_height/img_h)
                    
                    # Calculate new size
                    new_w = int(img_w * scale)
                    new_h = int(img_h * scale)
                    
                    # Resize image
                    preview_img = cv2.resize(preview_img, (new_w, new_h), 
                                           interpolation=cv2.INTER_AREA)
                    
                    # Convert to QImage and display
                    qimg = QImage(preview_img.data, new_w, new_h, 
                                new_w * 3, QImage.Format.Format_RGB888)
                    app.preview_label.setPixmap(QPixmap.fromImage(qimg))
                    QApplication.processEvents()

                # Save augmented image and labels
                output_image_path = Path(app.dest_path) / \
                    f"{image_file.stem}_{'_'.join(output_suffix)}.jpg"
                output_label_path = Path(app.dest_path) / \
                    f"{image_file.stem}_{'_'.join(output_suffix)}.txt"

                cv2.imwrite(str(output_image_path), augmented_image)
                
                with open(output_label_path, 'w') as f:
                    for box in augmented_boxes:
                        f.write(' '.join(map(str, box)) + '\n')

                valid_augmentations += 1

        app.progress_bar.setValue(100)

        # Show final results
        QMessageBox.information(
            app,
            "Augmentierung abgeschlossen",
            f"Augmentierung erfolgreich abgeschlossen!\n\n"
            f"Verarbeitet: {len(image_files)} Bilder\n"
            f"Gültige Augmentierungen: {valid_augmentations}\n"
            f"Ungültige Augmentierungen: {invalid_augmentations}"
        )
        
        # Return to preview mode if it was on
        if app.preview_checkbox.isChecked():
            app.stack.setCurrentIndex(1)
        
    except Exception as e:
        logger.critical(f"Unbehandelter Fehler: {str(e)}", exc_info=True)
        QMessageBox.critical(
            app, "Fehler",
            f"Ein unerwarteter Fehler ist aufgetreten: {str(e)}"
        )