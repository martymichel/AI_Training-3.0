"""Process execution for the augmentation application."""

import os
import cv2
import numpy as np
import logging
import shutil
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QApplication
from PyQt6.QtGui import QImage, QPixmap
from itertools import product

from utils.augmentation_utils import augment_image_with_boxes, detect_annotation_format
from project_manager import WorkflowStep


logger = logging.getLogger(__name__)

def analyze_dataset_format(app):
    """Analyze dataset to determine annotation format.
    
    Returns:
        tuple: (format, bbox_count, polygon_count, total_files)
    """
    if not app.source_path:
        return 'unknown', 0, 0, 0
    
    image_files = list(Path(app.source_path).rglob("*.jpg")) + \
                 list(Path(app.source_path).rglob("*.png"))
    
    if not image_files:
        return 'unknown', 0, 0, 0
    
    bbox_count = 0
    polygon_count = 0
    total_files = len(image_files)
    
    # Analyze first 20 files for efficiency
    sample_files = image_files[:20]
    
    for image_file in sample_files:
        label_file = image_file.with_suffix('.txt')
        if not label_file.exists():
            continue
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) == 5:
                    bbox_count += 1
                elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                    polygon_count += 1
        
        except Exception as e:
            logger.warning(f"Error analyzing {label_file}: {e}")
            continue
    
    if bbox_count > 0 and polygon_count > 0:
        return 'mixed', bbox_count, polygon_count, total_files
    elif polygon_count > 0:
        return 'polygon', bbox_count, polygon_count, total_files
    elif bbox_count > 0:
        return 'bbox', bbox_count, polygon_count, total_files
    else:
        return 'unknown', bbox_count, polygon_count, total_files
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

    # If no methods are active, no augmentation
    if not selected_methods:
        return len(image_files), len(image_files)

    # For each active method, we get 3 variations (original, level1, level2)
    # So total combinations is 3^n where n is number of active methods
    total_combinations = 3 ** len(selected_methods)
    
    # The flips don't affect the count as they're applied to a percentage of images
    # They don't create additional images
    
    # Calculate total output images
    total_augmentations = len(image_files) * total_combinations
    
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

        # Analyze dataset format
        dataset_format, bbox_count, polygon_count, total_files = analyze_dataset_format(app)
        
        if dataset_format == 'mixed':
            reply = QMessageBox.question(
                app, "Mixed Dataset erkannt",
                f"Ihr Dataset enthält sowohl Bounding Boxes ({bbox_count}) als auch Polygone ({polygon_count}).\n\n"
                f"Für konsistente Augmentation sollten alle Annotationen das gleiche Format haben.\n\n"
                f"Möchten Sie trotzdem fortfahren?\n"
                f"(Polygone werden bevorzugt behandelt, Boxes werden ignoriert)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        elif dataset_format == 'polygon':
            QMessageBox.information(
                app, "Polygon Dataset",
                f"Polygon-Dataset erkannt!\n\n"
                f"Gefunden: {polygon_count} Polygone\n"
                f"Augmentation unterstützt alle Transformationen für Polygone."
            )
        elif dataset_format == 'bbox':
            QMessageBox.information(
                app, "Bounding Box Dataset", 
                f"Bounding Box Dataset erkannt!\n\n"
                f"Gefunden: {bbox_count} Bounding Boxes\n"
                f"Augmentation verwendet optimierte Box-Transformationen."
            )
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

        # Perform augmentation for each image
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

            # Always include original image in the output
            # (original is the first of the 3 variations for each method)
            original_image_path = Path(app.dest_path) / f"{image_file.stem}_Original.jpg"
            original_label_path = Path(app.dest_path) / f"{image_file.stem}_Original.txt"
            
            # Copy the original image and label
            cv2.imwrite(str(original_image_path), image)
            if label_file:
                shutil.copy2(str(label_file), str(original_label_path))
            valid_augmentations += 1

            # Create total combinations for methods
            # Each method can have: no augmentation (0), level1 (1), level2 (2)
            method_combinations = list(product([0, 1, 2], repeat=len(selected_methods)))

            # Generate augmented variations
            for combination in method_combinations:
                # Skip the case of all levels=0 as we already included the original image
                if all(level == 0 for level in combination):
                    continue
                    
                augmented_image = image.copy()
                augmented_boxes = boxes.copy() if boxes else []
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
                    for annotation in augmented_boxes:
                        # Handle both bounding boxes and polygons
                        if len(annotation) == 5:
                            # Bounding box format
                            f.write(' '.join(map(str, annotation)) + '\n')
                        elif len(annotation) >= 7:
                            # Polygon format - ensure proper precision
                            class_id = int(annotation[0])
                            coords = [f"{coord:.6f}" for coord in annotation[1:]]
                            f.write(f"{class_id} {' '.join(coords)}\n")
                        else:
                            logger.warning(f"Invalid annotation format: {annotation}")

                valid_augmentations += 1

        app.progress_bar.setValue(100)

        # Show final results
        format_info = f"\nDataset-Format: {dataset_format.upper()}"
        if dataset_format == 'mixed':
            format_info += f"\n⚠️ Mixed Format: {bbox_count} Boxes, {polygon_count} Polygone"
        elif dataset_format == 'polygon':
            format_info += f"\n✅ Polygone: {polygon_count} erkannt"
        elif dataset_format == 'bbox':
            format_info += f"\n✅ Bounding Boxes: {bbox_count} erkannt"
        
        QMessageBox.information(
            app,
            "Augmentierung abgeschlossen",
            f"Augmentierung erfolgreich abgeschlossen!\n\n"
            f"Verarbeitet: {len(image_files)} Bilder\n"
            f"Gültige Augmentierungen: {valid_augmentations}\n"
            f"Ungültige Augmentierungen: {invalid_augmentations}"
            f"{format_info}"
        )
        
        # Return to preview mode if it was on
        if app.preview_checkbox.isChecked():
            app.stack.setCurrentIndex(1)

        if hasattr(app, 'project_manager') and app.project_manager:
            app.project_manager.mark_step_completed(WorkflowStep.AUGMENTATION)
        
    except Exception as e:
        logger.critical(f"Unbehandelter Fehler: {str(e)}", exc_info=True)
        QMessageBox.critical(
            app, "Fehler",
            f"Ein unerwarteter Fehler ist aufgetreten: {str(e)}"
        )