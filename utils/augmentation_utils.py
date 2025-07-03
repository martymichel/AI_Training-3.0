"""Utility functions for image augmentation."""

import cv2
import numpy as np
from PIL import Image
from albumentations import (
    ShiftScaleRotate, RandomBrightnessContrast, GaussianBlur, 
    BboxParams, Compose, PadIfNeeded, CenterCrop
)
# Albumentations 2.x reorganized some modules. Importing the flip transforms
# directly from ``albumentations`` works across versions and avoids import
# errors on newer releases.
from albumentations import HorizontalFlip, VerticalFlip
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def is_valid_image(image_path):
    """
    Validate if the image file is valid and can be opened.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if image is valid
    """
    try:
        # Try to open with PIL first
        with Image.open(image_path) as img:
            img.verify()
        
        # If verification passes, try to load with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"OpenCV could not load image: {image_path}")
            return False
            
        if len(img.shape) < 2:
            logger.warning(f"Invalid image dimensions: {image_path}")
            return False
            
        return True
    except Exception as e:
        logger.warning(f"Invalid or corrupted image {image_path}: {str(e)}")
        return False

def validate_box(box, image_shape, min_visibility=0.3, min_size=20):
    """
    Validate if a bounding box is valid after augmentation.
    
    Args:
        box (list): Box coordinates [x_min, y_min, x_max, y_max]
        image_shape (tuple): Image shape (height, width)
        min_visibility (float): Minimum required visibility (0-1)
        min_size (int): Minimum size in pixels
        
    Returns:
        bool: True if box is valid
    """
    height, width = image_shape[:2]
    x_min, y_min, x_max, y_max = box
    
    # Check if box is completely outside image
    if x_max < 0 or x_min > width or y_max < 0 or y_min > height:
        return False
        
    # Clip box to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    # Calculate visibility ratio
    orig_area = (box[2] - box[0]) * (box[3] - box[1])
    new_area = (x_max - x_min) * (y_max - y_min)
    if orig_area == 0:
        return False
    visibility = new_area / orig_area
    
    # Check minimum size
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width < 1 or box_height < 1:  # Mindestens 1 Pixel
        return False
        
    return visibility >= min_visibility

logger = logging.getLogger(__name__)

def robust_convert_boxes_to_albumentations(boxes, image_shape):
    """
    Konvertiert YOLO-Format Boxen in das Albumentations-Format (Pascal VOC).
    Pro Koordinate wird unterschieden, ob sie normalisiert (<1) oder absolut (>=1) ist.
    """
    if not boxes or not isinstance(boxes, (list, np.ndarray)):
        return []
    height, width = image_shape[:2]
    converted_boxes = []
    for box in boxes:
        if len(box) != 5:
            continue
        label = int(box[0])
        # Pro Koordinate: wenn kleiner als 1, dann als normalisiert interpretieren
        x_center = box[1] * width if box[1] < 1 else box[1]
        y_center = box[2] * height if box[2] < 1 else box[2]
        box_width = box[3] * width if box[3] < 1 else box[3]
        box_height = box[4] * height if box[4] < 1 else box[4]
        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2
        if x_min < x_max and y_min < y_max:
            converted_boxes.append([x_min, y_min, x_max, y_max, label])
    return converted_boxes

def fallback_clip_boxes(boxes, image_shape):
    """
    Clippt die ursprünglichen Boxen an die Bildgrenzen und konvertiert sie in
    normalisiertes YOLO-Format. Wird als Fallback verwendet, falls keine Boxen
    nach der Transformation übrig bleiben.
    """
    height, width = image_shape[:2]
    clipped_boxes = []
    for box in boxes:
        if len(box) != 5:
            continue
        label = int(box[0])
        x_center = box[1] * width if box[1] < 1 else box[1]
        y_center = box[2] * height if box[2] < 1 else box[2]
        box_width = box[3] * width if box[3] < 1 else box[3]
        box_height = box[4] * height if box[4] < 1 else box[4]
        x_min = max(0, x_center - box_width / 2)
        y_min = max(0, y_center - box_height / 2)
        x_max = min(width, x_center + box_width / 2)
        y_max = min(height, y_center + box_height / 2)
        new_x_center = (x_min + x_max) / 2 / width
        new_y_center = (y_min + y_max) / 2 / height
        new_w = (x_max - x_min) / width
        new_h = (y_max - y_min) / height
        clipped_boxes.append([label, new_x_center, new_y_center, new_w, new_h])
    return clipped_boxes

def augment_image_with_boxes(image, boxes, method, level1, level2, min_visibility=0.3, min_size=20):
    """
    Fuehrt eine robuste Augmentierung eines Bildes mit zugehoerigen Bounding Boxes durch.
    Dabei werden alle transformierten Boxen (selbst bei inkonsequenten Eingabedaten)
    übernommen – statt das Bild zu verwerfen.
    """
    if image is None:
        logger.error("Ungultiges Bild")
        return None, []
    try:
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            logger.error("Ungultige Bilddimensionen")
            return None, []

        transform_list = []
        boxes = [] if boxes is None else boxes

        if method == "Shift":
            # Zufällige Verschiebung zwischen Level 1 und Level 2
            shift_limit = np.random.uniform(level1 / 100, level2 / 100)
            # Zufällige Richtung (positiv/negativ) für beide Achsen
            shift_x = shift_limit * (1 if np.random.random() < 0.5 else -1)
            shift_y = shift_limit * (1 if np.random.random() < 0.5 else -1)
            transform_list.append(ShiftScaleRotate(
                shift_limit=(shift_x, shift_y),
                scale_limit=0,
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ))
        elif method == "Rotate":
            rotate_limit = np.random.uniform(level1 * 360 / 100, level2 * 360 / 100)
            transform_list.append(ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0,
                rotate_limit=int(rotate_limit),
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ))
        elif method == "Zoom":
            min_scale = 1.0 + (level1 / 100.0)
            max_scale = 1.0 + (level2 / 100.0)
            scale = np.random.uniform(min_scale, max_scale)
            transform_list.append(ShiftScaleRotate(
                shift_limit=0,
                scale_limit=(scale - 1.0, scale - 1.0),
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ))
        elif method == "Brightness":
            # Zufällige Helligkeitsänderung zwischen Level 1 und Level 2
            brightness_limit = np.random.uniform(level1 / 100, level2 / 100)
            # Zufällige Richtung (heller/dunkler)
            if np.random.random() < 0.5:
                brightness_limit = -brightness_limit
            transform_list.append(RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=0,
                p=1.0
            ))
        elif method == "Blur":
            # Zufälliger Blur-Radius zwischen Level 1 und Level 2
            blur_limit = int(np.random.uniform(level1 / 100, level2 / 100))
            transform_list.append(GaussianBlur(
                blur_limit=(blur_limit, blur_limit),
                p=1.0
            ))
        elif method == "HorizontalFlip":
            transform_list.append(HorizontalFlip(p=1.0))
        elif method == "VerticalFlip":
            transform_list.append(VerticalFlip(p=1.0))
        else:
            logger.warning(f"Unbekannte Augmentierungsmethode: {method}")
            return image, boxes

        transform_list.append(PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_REFLECT,
            p=1.0
        ))

        # Konvertiere alle Boxen, auch wenn sie gemischte Koordinaten enthalten
        albumentations_boxes = robust_convert_boxes_to_albumentations(boxes, image.shape)

        if albumentations_boxes:
            labels = [box[4] for box in albumentations_boxes]
            albumentations_boxes_coords = [box[:4] for box in albumentations_boxes]
            transform = Compose(
                transform_list,
                bbox_params=BboxParams(
                    format='pascal_voc',
                    min_visibility=min_visibility,
                    label_fields=['labels']
                )
            )
            transformed = transform(
                image=image,
                bboxes=albumentations_boxes_coords,
                labels=labels
            )
            augmented_boxes = []
            for box, label in zip(transformed['bboxes'], transformed['labels']):
                x_min, y_min, x_max, y_max = box
                # Keine Filterung mehr – alle transformierten Boxen werden übernommen
                x_center = (x_min + x_max) / (2 * width)
                y_center = (y_min + y_max) / (2 * height)
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height
                augmented_boxes.append([label, x_center, y_center, box_width, box_height])
        else:
            transform = Compose(transform_list)
            transformed = transform(image=image)
            augmented_boxes = boxes if boxes else []

        # Fallback: Falls aus irgendeinem Grund keine Boxen resultieren, clippe die Originalboxen
        if not augmented_boxes and boxes:
            logger.warning(f"Keine transformierten Boxen nach {method} Augmentierung. Wende Fallback an.")
            augmented_boxes = fallback_clip_boxes(boxes, image.shape)

        return transformed['image'], augmented_boxes

    except Exception as e:
        logger.error(f"Fehler bei der Augmentierung: {e}")
        return None, []


def convert_boxes_to_albumentations(boxes, image_shape):
    """
    Convert YOLO-format bounding boxes to Albumentations format.

    Args:
        boxes (list): List of bounding boxes in YOLO format
        image_shape (tuple): Shape of the image (height, width, channels)

    Returns:
        list: Converted bounding boxes in Albumentations format
    """
    if not boxes or not isinstance(boxes, (list, np.ndarray)):
        return []
        
    try:
        height, width = image_shape[:2]
        converted_boxes = []

        for box in boxes:
            if len(box) != 5:
                continue  # Skip invalid boxes

            label = int(box[0])
            # Normalisiere Koordinaten auf [0, 1]
            x_center = box[1] % 1.0 if box[1] > 1 else box[1]
            y_center = box[2] % 1.0 if box[2] > 1 else box[2]
            box_width = box[3] % 1.0 if box[3] > 1 else box[3]
            box_height = box[4] % 1.0 if box[4] > 1 else box[4]
            
            # Skip boxes that are too small
            if box_width * width < 10 or box_height * height < 10:
                continue

            x_min = (x_center - box_width / 2) * width
            y_min = (y_center - box_height / 2) * height
            x_max = (x_center + box_width / 2) * width
            y_max = (y_center + box_height / 2) * height

            if x_min < x_max and y_min < y_max:
                converted_boxes.append([x_min, y_min, x_max, y_max, label])

        return converted_boxes

    except Exception as e:
        logger.error(f"Fehler bei der Konvertierung der Bounding Boxes: {e}")
        return []