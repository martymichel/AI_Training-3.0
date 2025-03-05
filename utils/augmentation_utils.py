"""Utility functions for image augmentation."""

import cv2
import numpy as np
from albumentations import (
    ShiftScaleRotate, RandomBrightnessContrast, GaussianBlur, 
    BboxParams, Compose, PadIfNeeded, CenterCrop
)
import logging
from albumentations.augmentations.geometric.transforms import HorizontalFlip

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
        return False
        
    return visibility >= min_visibility

def augment_image_with_boxes(image, boxes, method, level1, level2, min_visibility=0.3, min_size=20):
    """
    Augments an image and its bounding boxes based on the selected method and levels.
    
    Args:
        image (numpy.ndarray): The input image
        boxes (list): List of bounding boxes in YOLO format
        method (str): The augmentation method (Shift, Rotate, Zoom, Brightness, Blur)
        level1 (float): Lower bound of augmentation intensity
        level2 (float): Upper bound of augmentation intensity
        min_visibility (float): Minimum required visibility of box after augmentation
        min_size (int): Minimum size of box in pixels
    
    Returns:
        tuple: (augmented_image, augmented_boxes)
    """
    if not boxes:
        return image, []

    try:
        height, width = image.shape[:2]
        transform_list = []

        # Configure transformation based on method
        if method == "Shift":
            shift_x = np.random.uniform(-level2 / 100, level2 / 100)
            shift_y = np.random.uniform(-level2 / 100, level2 / 100) 
            transform_list.append(ShiftScaleRotate(
                shift_limit=(shift_x, shift_y),
                scale_limit=0,
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ))
        elif method == "Rotate":  # Limit rotation to avoid extreme angles
            rotate_limit = np.random.uniform(level1 * 360 / 100, level2 * 360 / 100)
            transform_list.append(ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0,
                rotate_limit=int(rotate_limit),
                border_mode=cv2.BORDER_REFLECT,
                fit_output=True,
                p=1.0
            ))
        elif method == "Zoom":  # Ensure zoom doesn't cut off too much
            # Convert percentage to scale factor (e.g., 120% -> 1.2, 80% -> 0.8)
            min_scale = 1.0 + (level1 / 100.0)  # e.g., 110% -> 1.1
            max_scale = 1.0 + (level2 / 100.0)  # e.g., 150% -> 1.5
            scale = np.random.uniform(min_scale, max_scale)
            transform_list.append(ShiftScaleRotate(
                shift_limit=0,
                scale_limit=(scale - 1.0, scale - 1.0),  # Same scale for both dimensions
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ))
        elif method == "Brightness":  # Non-geometric transform
            brightness_limit = np.random.uniform(-level2 / 100, level2 / 100)
            transform_list.append(RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=0,
                p=1.0
            ))
        elif method == "Blur":
            blur_limit = int(np.random.uniform(level1 * 5 / 100, level2 * 5 / 100))
            transform_list.append(GaussianBlur(
                blur_limit=(blur_limit, blur_limit),
                p=1.0
            ))
        elif method == "HorizontalFlip":
            transform_list.append(HorizontalFlip(p=1.0))
        else:
            logging.warning(f"Unbekannte Augmentierungsmethode: {method}")
            return image, boxes.copy()

        transform_list.append(PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_REFLECT,
            p=1.0
        ))

        # Convert and validate boxes
        albumentations_boxes = convert_boxes_to_albumentations(boxes, image.shape)
        if albumentations_boxes:
            labels = [box[4] for box in albumentations_boxes]
            albumentations_boxes = [box[:4] for box in albumentations_boxes]

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
                bboxes=albumentations_boxes,
                labels=labels
            )
            
            # Process and validate transformed boxes
            augmented_boxes = []
            for box, label in zip(transformed['bboxes'], transformed['labels']):
                x_min, y_min, x_max, y_max = box
                
                # Validate transformed box
                if not validate_box([x_min, y_min, x_max, y_max], 
                                  transformed['image'].shape,
                                  min_visibility=min_visibility,
                                  min_size=min_size):
                    logger.debug(
                        f"Box rejected after {method}: "
                        f"class={label}, coords=[{x_min:.1f}, {y_min:.1f}, "
                        f"{x_max:.1f}, {y_max:.1f}]"
                    )
                    continue

                # Convert to YOLO format
                x_center = (x_min + x_max) / (2 * width)
                y_center = (y_min + y_max) / (2 * height)
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height

                augmented_boxes.append([label, x_center, y_center, box_width, box_height])

            if not augmented_boxes:
                logger.warning(f"All boxes were rejected after {method} augmentation")
                return image, boxes.copy()

            return transformed['image'], augmented_boxes
        else:
            transformed = transform(image=image)
            return transformed['image'], []

    except Exception as e:
        logging.error(f"Fehler während der Augmentierung: {e}")
        return image, boxes.copy()

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
        return None
        
    try:
        height, width = image_shape[:2]
        converted_boxes = []

        for box in boxes:
            if len(box) != 5:
                continue  # Skip invalid boxes

            label = int(box[0])
            x_center, y_center, box_width, box_height = box[1:]

            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            box_width = np.clip(box_width, 0, 1)
            box_height = np.clip(box_height, 0, 1)
            
            # Skip boxes that are too small
            if box_width * width < 10 or box_height * height < 10:
                continue

            x_min = (x_center - box_width / 2) * width
            y_min = (y_center - box_height / 2) * height
            x_max = (x_center + box_width / 2) * width
            y_max = (y_center + box_height / 2) * height

            if x_min < x_max and y_min < y_max:
                converted_boxes.append([x_min, y_min, x_max, y_max, label])

        return converted_boxes if converted_boxes else None

    except Exception as e:
        logger.error(f"Fehler bei der Konvertierung der Bounding Boxes: {e}")
        return None