"""Utility functions for image augmentation."""

import cv2
import numpy as np
from albumentations import (
    ShiftScaleRotate, RandomBrightnessContrast, GaussianBlur,
    BboxParams, Compose, PadIfNeeded
)
import logging

def augment_image_with_boxes(image, boxes, method, level1, level2):
    """
    Augments an image and its bounding boxes based on the selected method and levels.
    
    Args:
        image (numpy.ndarray): The input image
        boxes (list): List of bounding boxes in YOLO format
        method (str): The augmentation method (Shift, Rotate, Zoom, Brightness, Blur)
        level1 (float): Lower bound of augmentation intensity
        level2 (float): Upper bound of augmentation intensity
    
    Returns:
        tuple: (augmented_image, augmented_boxes)
    """
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
        elif method == "Rotate":
            rotate_limit = np.random.uniform(level1 * 360 / 100, level2 * 360 / 100)
            transform_list.append(ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0,
                rotate_limit=int(rotate_limit),
                border_mode=cv2.BORDER_REFLECT,
                fit_output=True,
                p=1.0
            ))
        elif method == "Zoom":
            scale_limit = np.random.uniform(level1 / 100, level2 / 100)
            transform_list.append(ShiftScaleRotate(
                shift_limit=0,
                scale_limit=scale_limit,
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ))
        elif method == "Brightness":
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
        else:
            logging.warning(f"Unbekannte Augmentierungsmethode: {method}")
            return image, boxes

        transform_list.append(PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_REFLECT,
            p=1.0
        ))

        # Handle empty boxes case (background images)
        if not boxes:
            transform = Compose(transform_list)
            transformed = transform(image=image)
            return transformed['image'], []

        # Handle case with bounding boxes
        albumentations_boxes = convert_boxes_to_albumentations(boxes, image.shape)
        if albumentations_boxes:
            labels = [box[4] for box in albumentations_boxes]
            albumentations_boxes = [box[:4] for box in albumentations_boxes]

            transform = Compose(
                transform_list,
                bbox_params=BboxParams(
                    format='pascal_voc',
                    min_visibility=0.2,
                    label_fields=['labels']
                )
            )

            transformed = transform(
                image=image,
                bboxes=albumentations_boxes,
                labels=labels
            )
            
            augmented_boxes = []
            for box, label in zip(transformed['bboxes'], transformed['labels']):
                x_min, y_min, x_max, y_max = box
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(width, x_max), min(height, y_max)

                box_width = x_max - x_min
                box_height = y_max - y_min
                if box_width * box_height < 0.00001 * width * height:
                    logging.debug(f"Zu kleine Box ignoriert: {[x_min, y_min, x_max, y_max]}")
                    continue

                x_center = (x_min + x_max) / (2 * width)
                y_center = (y_min + y_max) / (2 * height)
                box_width /= width
                box_height /= height

                augmented_boxes.append([label, x_center, y_center, box_width, box_height])

            return transformed['image'], augmented_boxes
        else:
            transform = Compose(transform_list)
            transformed = transform(image=image)
            return transformed['image'], []

    except Exception as e:
        logging.error(f"Fehler während der Augmentierung: {e}")
        return image, boxes

def convert_boxes_to_albumentations(boxes, image_shape):
    """
    Convert YOLO-format bounding boxes to Albumentations format.

    Args:
        boxes (list): List of bounding boxes in YOLO format
        image_shape (tuple): Shape of the image (height, width, channels)

    Returns:
        list: Converted bounding boxes in Albumentations format
    """
    if not boxes:
        return None
        
    try:
        height, width = image_shape[:2]
        converted_boxes = []

        for box in boxes:
            if len(box) != 5:
                continue

            label = int(box[0])
            x_center, y_center, box_width, box_height = box[1:]

            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            box_width = np.clip(box_width, 0, 1)
            box_height = np.clip(box_height, 0, 1)

            x_min = (x_center - box_width / 2) * width
            y_min = (y_center - box_height / 2) * height
            x_max = (x_center + box_width / 2) * width
            y_max = (y_center + box_height / 2) * height

            if x_min < x_max and y_min < y_max:
                converted_boxes.append([x_min, y_min, x_max, y_max, label])

        return converted_boxes if converted_boxes else None

    except Exception as e:
        logging.error(f"Fehler bei der Konvertierung der Bounding Boxes: {e}")
        return None