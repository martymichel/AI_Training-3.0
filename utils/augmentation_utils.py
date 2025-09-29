"""Utility functions for image augmentation."""

import cv2
import numpy as np
from PIL import Image
from albumentations import (
    Compose, Rotate, RandomScale, OpticalDistortion, HorizontalFlip, VerticalFlip,
    HueSaturationValue, Blur, RandomBrightnessContrast, ShiftScaleRotate, GaussianBlur
)
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def detect_annotation_format(boxes):
    """Detect if annotations are bounding boxes or polygons.
    
    Args:
        boxes (list): List of annotation boxes
        
    Returns:
        str: 'bbox', 'polygon', or 'mixed'
    """
    if not boxes:
        return 'unknown'
    
    bbox_count = 0
    polygon_count = 0
    
    for box in boxes:
        if len(box) == 5:
            bbox_count += 1
        elif len(box) >= 7 and (len(box) - 1) % 2 == 0:
            polygon_count += 1
    
    if bbox_count > 0 and polygon_count > 0:
        return 'mixed'
    elif polygon_count > 0:
        return 'polygon'
    elif bbox_count > 0:
        return 'bbox'
    else:
        return 'unknown'

def parse_polygon_annotations(boxes):
    """Parse polygon annotations into keypoints for Albumentations.
    
    Args:
        boxes (list): List of polygon annotations [class_id, x1, y1, x2, y2, ...]
        
    Returns:
        tuple: (keypoints_list, class_ids)
    """
    keypoints_list = []
    class_ids = []
    
    for box in boxes:
        if len(box) >= 7 and (len(box) - 1) % 2 == 0:
            class_ids.append(int(box[0]))
            # Extract coordinate pairs
            coords = box[1:]
            keypoints = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = float(coords[i])
                    y = float(coords[i + 1])
                    # Ensure coordinates are normalized (0-1)
                    x = x if x <= 1.0 else x / 1000.0
                    y = y if y <= 1.0 else y / 1000.0
                    keypoints.extend([x, y])
            keypoints_list.append(keypoints)
    
    return keypoints_list, class_ids

def convert_keypoints_to_polygons(keypoints_list, class_ids):
    """Convert augmented keypoints back to polygon format.
    
    Args:
        keypoints_list (list): List of augmented keypoints
        class_ids (list): Corresponding class IDs
        
    Returns:
        list: Polygons in YOLO format [class_id, x1, y1, x2, y2, ...]
    """
    result_polygons = []
    
    for keypoints, class_id in zip(keypoints_list, class_ids):
        if len(keypoints) >= 6:  # At least 3 points (6 coordinates)
            # Filter out invalid keypoints (outside 0-1 range)
            valid_coords = []
            for i in range(0, len(keypoints), 2):
                if i + 1 < len(keypoints):
                    x, y = keypoints[i], keypoints[i + 1]
                    # Keep points that are mostly within bounds (allow small overflow)
                    if -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1:
                        # Clamp to valid range
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        valid_coords.extend([x, y])
            
            # Only keep polygon if we have at least 3 valid points
            if len(valid_coords) >= 6:
                polygon = [int(class_id)] + valid_coords
                result_polygons.append(polygon)
    
    return result_polygons

def augment_image_with_polygons(image, polygons, method, level1, level2, min_visibility=0.3, min_size=20):
    """
    Augment image with polygon annotations using Albumentations keypoints.
    
    Args:
        image: Input image (numpy array)
        polygons: List of polygon annotations [class_id, x1, y1, x2, y2, ...]
        method: Augmentation method
        level1, level2: Augmentation parameters
        min_visibility: Minimum visibility threshold (not used for polygons)
        min_size: Minimum size threshold
        
    Returns:
        tuple: (augmented_image, augmented_polygons)
    """
    if image is None:
        logger.error("Invalid image for polygon augmentation")
        return None, []
    
    try:
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            logger.error("Invalid image dimensions")
            return None, []

        # Parse polygons into keypoints format
        keypoints_list, class_ids = parse_polygon_annotations(polygons)
        
        if not keypoints_list and polygons:
            logger.warning("No valid polygon annotations found")
            return image, polygons

        # Create transform based on method
        transforms = []
        
        if method == "Rotate":
            rotate_angle = np.random.uniform(level1, level2)
            transforms.append(Rotate(
                limit=(rotate_angle, rotate_angle), 
                p=1.0, 
                border_mode=cv2.BORDER_REFLECT_101
            ))
            
        elif method == "Zoom" or method == "Scale":
            scale_percent = np.random.uniform(level1, level2)
            scale_factor = 1.0 + scale_percent / 100.0
            scale_factor = max(0.1, min(scale_factor, 3.0))
            transforms.append(RandomScale(
                scale_limit=(scale_factor - 1, scale_factor - 1),
                p=1.0,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101
            ))
            
        elif method == "HorizontalFlip":
            transforms.append(HorizontalFlip(p=1.0))
            
        elif method == "VerticalFlip":
            transforms.append(VerticalFlip(p=1.0))
            
        elif method == "Brightness":
            brightness_change = np.random.uniform(level1/100, level2/100)
            if np.random.random() < 0.5:
                brightness_change = -brightness_change
            transforms.append(RandomBrightnessContrast(
                brightness_limit=brightness_change,
                contrast_limit=brightness_change,
                p=1.0
            ))
            
        elif method == "Blur":
            blur_limit = int(np.random.uniform(level1, level2))
            if blur_limit > 0:
                transforms.append(Blur(
                    blur_limit=(blur_limit, blur_limit*2+1),
                    p=1.0
                ))
                
        elif method == "HSV":
            hsv_shift = np.random.uniform(level1, level2)
            transforms.append(HueSaturationValue(
                hue_shift_limit=hsv_shift,
                sat_shift_limit=hsv_shift,
                val_shift_limit=0,
                p=1.0
            ))
            
        elif method == "Shift":
            shift_limit = np.random.uniform(level1 / 100, level2 / 100)
            shift_x = shift_limit * (1 if np.random.random() < 0.5 else -1)
            shift_y = shift_limit * (1 if np.random.random() < 0.5 else -1)
            transforms.append(ShiftScaleRotate(
                shift_limit_x=(shift_x, shift_x),
                shift_limit_y=(shift_y, shift_y),
                scale_limit=0,
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ))
            
        elif method == "Distortion":
            distort_limit = np.random.uniform(level1, level2)
            if distort_limit > 0:
                transforms.append(OpticalDistortion(
                    distort_limit=distort_limit,
                    p=1.0,
                    border_mode=cv2.BORDER_REFLECT_101
                ))
        else:
            logger.warning(f"Unknown augmentation method: {method}")
            return image, polygons

        if not transforms:
            return image, polygons

        # Create compose transform with keypoints format for polygons
        transform = Compose(
            transforms, 
            keypoint_params={'format': 'xy', 'remove_invisible': True}
        )

        try:
            # Apply transformation to each polygon separately
            augmented_polygons = []
            
            for keypoints, class_id in zip(keypoints_list, class_ids):
                # Convert keypoints to list of (x, y) tuples
                keypoint_pairs = []
                for i in range(0, len(keypoints), 2):
                    if i + 1 < len(keypoints):
                        keypoint_pairs.append((keypoints[i], keypoints[i + 1]))
                
                # Apply transformation
                augmented = transform(
                    image=image,
                    keypoints=keypoint_pairs
                )
                
                aug_image = augmented['image']
                aug_keypoints = augmented['keypoints']
                
                # Convert back to flat coordinate list
                if aug_keypoints:
                    flat_coords = []
                    for x, y in aug_keypoints:
                        flat_coords.extend([x, y])
                    
                    # Only keep if we have at least 3 points
                    if len(flat_coords) >= 6:
                        polygon = [int(class_id)] + flat_coords
                        augmented_polygons.append(polygon)
            
            return aug_image, augmented_polygons
            
        except Exception as e:
            logger.error(f"Error during polygon transformation {method}: {str(e)}")
            return image, polygons

    except Exception as e:
        logger.error(f"Error in polygon augmentation: {e}")
        return None, []

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

def parse_yolo_label_from_boxes(boxes):
    """
    Parse YOLO format boxes into separate components.
    
    Args:
        boxes (list): List of YOLO format boxes [class_id, x_center, y_center, width, height]
        
    Returns:
        tuple: (bboxes, class_ids) where bboxes are in normalized YOLO format
    """
    if not boxes:
        return [], []
    
    bboxes = []
    class_ids = []
    
    for box in boxes:
        if len(box) >= 5:
            class_ids.append(int(box[0]))
            # Ensure coordinates are normalized (0-1)
            x_center = float(box[1]) if box[1] <= 1.0 else float(box[1]) / 1000.0
            y_center = float(box[2]) if box[2] <= 1.0 else float(box[2]) / 1000.0
            width = float(box[3]) if box[3] <= 1.0 else float(box[3]) / 1000.0
            height = float(box[4]) if box[4] <= 1.0 else float(box[4]) / 1000.0
            
            bboxes.append([x_center, y_center, width, height])
    
    return bboxes, class_ids

def convert_to_yolo_format(augmented_bboxes, augmented_class_ids):
    """
    Convert augmented bboxes back to YOLO format.
    
    Args:
        augmented_bboxes (list): Augmented bounding boxes
        augmented_class_ids (list): Corresponding class IDs
        
    Returns:
        list: Boxes in YOLO format [class_id, x_center, y_center, width, height]
    """
    result_boxes = []
    
    for bbox, class_id in zip(augmented_bboxes, augmented_class_ids):
        if len(bbox) >= 4:
            # Filter out very small boxes (best practice)
            if bbox[2] < 0.01 or bbox[3] < 0.01:  # width or height < 1% of image
                continue
                
            result_boxes.append([
                int(class_id),
                float(bbox[0]),  # x_center
                float(bbox[1]),  # y_center
                float(bbox[2]),  # width
                float(bbox[3])   # height
            ])
    
    return result_boxes

def augment_image_with_boxes(image, boxes, method, level1, level2, min_visibility=0.3, min_size=20):
    """
    Augmentiert ein Bild mit zugehörigen Bounding Boxes unter Verwendung von Albumentations.
    Updated to handle both bounding boxes and polygons automatically.
    
    Args:
        image: Input image (numpy array)
        boxes: List of YOLO format boxes OR polygons
        method: Augmentation method
        level1, level2: Augmentation parameters
        min_visibility: Minimum required visibility (0-1)
        min_size: Minimum size in pixels
        
    Returns:
        tuple: (augmented_image, augmented_annotations)
    """
    # Automatically detect annotation format
    annotation_format = detect_annotation_format(boxes)
    
    if annotation_format == 'polygon':
        return augment_image_with_polygons(image, boxes, method, level1, level2, min_visibility, min_size)
    elif annotation_format == 'mixed':
        logger.warning("Mixed annotation format detected - using only bounding boxes")
        # Filter to only bounding boxes
        bbox_only = [box for box in boxes if len(box) == 5]
        return augment_image_with_bboxes_only(image, bbox_only, method, level1, level2, min_visibility, min_size)
    else:
        # Default to bounding box augmentation
        return augment_image_with_bboxes_only(image, boxes, method, level1, level2, min_visibility, min_size)

def augment_image_with_bboxes_only(image, boxes, method, level1, level2, min_visibility=0.3, min_size=20):
    """
    Original bounding box augmentation function (renamed for clarity).
    """
    if image is None:
        logger.error("Ungültiges Bild")
        return None, []
    
    try:
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            logger.error("Ungültige Bilddimensionen")
            return None, []

        # Parse YOLO boxes into the format expected by Albumentations
        bboxes, class_ids = parse_yolo_label_from_boxes(boxes)
        
        if not bboxes and boxes:
            logger.warning("Keine gültigen Bounding Boxes gefunden")
            return image, boxes

        # Create transform based on method - using the same approach as annotation_DEV.py
        transforms = []
        
        if method == "Rotate":
            # Random rotation between level1 and level2
            rotate_angle = np.random.uniform(level1, level2)
            transforms.append(Rotate(
                limit=(rotate_angle, rotate_angle), 
                p=1.0, 
                border_mode=cv2.BORDER_REFLECT_101
            ))
            
        elif method == "Zoom" or method == "Scale":
            # Convert percentage levels to scale factor and keep values reasonable
            scale_percent = np.random.uniform(level1, level2)
            scale_factor = 1.0 + scale_percent / 100.0
            # Clamp extreme values to avoid huge images that cause OOM errors
            scale_factor = max(0.1, min(scale_factor, 3.0))
            transforms.append(RandomScale(
                scale_limit=(scale_factor - 1, scale_factor - 1),
                p=1.0,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101
            ))
            
            
        elif method == "HorizontalFlip":
            transforms.append(HorizontalFlip(p=1.0))
            
        elif method == "VerticalFlip":
            transforms.append(VerticalFlip(p=1.0))
            
        elif method == "Brightness":
            # Random brightness between level1 and level2
            brightness_change = np.random.uniform(level1/100, level2/100)
            if np.random.random() < 0.5:
                brightness_change = -brightness_change
            transforms.append(RandomBrightnessContrast(
                brightness_limit=brightness_change,
                contrast_limit=brightness_change,
                p=1.0
            ))
            
        elif method == "Blur":
            # Random blur between level1 and level2
            blur_limit = int(np.random.uniform(level1, level2))
            if blur_limit > 0:
                transforms.append(Blur(
                    blur_limit=(blur_limit, blur_limit*2+1),
                    p=1.0
                ))
                
        elif method == "HSV":
            # Random HSV shift
            hsv_shift = np.random.uniform(level1, level2)
            transforms.append(HueSaturationValue(
                hue_shift_limit=hsv_shift,
                sat_shift_limit=hsv_shift,
                val_shift_limit=0,
                p=1.0
            ))
            
        elif method == "Shift":
            # Random shift between level1 and level2 (in percentage)
            shift_limit = np.random.uniform(level1 / 100, level2 / 100)
            # Random direction (positive/negative) for both axes
            shift_x = shift_limit * (1 if np.random.random() < 0.5 else -1)
            shift_y = shift_limit * (1 if np.random.random() < 0.5 else -1)
            transforms.append(ShiftScaleRotate(
                shift_limit_x=(shift_x, shift_x),
                shift_limit_y=(shift_y, shift_y),
                scale_limit=0,
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ))
            
        elif method == "Distortion":
            # Optical distortion
            distort_limit = np.random.uniform(level1, level2)
            if distort_limit > 0:
                transforms.append(OpticalDistortion(
                    distort_limit=distort_limit,
                    p=1.0,
                    border_mode=cv2.BORDER_REFLECT_101
                ))
        else:
            logger.warning(f"Unbekannte Augmentierungsmethode: {method}")
            return image, boxes

        if not transforms:
            return image, boxes

        # Create the compose transform with YOLO bbox format (same as annotation_DEV.py)
        transform = Compose(
            transforms, 
            bbox_params={'format': 'yolo', 'label_fields': ['category_ids']}
        )

        try:
            # Apply transformation
            augmented = transform(
                image=image, 
                bboxes=bboxes, 
                category_ids=class_ids
            )
            
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_ids = augmented['category_ids']
            
            # Convert back to YOLO format and filter small boxes
            result_boxes = convert_to_yolo_format(aug_bboxes, aug_class_ids)
            
            return aug_image, result_boxes
            
        except Exception as e:
            logger.error(f"Fehler bei der Transformation {method}: {str(e)}")
            # Return original image and boxes as fallback
            return image, boxes

    except Exception as e:
        logger.error(f"Fehler bei der Augmentierung: {e}")
        return None, []

def validate_box(box, image_shape, min_visibility=0.3, min_size=20):
    """
    Validate if a bounding box is valid after augmentation.
    
    Args:
        box (list): Box coordinates in YOLO format [class_id, x_center, y_center, width, height]
        image_shape (tuple): Image shape (height, width)
        min_visibility (float): Minimum required visibility (0-1)
        min_size (int): Minimum size in pixels
        
    Returns:
        bool: True if box is valid
    """
    if len(box) < 5:
        return False
        
    height, width = image_shape[:2]
    _, x_center, y_center, box_width, box_height = box
    
    # Check if box dimensions are reasonable
    if box_width <= 0 or box_height <= 0:
        return False
        
    # Check if box is within image bounds (with some tolerance)
    if (x_center < -0.1 or x_center > 1.1 or 
        y_center < -0.1 or y_center > 1.1):
        return False
        
    # Check minimum size in pixels
    pixel_width = box_width * width
    pixel_height = box_height * height
    
    if pixel_width < min_size or pixel_height < min_size:
        return False
        
    return True

# Legacy functions for backwards compatibility
def robust_convert_boxes_to_albumentations(boxes, image_shape):
    """Legacy function - use parse_yolo_label_from_boxes instead."""
    logger.warning("Using deprecated function. Consider using parse_yolo_label_from_boxes.")
    bboxes, class_ids = parse_yolo_label_from_boxes(boxes)
    return [[*bbox, class_id] for bbox, class_id in zip(bboxes, class_ids)]

def fallback_clip_boxes(boxes, image_shape):
    """Legacy function for clipping boxes to image boundaries."""
    height, width = image_shape[:2]
    clipped_boxes = []
    
    for box in boxes:
        if len(box) >= 5:
            class_id = int(box[0])
            x_center = max(0.0, min(1.0, float(box[1])))
            y_center = max(0.0, min(1.0, float(box[2])))
            box_width = max(0.01, min(1.0, float(box[3])))  # Minimum 1% width
            box_height = max(0.01, min(1.0, float(box[4])))  # Minimum 1% height
            
            clipped_boxes.append([class_id, x_center, y_center, box_width, box_height])
    
    return clipped_boxes

def convert_boxes_to_albumentations(boxes, image_shape):
    """Legacy function - use parse_yolo_label_from_boxes instead."""
    return robust_convert_boxes_to_albumentations(boxes, image_shape)