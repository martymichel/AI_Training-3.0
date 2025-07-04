"""Preview functionality for the augmentation application."""

import os
import cv2
import logging
import random
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from utils.augmentation_utils import augment_image_with_boxes
from pathlib import Path

logger = logging.getLogger(__name__)

def load_sample_image(app):
    """Load a sample image for preview."""
    if not app.source_path:
        return
        
    try:
        # Collect available images and pick one at random
        image_files = list(Path(app.source_path).rglob("*.jpg")) + \
                     list(Path(app.source_path).rglob("*.png"))
        if not image_files:
            return

        image_path = str(random.choice(image_files))
        label_path = os.path.splitext(image_path)[0] + ".txt"
        
        # Load image
        app.sample_image = cv2.imread(image_path)
        if app.sample_image is None:
            logger.error(f"Failed to load sample image: {image_path}")
            return
            
        # Load boxes if label exists
        app.sample_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            app.sample_boxes.append(list(map(float, parts)))
                    except Exception as e:
                        logger.error(f"Error parsing label: {e}")
        
        # Show original image
        display_image(app.sample_image, app.original_preview)
        
        # Update preview
        generate_preview(app)
        
    except Exception as e:
        logger.error(f"Error loading sample image: {e}")

def display_image(image, label_widget, max_size=300):
    """Display image in a label widget with proper scaling."""
    if image is None:
        return
        
    # Convert to RGB for Qt
    display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Calculate aspect ratio preserving scale
    h, w = display_img.shape[:2]
    scale = min(max_size/w, max_size/h)
    
    # Resize image
    new_w = int(w * scale)
    new_h = int(h * scale)
    display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convert to QImage and display
    qimg = QImage(display_img.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
    label_widget.setPixmap(QPixmap.fromImage(qimg))

def toggle_preview_mode(app, state):
    """Toggle preview visibility."""
    is_visible = state == Qt.CheckState.Checked.value
    app.refresh_preview_button.setEnabled(is_visible)
    
    # Switch to appropriate stack index
    app.stack.setCurrentIndex(1 if is_visible else 0)
    
    # Update preview if checked
    if is_visible:
        generate_preview(app)

def update_preview(app):
    """Schedule an update to the preview images."""
    if not app.preview_checkbox.isChecked() or app.sample_image is None:
        return
        
    # Use a timer to debounce frequent updates
    app.preview_timer.start(100)

def generate_preview(app):
    """Generate preview images for Level 1 and Level 2 of active augmentations."""
    # Ensure preview is enabled when clicking "Neue Vorschau"
    if not app.preview_checkbox.isChecked():
        app.preview_checkbox.setChecked(True)
    
    if app.sample_image is None:
        return
        
    # Display original image
    display_image(app.sample_image, app.original_preview)
    
    # Get active methods
    active_methods = []
    for method in app.methods:
        method_key = get_method_key(method)
        if method_key in app.method_levels:
            checkbox, level1_spin, level2_spin = app.method_levels[method_key]
            if checkbox.isChecked():
                active_methods.append((method_key, level1_spin.value(), level2_spin.value()))
    
    # If no active methods and no flips, clear previews and return
    if not active_methods and not app.horizontal_flip.isChecked() and not app.vertical_flip.isChecked():
        app.level1_preview.clear()
        app.level2_preview.clear()
        return
    
    # Generate Level 1 preview - use EXACT Level 1 values (not random ranges)
    level1_img = app.sample_image.copy()
    level1_boxes = app.sample_boxes.copy() if app.sample_boxes else []
    
    # Apply each active method at exactly Level 1 value
    for method_key, level1, _ in active_methods:
        # Use the exact level1 value for both min and max to get consistent preview
        level1_img, level1_boxes = augment_image_with_boxes(
            level1_img, level1_boxes, method_key, level1, level1,  # Same value for min and max = exact level
            min_visibility=app.settings.get('min_visibility', 0.3),
            min_size=app.settings.get('min_size', 20)
        )
    
    # Add flips for Level 1 if enabled (always apply for preview, not random)
    if app.horizontal_flip.isChecked():
        level1_img, level1_boxes = augment_image_with_boxes(
            level1_img, level1_boxes, "HorizontalFlip", 0, 0,
            min_visibility=app.settings.get('min_visibility', 0.3),
            min_size=app.settings.get('min_size', 20)
        )
    
    if app.vertical_flip.isChecked():
        level1_img, level1_boxes = augment_image_with_boxes(
            level1_img, level1_boxes, "VerticalFlip", 0, 0,
            min_visibility=app.settings.get('min_visibility', 0.3),
            min_size=app.settings.get('min_size', 20)
        )
    
    # Display Level 1 preview
    if level1_img is not None:
        display_image(level1_img, app.level1_preview)
    
    # Generate Level 2 preview - use EXACT Level 2 values (not random ranges)
    level2_img = app.sample_image.copy()
    level2_boxes = app.sample_boxes.copy() if app.sample_boxes else []
    
    # Apply each active method at exactly Level 2 value
    for method_key, _, level2 in active_methods:
        # Use the exact level2 value for both min and max to get consistent preview
        level2_img, level2_boxes = augment_image_with_boxes(
            level2_img, level2_boxes, method_key, level2, level2,  # Same value for min and max = exact level
            min_visibility=app.settings.get('min_visibility', 0.3),
            min_size=app.settings.get('min_size', 20)
        )
    
    # Add flips for Level 2 if enabled (always apply for preview, not random)
    if app.horizontal_flip.isChecked():
        level2_img, level2_boxes = augment_image_with_boxes(
            level2_img, level2_boxes, "HorizontalFlip", 0, 0,
            min_visibility=app.settings.get('min_visibility', 0.3),
            min_size=app.settings.get('min_size', 20)
        )
    
    if app.vertical_flip.isChecked():
        level2_img, level2_boxes = augment_image_with_boxes(
            level2_img, level2_boxes, "VerticalFlip", 0, 0,
            min_visibility=app.settings.get('min_visibility', 0.3),
            min_size=app.settings.get('min_size', 20)
        )
    
    # Display Level 2 preview
    if level2_img is not None:
        display_image(level2_img, app.level2_preview)

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