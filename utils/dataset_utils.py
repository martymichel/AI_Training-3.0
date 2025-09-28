"""Utility functions for dataset operations."""

import os
import shutil
import random
import yaml
from pathlib import Path
import logging
import cv2
import numpy as np
from typing import Dict, Set, List, Tuple
from PyQt6.QtCore import QThread, pyqtSignal

class DatasetSplitter:
    """Class for splitting image datasets into train/val/test sets."""
    
    def __init__(self):
        """Initialize the DatasetSplitter."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def detect_classes(self, label_files: List[Path]) -> Set[int]:
        """Detect all unique class IDs from label files."""
        classes = set()
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        # Convert float class ID to int
                        class_id = int(float(line.strip().split()[0]))
                        classes.add(class_id)
            except Exception as e:
                self.logger.warning(f"Error reading {label_file}: {e}")
        return classes

    def find_sample_image(self, image_files: List[Path], label_files: List[Path]) -> Tuple[Path, Path]:
        """Find an image that contains all classes."""
        class_counts = {}
        for img_path, label_path in zip(image_files, label_files):
            try:
                classes = set()
                with open(label_path, 'r') as f:
                    for line in f:
                        # Convert float class ID to int
                        class_id = int(float(line.strip().split()[0]))
                        classes.add(class_id)
                class_counts[img_path] = len(classes)
            except Exception as e:
                self.logger.warning(f"Error reading {label_path}: {e}")

        # Find image with most classes
        if class_counts:
            best_image = max(class_counts.items(), key=lambda x: x[1])[0]
            return best_image, best_image.with_suffix('.txt')
        return None, None

    def load_image_with_boxes(self, image_path: Path, label_path: Path, 
                            highlight_class: int = None) -> Tuple[np.ndarray, List[Tuple]]:
        """Load image and draw bounding boxes."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None, []

        height, width = image.shape[:2]
        boxes = []

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # Convert float class ID to int
                    class_id = int(float(parts[0]))
                    x_center, y_center, w, h = map(float, parts[1:5])
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)
                    
                    boxes.append((class_id, x1, y1, x2, y2))
        except Exception as e:
            self.logger.error(f"Error reading boxes from {label_path}: {e}")

        return image, boxes

    def find_image_label_pairs(self, source_dir):
        """Find matching image and label files in the source directory."""
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(Path(source_dir).rglob(f"*{ext}"))
        
        valid_pairs = []
        for img_path in image_files:
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                valid_pairs.append((img_path, label_path))
            else:
                self.logger.warning(f"No label file found for: {img_path}")
        
        return valid_pairs

    def create_directories(self, output_dir):
        """Create the necessary output directories with images/labels subfolders."""
        dirs = {}
        for split in ['train', 'val', 'test']:
            root = os.path.join(output_dir, split)
            images = os.path.join(root, 'images')
            labels = os.path.join(root, 'labels')
            os.makedirs(images, exist_ok=True)
            os.makedirs(labels, exist_ok=True)
            dirs[split] = {
                'root': root,
                'images': images,
                'labels': labels,
            }

        return dirs

    def copy_files(self, file_pairs, destination_dirs, progress_callback=None):
        """Copy image and label files to destination directories."""
        successful_copies = 0

        for img_path, label_path in file_pairs:
            try:
                # Copy image
                img_dest = os.path.join(destination_dirs['images'], img_path.name)
                shutil.copy2(img_path, img_dest)

                # Copy label
                label_dest = os.path.join(destination_dirs['labels'], label_path.name)
                shutil.copy2(label_path, label_dest)
                
                successful_copies += 1
                
                if progress_callback:
                    progress_callback(f"Copied {successful_copies} pairs...")
                    
            except Exception as e:
                self.logger.error(f"Error copying {img_path}: {e}")
        
        return successful_copies

    def create_dataset_files(self, output_dir, split_dirs, split_data):
        """Create YOLO dataset files (train.txt, val.txt, data.yaml)."""
        # Create train.txt and val.txt
        for split_name in ['train', 'val']:
            txt_path = os.path.join(output_dir, f"{split_name}.txt")
            with open(txt_path, 'w') as f:
                for img_path, _ in split_data[split_name]:
                    f.write(str(os.path.join(
                        split_dirs[split_name]['images'],
                        img_path.name
                    )) + '\n')

        # Create data.yaml with consistent forward slashes for cross-platform compatibility
        yaml_content = {
            'path': output_dir.replace('\\', '/'),
            'train': os.path.join(output_dir, 'train.txt').replace('\\', '/'),
            'val': os.path.join(output_dir, 'val.txt').replace('\\', '/'),
            'names': self.class_names
        }

        yaml_path = os.path.join(output_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

    def split_dataset(self, source_dir, output_dir, train_ratio, val_ratio, class_names, progress_callback=None):
        """
        Split dataset into train/val/test sets.
        
        Args:
            source_dir (str): Source directory containing images and labels
            output_dir (str): Output directory for split dataset
            train_ratio (float): Ratio of training data (0-1)
            val_ratio (float): Ratio of validation data (0-1)
            class_names (dict): Dictionary mapping class IDs to names
            progress_callback (callable): Optional callback for progress updates
        """
        try:
            self.class_names = class_names  # Store class names for use in create_dataset_files
            if progress_callback:
                progress_callback("Finding image-label pairs...")
            
            # Find valid image-label pairs
            valid_pairs = self.find_image_label_pairs(source_dir)
            if not valid_pairs:
                raise ValueError("No valid image-label pairs found")
            
            # Create output directories
            split_dirs = self.create_directories(output_dir)
            
            # Shuffle and split data
            random.shuffle(valid_pairs)
            total_files = len(valid_pairs)
            train_size = int(train_ratio * total_files)
            val_size = int(val_ratio * total_files)
            
            split_data = {
                'train': valid_pairs[:train_size],
                'val': valid_pairs[train_size:train_size + val_size],
                'test': valid_pairs[train_size + val_size:]
            }
            
            # Copy files for each split
            for split_name, pairs in split_data.items():
                if progress_callback:
                    progress_callback(f"Copying {split_name} files...")
                self.copy_files(pairs, split_dirs[split_name], progress_callback)
            
            # Create dataset files
            if progress_callback:
                progress_callback("Creating dataset files...")
            self.create_dataset_files(output_dir, split_dirs, split_data)
            
            if progress_callback:
                progress_callback("Dataset split complete!")
            
        except Exception as e:
            self.logger.error(f"Error splitting dataset: {e}")
            raise

class DatasetSplitterThread(QThread):
    """Thread for dataset splitting operations."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, splitter, source_dir, output_dir, train_split, val_split, class_names):
        super().__init__()
        self.splitter = splitter
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.train_split = train_split
        self.val_split = val_split
        self.class_names = class_names

    def run(self):
        """Execute the dataset splitting operation."""
        try:
            self.splitter.split_dataset(
                self.source_dir,
                self.output_dir,
                self.train_split / 100,
                self.val_split / 100,
                self.class_names,
                progress_callback=self.progress.emit
            )
            self.finished.emit(True, "Dataset successfully split!")
        except Exception as e:
            self.finished.emit(False, str(e))