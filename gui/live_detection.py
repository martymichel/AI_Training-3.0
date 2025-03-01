"""Live object detection application module."""

import cv2
import numpy as np
from ultralytics import YOLO
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox,
    QFileDialog, QGroupBox, QSlider, QScrollArea, QListWidget,
    QListWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
import yaml
from pathlib import Path
import json
from PyQt6.QtWidgets import QDialog
import logging
import os

class CameraSelectionDialog(QDialog):
    """Dialog for selecting a camera device."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Camera list
        self.camera_list = QListWidget()
        layout.addWidget(self.camera_list)
        
        # Populate camera list
        self.available_cameras = self.find_cameras()
        for i, (name, _) in enumerate(self.available_cameras):
            item = QListWidgetItem(f"Camera {i}: {name}")
            self.camera_list.addItem(item)
        
        if not self.available_cameras:
            self.camera_list.addItem("No cameras found")
            self.camera_list.setEnabled(False)
        else:
            self.camera_list.setCurrentRow(0)
        
        # Buttons
        button_layout = QHBoxLayout()
        select_btn = QPushButton("Select")
        select_btn.clicked.connect(self.accept)
        select_btn.setEnabled(bool(self.available_cameras))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(select_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def find_cameras(self):
        """Find available camera devices."""
        available_cameras = []
        # Try opening each camera index
        for i in range(10):  # Check first 10 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera name if possible
                name = cap.getBackendName()
                available_cameras.append((name, i))
                cap.release()
        return available_cameras
    
    def get_selected_camera(self):
        """Get the selected camera index."""
        if not self.available_cameras:
            return None
        current_row = self.camera_list.currentRow()
        if current_row >= 0:
            return self.available_cameras[current_row][1]
        return None

class ThresholdSettingsDialog(QDialog):
    """Dialog for configuring detection thresholds."""
    
    def __init__(self, class_names, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Settings")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Motion parameters group
        motion_group = QGroupBox("Motion Detection")
        motion_layout = QVBoxLayout()
        
        # Motion threshold
        motion_label = QLabel("Motion Threshold:")
        self.motion_spin = QSpinBox()
        self.motion_spin.setRange(1, 255)
        self.motion_spin.setValue(settings.get('motion_threshold', 110) if settings else 110)
        motion_layout.addWidget(motion_label)
        motion_layout.addWidget(self.motion_spin)
        
        # Static frames
        static_label = QLabel("Min Static Frames:")
        self.static_spin = QSpinBox()
        self.static_spin.setRange(1, 10)
        self.static_spin.setValue(settings.get('static_frame_min', 3) if settings else 3)
        motion_layout.addWidget(static_label)
        motion_layout.addWidget(self.static_spin)
        
        motion_group.setLayout(motion_layout)
        layout.addWidget(motion_group)
        
        # IoU threshold group
        iou_group = QGroupBox("IoU Threshold")
        iou_layout = QVBoxLayout()
        
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(int(settings.get('iou_threshold', 0.45) * 100) if settings else 45)
        self.iou_label = QLabel(f"IoU: {self.iou_slider.value()/100:.2f}")
        self.iou_slider.valueChanged.connect(lambda v: self.iou_label.setText(f"IoU: {v/100:.2f}"))
        
        iou_layout.addWidget(self.iou_label)
        iou_layout.addWidget(self.iou_slider)
        iou_group.setLayout(iou_layout)
        layout.addWidget(iou_group)
        
        # Class thresholds group
        self.class_group = QGroupBox("Class Thresholds")
        self.class_layout = QVBoxLayout()
        self.class_sliders = {}
        
        for class_id, class_name in class_names.items():
            group = QGroupBox(f"Class {class_id}: {class_name}")
            slider_layout = QVBoxLayout()
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(1, 99)
            default_value = int(settings.get('class_thresholds', {}).get(str(class_id), 0.7) * 100) if settings else 70
            slider.setValue(default_value)
            
            label = QLabel(f"Confidence: {default_value/100:.2f}")
            slider.valueChanged.connect(lambda v, l=label: l.setText(f"Confidence: {v/100:.2f}"))
            
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            group.setLayout(slider_layout)
            
            self.class_layout.addWidget(group)
            self.class_sliders[class_id] = (slider, label)
        
        self.class_group.setLayout(self.class_layout)
        layout.addWidget(self.class_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def get_settings(self):
        """Get current settings as dictionary."""
        return {
            'motion_threshold': self.motion_spin.value(),
            'static_frame_min': self.static_spin.value(),
            'iou_threshold': self.iou_slider.value() / 100,
            'class_thresholds': {
                str(class_id): slider[0].value() / 100
                for class_id, slider in self.class_sliders.items()
            }
        }

class FrameConfigDialog(QDialog):
    """Dialog for configuring frame colors based on class detection."""
    
    def __init__(self, class_names, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame Configuration")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        info_label = QLabel(
            "Assign classes to frame colors. Red frame has priority over green "
            "when both conditions are met."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Class assignments
        self.class_assignments = {}
        for class_id, class_name in class_names.items():
            group = QGroupBox(f"Class {class_id}: {class_name}")
            group_layout = QHBoxLayout()
            
            combo = QComboBox()
            combo.addItems(["None", "Green Frame", "Red Frame"])
            default_value = settings.get('frame_assignments', {}).get(str(class_id), "None") if settings else "None"
            combo.setCurrentText(default_value)
            
            group_layout.addWidget(combo)
            group.setLayout(group_layout)
            layout.addWidget(group)
            self.class_assignments[class_id] = combo
        
        # Minimum detection thresholds
        threshold_group = QGroupBox("Minimum Detections")
        threshold_layout = QVBoxLayout()
        
        # Green frame threshold
        green_layout = QHBoxLayout()
        green_layout.addWidget(QLabel("Green Frame Min:"))
        self.green_threshold = QSpinBox()
        self.green_threshold.setRange(1, 20)
        self.green_threshold.setValue(settings.get('green_threshold', 4) if settings else 4)
        green_layout.addWidget(self.green_threshold)
        threshold_layout.addLayout(green_layout)
        
        # Red frame threshold
        red_layout = QHBoxLayout()
        red_layout.addWidget(QLabel("Red Frame Min:"))
        self.red_threshold = QSpinBox()
        self.red_threshold.setRange(1, 20)
        self.red_threshold.setValue(settings.get('red_threshold', 1) if settings else 1)
        red_layout.addWidget(self.red_threshold)
        threshold_layout.addLayout(red_layout)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def get_settings(self):
        """Get current settings as dictionary."""
        return {
            'frame_assignments': {
                str(class_id): combo.currentText()
                for class_id, combo in self.class_assignments.items()
            },
            'green_threshold': self.green_threshold.value(),
            'red_threshold': self.red_threshold.value()
        }

class DetectionThread(QThread):
    """Thread for running object detection."""
    
    frame_ready = pyqtSignal(QImage, list)  # frame and detection results
    error = pyqtSignal(str)
    
    def __init__(self, model_path, class_names, camera_id=0):
        super().__init__()
        self.model_path = model_path
        self.class_names = class_names
        self.camera_id = camera_id
        self.running = False
        self.camera = None
        self.model = None
        self.paused = False
        self.frame_config = None
        
        # Class-specific thresholds
        self.class_thresholds = {}
        self.iou_threshold = 0.45
        
        # Motion detection parameters
        self.prev_gray = None
        self.static_buffer = []
        self.in_static_period = False
        self.evaluation_done = False
        
        # Detection parameters
        self.motion_threshold = 110
        self.static_frame_min = 3
        self.confidence_threshold = 0.7
        
    def run(self):
        """Main detection loop."""
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            # Load YOLO model
            self.model = YOLO(self.model_path)
            
            self.running = True
            while self.running:
                if self.paused:
                    self.msleep(100)
                    continue
                
                ret, frame = self.camera.read()
                if not ret:
                    raise Exception("Failed to grab frame")
                
                # Motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.prev_gray is None:
                    self.prev_gray = gray
                    continue
                
                diff = cv2.absdiff(gray, self.prev_gray)
                max_diff = np.max(diff)
                self.prev_gray = gray
                
                # Process frame based on motion state
                results = None
                if max_diff < self.motion_threshold:
                    self.static_buffer.append(max_diff)
                    if len(self.static_buffer) > 10:  # Keep last 10 frames
                        self.static_buffer.pop(0)
                    # Calculate if period is static based on recent frames
                    self.in_static_period = np.mean(self.static_buffer) < self.motion_threshold
                    
                    # Run detection with IoU threshold
                    results = self.model(
                        frame,
                        iou=self.iou_threshold
                    )[0]
                    
                    boxes = results.boxes
                    annotated_frame = frame.copy()
                    
                    if boxes is not None and len(boxes) > 0:
                        cls_array = boxes.cls.cpu().numpy()
                        conf_array = boxes.conf.cpu().numpy()
                        xyxy = boxes.xyxy.cpu().numpy()
                        
                        # Apply class-specific thresholds after detection
                        valid_detections = np.zeros_like(cls_array, dtype=bool)
                        for class_id_str, threshold in self.class_thresholds.items():
                            class_id = int(class_id_str)
                            class_mask = (cls_array == class_id) & (conf_array >= float(threshold))
                            valid_detections |= class_mask
                        
                        # Filter boxes based on thresholds
                        cls_array = cls_array[valid_detections]
                        conf_array = conf_array[valid_detections]
                        xyxy = xyxy[valid_detections]
                        
                        # Draw bounding boxes
                        for i in range(len(cls_array)):
                            x1, y1, x2, y2 = map(int, xyxy[i])
                            cls = int(cls_array[i])
                            conf = conf_array[i]
                            
                            # Set color based on class
                            if cls == 0:
                                color = (20, 255, 57)  # neon green
                            elif cls == 1:
                                color = (0, 0, 255)    # red
                            else:
                                color = (238, 130, 238)  # violet
                            
                            # Draw box and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{self.class_names.get(cls, str(cls))} {conf:.2f}"
                            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 0)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        if self.frame_config:
                            # Count detections by frame color
                            green_count = 0
                            red_count = 0
                            
                            for class_id in np.unique(cls_array):
                                count = np.sum(cls_array == class_id)
                                assignment = self.frame_config['frame_assignments'].get(str(int(class_id)), "None")
                                
                                if assignment == "Green Frame":
                                    green_count += count
                                elif assignment == "Red Frame":
                                    red_count += count
                            
                            # Apply frame based on counts and thresholds
                            if red_count >= self.frame_config['red_threshold']:
                                annotated_frame = self.draw_border(annotated_frame, (0, 0, 255), 30)  # Red border
                            elif green_count >= self.frame_config['green_threshold']:
                                annotated_frame = self.draw_border(annotated_frame, (0, 255, 0), 10)  # Green border
                    else:
                        annotated_frame = frame.copy()
                        
                else:
                    # Reset state on motion
                    self.in_static_period = False
                    self.static_buffer = []
                    annotated_frame = frame.copy()
                
                # Convert frame to QImage
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    rgb_frame.data,
                    w, h,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                )
                
                # Emit frame with results
                self.frame_ready.emit(qt_image, [results, max_diff] if results else [None, max_diff])
                
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self.camera:
                self.camera.release()
    
    def draw_border(self, frame, color, thickness):
        """Draw colored border around frame."""
        h, w = frame.shape[:2]
        return cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
    
    def stop(self):
        """Stop detection thread."""
        self.running = False
        self.wait()

class LiveDetectionApp(QMainWindow):
    """Main window for live object detection."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Object Detection")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize attributes
        self.detection_thread = None
        
        # Model and class info
        self.model_path = None
        self.class_names = {}
        self.num_classes = 0
        self.settings_file = "detection_settings.json"
        self.settings = self.load_settings()
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Model selection
        model_label = QLabel("Model (.pt):")
        self.model_path_label = QLabel("Not selected")
        model_btn = QPushButton("Browse")
        model_btn.clicked.connect(self.browse_model)
        
        # YAML selection
        yaml_label = QLabel("Data (.yaml):")
        self.yaml_path_label = QLabel("Not selected")
        yaml_btn = QPushButton("Browse")
        yaml_btn.clicked.connect(self.browse_yaml)
        
        # Connect button
        self.connect_btn = QPushButton("Start Detection")
        self.connect_btn.clicked.connect(self.toggle_detection)
        self.connect_btn.setEnabled(False)
        
        controls_layout.addWidget(model_label)
        controls_layout.addWidget(self.model_path_label)
        controls_layout.addWidget(model_btn)
        controls_layout.addWidget(yaml_label)
        controls_layout.addWidget(self.yaml_path_label)
        controls_layout.addWidget(yaml_btn)
        controls_layout.addWidget(self.connect_btn)
        layout.addLayout(controls_layout)

        # Settings buttons
        settings_layout = QHBoxLayout()
        threshold_btn = QPushButton("Detection Settings...")
        threshold_btn.clicked.connect(self.show_threshold_settings)
        frame_btn = QPushButton("Frame Settings...")
        frame_btn.clicked.connect(self.show_frame_settings)
        settings_layout.addWidget(threshold_btn)
        settings_layout.addWidget(frame_btn)
        layout.addLayout(settings_layout)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        
        # Status display
        self.status_label = QLabel()
        status_font = QFont()
        status_font.setPointSize(12)
        self.status_label.setFont(status_font)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
    def browse_model(self):
        """Open file dialog to select YOLO model."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "Model Files (*.pt)"
        )
        if path:
            self.model_path = path
            self.model_path_label.setText(Path(path).name)
            self.check_ready()
    
    def browse_yaml(self):
        """Open file dialog to select data YAML file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Data YAML", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            try:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' not in data:
                        raise ValueError("YAML file must contain 'names' field")
                    
                    self.class_names = data['names']
                    if isinstance(self.class_names, list):
                        self.class_names = {i: name for i, name in enumerate(self.class_names)}
                    
                    self.num_classes = len(self.class_names)
                    if self.num_classes > 8:
                        raise ValueError("Maximum 8 classes supported")
                    
                    self.yaml_path_label.setText(Path(path).name)
                    self.check_ready()
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Invalid YAML file: {str(e)}")
    
    def load_settings(self):
        """Load settings from JSON file."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load settings: {e}")
        return {}
    
    def save_settings(self):
        """Save settings to JSON file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")
    
    def show_threshold_settings(self):
        """Show threshold settings dialog."""
        if not self.class_names:
            QMessageBox.warning(self, "Warning", "Please load a YAML file first")
            return
        
        dialog = ThresholdSettingsDialog(self.class_names, self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings.update(dialog.get_settings())
            self.save_settings()
            # Update running thread if active
            if self.detection_thread and self.detection_thread.running:
                self.detection_thread.motion_threshold = self.settings.get('motion_threshold', 110)
                self.detection_thread.static_frame_min = self.settings.get('static_frame_min', 3)
                self.detection_thread.iou_threshold = self.settings.get('iou_threshold', 0.45)
                self.detection_thread.class_thresholds = self.settings.get('class_thresholds', {})
    
    def show_frame_settings(self):
        """Show frame configuration dialog."""
        if not self.class_names:
            QMessageBox.warning(self, "Warning", "Please load a YAML file first")
            return
        
        dialog = FrameConfigDialog(self.class_names, self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings.update(dialog.get_settings())
            self.save_settings()
            # Update running thread if active
            if self.detection_thread and self.detection_thread.running:
                self.detection_thread.frame_config = {
                    'frame_assignments': self.settings.get('frame_assignments', {}),
                    'green_threshold': self.settings.get('green_threshold', 4),
                    'red_threshold': self.settings.get('red_threshold', 1)
                }
    
    def check_ready(self):
        """Check if all required files are selected."""
        self.connect_btn.setEnabled(
            self.model_path is not None and
            self.num_classes > 0
        )
    
    def toggle_detection(self):
        """Start or stop detection."""
        if self.detection_thread and self.detection_thread.running:
            try:
                # Stop detection
                self.detection_thread.stop()
                self.detection_thread = None
                self.connect_btn.setText("Start Detection")
                self.status_label.clear()
                self.video_label.clear()
                self.video_label.setStyleSheet("background-color: black;")
            except Exception as e:
                logging.error(f"Error stopping detection: {e}")
        else:
            # Start detection
            try:
                # Show camera selection dialog
                camera_dialog = CameraSelectionDialog(self)
                if camera_dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                
                camera_id = camera_dialog.get_selected_camera()
                if camera_id is None:
                    QMessageBox.critical(self, "Error", "No camera selected")
                    return
                
                self.detection_thread = DetectionThread(
                    model_path=self.model_path,
                    class_names=self.class_names,
                    camera_id=camera_id
                )
                
                # Update parameters
                self.detection_thread.motion_threshold = self.settings.get('motion_threshold', 110)
                self.detection_thread.static_frame_min = self.settings.get('static_frame_min', 3)
                self.detection_thread.iou_threshold = self.settings.get('iou_threshold', 0.45)
                self.detection_thread.class_thresholds = self.settings.get('class_thresholds', {})
                self.detection_thread.frame_config = {
                    'frame_assignments': self.settings.get('frame_assignments', {}),
                    'green_threshold': self.settings.get('green_threshold', 4),
                    'red_threshold': self.settings.get('red_threshold', 1)
                }
                
                self.detection_thread.frame_ready.connect(self.update_frame)
                self.detection_thread.error.connect(self.handle_error)
                self.detection_thread.start()
                
                self.connect_btn.setText("Stop Detection")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def update_frame(self, image, results):
        """Update video display with detection results."""
        # Scale image to fit display
        try:
            scaled_pixmap = QPixmap.fromImage(image).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        
            # Update status
            if results[0] is not None:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    cls_array = boxes.cls.cpu().numpy()
                    
                    # Count detections per class using class-specific thresholds
                    counts = {}
                    for class_id in self.class_names.keys():
                        counts[class_id] = np.sum(cls_array == class_id)
                    
                    # Build status string
                    status_parts = []
                    for class_id, count in counts.items():
                        status_parts.append(f"{self.class_names[class_id]}: {count}")
                    status_parts.extend([
                        f"Motion: {results[1]:.2f}",
                        f"Static: {self.detection_thread.in_static_period}"
                    ])
                    status = " | ".join(status_parts)
                    self.status_label.setText(status)
        except Exception as e:
            logging.error(f"Error updating frame: {e}")
    
    def handle_error(self, message):
        """Handle detection errors."""
        QMessageBox.critical(self, "Error", message)
        self.toggle_detection()
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.detection_thread and self.detection_thread.running:
            self.detection_thread.stop()
        event.accept()