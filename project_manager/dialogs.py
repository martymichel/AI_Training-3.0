"""Dialog classes for project management."""

import sys
import os
import time
from pathlib import Path
from typing import Optional, List
import logging

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QListWidget, QListWidgetItem, QMessageBox,
    QFileDialog, QFrame, QScrollArea, QWidget, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon

from .config import ProjectConfig
from .utils import (
    validate_project_structure, get_project_info, create_project_structure,
    detect_legacy_project, migrate_legacy_project, get_legacy_projects
)

logger = logging.getLogger(__name__)

class ProjectCard(QFrame):
    """Card widget representing a project."""
    
    clicked = pyqtSignal(str)
    
    def __init__(self, project_path: str, parent=None):
        super().__init__(parent)
        self.project_path = project_path
        self.init_ui()
    
    def init_ui(self):
        """Initialize the project card UI."""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setObjectName("projectCard")  # For CSS targeting
        self.setFixedHeight(120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)
        
        # Get project info
        try:
            project_info = get_project_info(Path(self.project_path))
        except Exception:
            # Fallback for legacy projects
            project_info = {
                'project_name': Path(self.project_path).name,
                'created_date': 'Unknown',
                'last_modified': 'Legacy Project'
            }
        
        # Project name
        project_name = project_info.get('project_name', Path(self.project_path).name)
        name_label = QLabel(project_name)
        name_font = QFont()
        name_font.setPointSize(14)
        name_font.setWeight(QFont.Weight.Bold)
        name_label.setFont(name_font)
        layout.addWidget(name_label)
        
        # Project path
        path_label = QLabel(self.project_path)
        path_font = QFont()
        path_font.setPointSize(10)
        path_label.setFont(path_font)
        path_label.setStyleSheet("color: #666;")
        layout.addWidget(path_label)
        
        # Last modified
        modified_info = project_info.get('last_modified', 'Unknown')
        if modified_info != 'Unknown' and len(modified_info) > 10:
            modified_info = modified_info[:10]
        
        # Add legacy indicator
        legacy_indicator = ""
        if detect_legacy_project(Path(self.project_path)):
            legacy_indicator = " (Legacy)"
        
        modified_label = QLabel(f"Modified: {modified_info}{legacy_indicator}")
        modified_font = QFont()
        modified_font.setPointSize(9)
        modified_label.setFont(modified_font)
        modified_label.setStyleSheet("color: #999;")
        layout.addWidget(modified_label)
        
        # Styling
        # Styling wird über parent dialog gesetzt
    
    def mousePressEvent(self, event):
        """Handle mouse click events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.project_path)

class NewProjectDialog(QDialog):
    """Dialog for creating a new project."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setModal(True)
        self.setFixedSize(500, 400)
        
        # Fixed professional styling - matching main dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
                padding: 4px;
                background: transparent;
            }
            QLineEdit, QTextEdit {
                background-color: white;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
                color: #2c3e50;
                font-size: 14px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #3498db;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Project name
        layout.addWidget(QLabel("Project Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter project name...")
        layout.addWidget(self.name_input)
        
        # Project directory
        layout.addWidget(QLabel("Project Directory:"))
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select directory...")
        self.dir_input.setReadOnly(True)
        dir_button = QPushButton("Browse")
        dir_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(dir_button)
        layout.addLayout(dir_layout)
        
        # Description
        layout.addWidget(QLabel("Description (Optional):"))
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("Enter project description...")
        self.description_input.setMaximumHeight(100)
        layout.addWidget(self.description_input)
        
        # Author
        layout.addWidget(QLabel("Author (Optional):"))
        self.author_input = QLineEdit()
        self.author_input.setPlaceholderText("Enter author name...")
        layout.addWidget(self.author_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        create_button = QPushButton("Create Project")
        create_button.clicked.connect(self.create_project)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(create_button)
        layout.addLayout(button_layout)
        
        # Set default directory to current working directory
        default_dir = Path.cwd() / "AI_Projects"
        self.dir_input.setText(str(default_dir))
    
    def browse_directory(self):
        """Browse for project directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Project Directory"
        )
        if directory:
            self.dir_input.setText(directory)
    
    def create_project(self):
        """Create the new project."""
        name = self.name_input.text().strip()
        directory = self.dir_input.text().strip()
        description = self.description_input.toPlainText().strip()
        author = self.author_input.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a project name.")
            return
        
        if not directory:
            QMessageBox.warning(self, "Error", "Please select a project directory.")
            return
        
        # Create project directory
        project_root = Path(directory) / name
        if project_root.exists():
            QMessageBox.warning(
                self, "Error", 
                f"Project directory already exists: {project_root}"
            )
            return
        
        try:
            # Create directory structure
            create_project_structure(project_root)
            
            # Create and save config
            config = ProjectConfig.create_new(
                project_name=name,
                project_root=str(project_root),
                description=description,
                author=author
            )
            
            config_file = project_root / ".project" / "config.json"
            config.save_to_file(config_file)
            
            self.project_path = str(project_root)
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to create project: {str(e)}"
            )

class ProjectManagerDialog(QDialog):
    """Main dialog for project management."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Vision Tools - Project Manager")
        self.setModal(True)
        self.resize(900, 600)
        
        # Fixed professional styling - not OS dependent
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
                padding: 4px;
                background: transparent;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            QFrame#projectCard {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
            }
            QFrame#projectCard:hover {
                border-color: #3498db;
                background-color: #e3f2fd;
            }
            QScrollArea {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
            }
            QTextEdit, QLineEdit {
                background-color: white;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
                color: #2c3e50;
                font-size: 14px;
            }
            QTextEdit:focus, QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        
        self.selected_project_path = None
        self.init_ui()
        self.load_recent_projects()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        # Set fixed, professional styling for readability
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                color: #212529;
            }
            QLabel {
                color: #212529;
                font-weight: 500;
                padding: 4px;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #dee2e6;
            }
            QFrame#projectCard {
                background-color: white;
                border: 2px solid #dee2e6;
                border-radius: 8px;
            }
            QFrame#projectCard:hover {
                border-color: #007bff;
                background-color: #f8f9fa;
            }
            QScrollArea {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Select or Create a Project")
        header_font = QFont()
        header_font.setPointSize(18)
        header_font.setWeight(QFont.Weight.Bold)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)
        
        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(content_splitter)
        
        # Recent projects area
        recent_frame = QFrame()
        recent_frame.setFixedWidth(400)
        recent_layout = QVBoxLayout(recent_frame)
        
        recent_label = QLabel("Recent Projects")
        recent_font = QFont()
        recent_font.setPointSize(12)
        recent_font.setWeight(QFont.Weight.Medium)
        recent_label.setFont(recent_font)
        recent_layout.addWidget(recent_label)
        
        # Scroll area for project cards
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.projects_widget = QWidget()
        self.projects_layout = QVBoxLayout(self.projects_widget)
        scroll_area.setWidget(self.projects_widget)
        recent_layout.addWidget(scroll_area)
        
        content_splitter.addWidget(recent_frame)
        
        # Actions area
        actions_frame = QFrame()
        actions_layout = QVBoxLayout(actions_frame)
        
        actions_label = QLabel("Actions")
        actions_font = QFont()
        actions_font.setPointSize(12)
        actions_font.setWeight(QFont.Weight.Medium)
        actions_label.setFont(actions_font)
        actions_layout.addWidget(actions_label)
        
        # Action buttons
        new_project_btn = QPushButton("Create New Project")
        new_project_btn.setMinimumHeight(50)
        new_project_btn.clicked.connect(self.create_new_project)
        actions_layout.addWidget(new_project_btn)
        
        open_project_btn = QPushButton("Open Existing Project")
        open_project_btn.setMinimumHeight(50)
        open_project_btn.clicked.connect(self.open_existing_project)
        actions_layout.addWidget(open_project_btn)
        
        actions_layout.addStretch()
        
        # Info area
        info_label = QLabel(
            "AI Vision Tools helps you manage complete machine learning workflows "
            "from data collection to deployment. Each project contains all necessary "
            "components for training and deploying computer vision models."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; padding: 20px; background-color: #f5f5f5; border-radius: 8px;")
        actions_layout.addWidget(info_label)
        
        content_splitter.addWidget(actions_frame)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(self.reject)
        button_layout.addWidget(quit_button)
        
        button_layout.addStretch()
        
        ok_button = QPushButton("Open Project")
        ok_button.setEnabled(False)
        ok_button.clicked.connect(self.accept)
        self.ok_button = ok_button
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
    
    def load_recent_projects(self):
        """Load and display recent projects."""
        recent_projects = self.get_recent_projects()
        
        for project_path in recent_projects:
            card = ProjectCard(project_path)
            card.clicked.connect(self.select_project)
            self.projects_layout.addWidget(card)
        
        self.projects_layout.addStretch()
    
    def get_recent_projects(self) -> List[str]:
        """Get list of recent project paths."""
        recent_projects = []
        legacy_projects = []
        
        # Look for projects in common locations
        search_paths = [
            Path.cwd(),
            Path("C:/Users/Michel/AI_Vision_Projects"),  # User's specific legacy location
            Path("C:/Users/Michel/AI_Vision_Projects"),  # Your specific legacy location
            Path.home() / "AI_Projects",
            Path.home() / "Documents" / "AI_Projects",
            Path.home() / "AI_Vision_Projects"  # Alternative location
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for item in search_path.iterdir():
                    if item.is_dir():
                        # Check for new project structure
                        is_new_valid, _ = validate_project_structure(item)
                        if is_new_valid:
                            recent_projects.append(str(item))
                        # Check for legacy project structure
                        elif detect_legacy_project(item):
                            legacy_projects.append(str(item))
        
        # Combine new and legacy projects, with new projects first
        all_projects = recent_projects + legacy_projects
        
        return all_projects[:15]  # Limit to 15 most recent
    
    def select_project(self, project_path: str):
        """Select a project."""
        # Check if this is a legacy project and offer migration
        if detect_legacy_project(Path(project_path)):
            reply = QMessageBox.question(
                self, "Legacy Project Detected",
                f"This appears to be an older project format.\n\n"
                f"Would you like to migrate it to the new format?\n"
                f"(This is safe and won't delete any data)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    if migrate_legacy_project(Path(project_path)):
                        QMessageBox.information(
                            self, "Migration Successful",
                            "Project has been successfully migrated to the new format!"
                        )
                    else:
                        QMessageBox.warning(
                            self, "Migration Failed",
                            "Could not migrate project. You can still use it, but some features may be limited."
                        )
                except Exception as e:
                    QMessageBox.critical(
                        self, "Migration Error",
                        f"Error during migration: {str(e)}"
                    )
        
        self.selected_project_path = project_path
        self.ok_button.setEnabled(True)
        
        # Highlight selected card
        for i in range(self.projects_layout.count()):
            item = self.projects_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, 'project_path'):
                    if widget.project_path == project_path:
                        widget.setStyleSheet("""
                            ProjectCard {
                                background-color: #e3f2fd;
                                border: 2px solid #2196F3;
                                border-radius: 8px;
                            }
                        """)
                    else:
                        widget.setStyleSheet("""
                            ProjectCard {
                                background-color: white;
                                border: 2px solid #e0e0e0;
                                border-radius: 8px;
                            }
                            ProjectCard:hover {
                                border-color: #2196F3;
                                background-color: #f5f5f5;
                            }
                        """)
    
    def create_new_project(self):
        """Create a new project."""
        dialog = NewProjectDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.selected_project_path = dialog.project_path
            self.accept()
    
    def open_existing_project(self):
        """Open an existing project."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Project Directory"
        )
        if directory:
            is_valid, error_msg = validate_project_structure(Path(directory))
            if is_valid:
                self.selected_project_path = directory
                self.accept()
            else:
                QMessageBox.warning(
                    self, "Invalid Project", 
                    f"Selected directory is not a valid project:\n{error_msg}"
                )
    
    def get_selected_project(self) -> Optional[str]:
        """Get the selected project path."""
        return self.selected_project_path

class WorkflowStatusWidget(QWidget):
    """Widget for displaying workflow status."""
    
    def __init__(self, workflow_manager, parent=None):
        super().__init__(parent)
        self.workflow_manager = workflow_manager
        self.init_ui()
    
    def init_ui(self):
        """Initialize the status widget UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Workflow Status")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Status items
        from .workflow import WorkflowStep
        
        for step in WorkflowStep:
            step_widget = QWidget()
            step_layout = QHBoxLayout(step_widget)
            
            # Status indicator
            status_label = QLabel("✓" if self.workflow_manager.is_step_completed(step) else "○")
            status_label.setFixedWidth(20)
            step_layout.addWidget(status_label)
            
            # Step name
            name_label = QLabel(step.value.replace('_', ' ').title())
            step_layout.addWidget(name_label)
            
            step_layout.addStretch()
            layout.addWidget(step_widget)
        
        layout.addStretch()

class ContinualTrainingDialog(QDialog):
    """Dialog for continual training setup."""
    
    def __init__(self, project_manager, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.setWindowTitle("Continual Learning")
        self.setModal(True)
        self.setFixedSize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Info
        info_label = QLabel(
            "Continual Learning allows you to update your trained model with new data "
            "without starting training from scratch. This is useful when you have "
            "collected additional labeled images or want to improve model performance."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Current model info
        model_info = QLabel("Loading model information...")
        layout.addWidget(model_info)
        
        # New data source
        layout.addWidget(QLabel("Source for additional training data:"))
        
        source_layout = QHBoxLayout()
        self.source_input = QLineEdit()
        source_button = QPushButton("Browse")
        source_button.clicked.connect(self.browse_source)
        source_layout.addWidget(self.source_input)
        source_layout.addWidget(source_button)
        layout.addLayout(source_layout)
        
        # Training parameters
        layout.addWidget(QLabel("Additional epochs:"))
        from PyQt6.QtWidgets import QSpinBox
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(10)
        layout.addWidget(self.epochs_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        start_button = QPushButton("Start Continual Training")
        start_button.clicked.connect(self.start_training)
        button_layout.addWidget(start_button)
        
        layout.addLayout(button_layout)
        
        # Load current model info
        self.load_model_info(model_info)
    
    def load_model_info(self, info_label):
        """Load information about the current model."""
        try:
            latest_model = self.project_manager.get_latest_model_path()
            if latest_model:
                info_text = f"Current model: {latest_model.name}\nLocation: {latest_model.parent}"
            else:
                info_text = "No trained model found. Please complete initial training first."
            info_label.setText(info_text)
        except Exception as e:
            info_label.setText(f"Error loading model info: {e}")
    
    def browse_source(self):
        """Browse for additional training data source."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Additional Training Data"
        )
        if directory:
            self.source_input.setText(directory)
    
    def start_training(self):
        """Start continual training."""
        source_dir = self.source_input.text().strip()
        epochs = self.epochs_input.value()
        
        if not source_dir:
            QMessageBox.warning(self, "Error", "Please select a source directory.")
            return
        
        try:
            # Here you would implement the actual continual training logic
            # For now, just show a message
            QMessageBox.information(
                self, "Training Started", 
                f"Continual training started with {epochs} additional epochs."
            )
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to start continual training: {str(e)}"
            )