"""Project configuration and data classes."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProjectConfig:
    """Configuration for a project."""
    project_name: str
    project_root: str
    created_date: str
    last_modified: str
    description: str = ""
    author: str = ""
    version: str = "1.0"
    classes: Dict[int, str] = None
    class_colors: Dict[int, str] = None
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.classes is None:
            self.classes = {}
        if self.class_colors is None:
            self.class_colors = {}
    
    @classmethod
    def create_new(cls, project_name: str, project_root: str, description: str = "", author: str = ""):
        """Create a new project configuration."""
        now = datetime.now().isoformat()
        return cls(
            project_name=project_name,
            project_root=str(project_root),
            created_date=now,
            last_modified=now,
            description=description,
            author=author,
            classes={},
            class_colors={}
        )
    
    @classmethod
    def from_file(cls, config_path: Path):
        """Load configuration from file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert string keys back to int for classes and class_colors
            if 'classes' in data and isinstance(data['classes'], dict):
                data['classes'] = {int(k): v for k, v in data['classes'].items()}
            if 'class_colors' in data and isinstance(data['class_colors'], dict):
                data['class_colors'] = {int(k): v for k, v in data['class_colors'].items()}
            
            return cls(**data)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def save_to_file(self, config_path: Path):
        """Save configuration to file."""
        try:
            # Update last modified
            self.last_modified = datetime.now().isoformat()
            
            # Convert int keys to string for JSON serialization
            data = asdict(self)
            if 'classes' in data and isinstance(data['classes'], dict):
                data['classes'] = {str(k): v for k, v in data['classes'].items()}
            if 'class_colors' in data and isinstance(data['class_colors'], dict):
                data['class_colors'] = {str(k): v for k, v in data['class_colors'].items()}
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise
    
    def update_classes(self, classes: Dict[int, str], class_colors: Dict[int, str] = None):
        """Update class information."""
        self.classes = classes.copy()
        if class_colors:
            self.class_colors = class_colors.copy()
        self.last_modified = datetime.now().isoformat()