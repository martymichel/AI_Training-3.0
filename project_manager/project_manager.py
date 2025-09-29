"""Main project manager class."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import json

from .config import ProjectConfig
from .workflow import WorkflowStep, WorkflowManager
from .utils import (
    get_recommended_model_type, get_default_model_path, load_json_settings,
    save_json_settings, find_latest_model, get_next_experiment_name,
    create_project_structure, validate_project_structure, detect_legacy_project,
    migrate_legacy_project
)

logger = logging.getLogger(__name__)

class ProjectManager:
    """Main class for managing AI Vision projects."""
    
    def __init__(self, project_root: str):
        """Initialize project manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        
        # Check if this is a legacy project and migrate if needed
        if detect_legacy_project(self.project_root):
            logger.info(f"Detected legacy project: {self.project_root.name}")
            
            # Try to migrate automatically
            if migrate_legacy_project(self.project_root):
                logger.info(f"Successfully migrated legacy project: {self.project_root.name}")
            else:
                logger.warning(f"Could not migrate legacy project, using compatibility mode")
                # Create minimal config for compatibility
                self._create_minimal_config()
        
        # Validate project structure (now should be valid after migration)
        is_valid, error_msg = validate_project_structure(self.project_root)
        if not is_valid:
            # Try to create missing structure
            try:
                create_project_structure(self.project_root)
                self._create_minimal_config()
            except Exception as e:
                raise ValueError(f"Invalid project structure and could not fix: {error_msg}")
        
        # Load configuration
        config_file = self.project_root / ".project" / "config.json"
        self.config = ProjectConfig.from_file(config_file)
        
        # Initialize workflow manager
        self.workflow_manager = WorkflowManager(self.project_root)
        
        logger.info(f"Project manager initialized for: {self.config.project_name}")
    
    def _create_minimal_config(self):
        """Create minimal config for legacy projects."""
        try:
            project_dir = self.project_root / ".project"
            project_dir.mkdir(exist_ok=True)
            
            config_data = {
                "project_name": self.project_root.name,
                "project_root": str(self.project_root),
                "created_date": "2024-01-01T00:00:00",
                "last_modified": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "description": "Legacy project (automatically migrated)",
                "author": "",
                "version": "1.0",
                "classes": {},
                "class_colors": {}
            }
            
            config_file = project_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error creating minimal config: {e}")
    
    def save_config(self):
        """Save current configuration."""
        config_file = self.project_root / ".project" / "config.json"
        self.config.save_to_file(config_file)
    
    # ==================== DIRECTORY MANAGEMENT ====================
    
    def get_raw_images_dir(self) -> Path:
        """Get raw images directory."""
        return self.project_root / "01_raw_images"
    
    def get_labeled_dir(self) -> Path:
        """Get labeled data directory."""
        return self.project_root / "02_labeled"
    
    def get_augmented_dir(self) -> Path:
        """Get augmented data directory."""
        return self.project_root / "03_augmented"
    
    def get_split_dir(self) -> Path:
        """Get split dataset directory."""
        return self.project_root / "04_splitted"
    
    def get_models_dir(self) -> Path:
        """Get models directory."""
        return self.project_root / "05_models"
    
    def get_verification_dir(self) -> Path:
        """Get verification results directory."""
        return self.project_root / "06_verification"
    
    def get_detection_dir(self) -> Path:
        """Get live detection directory."""
        return self.project_root / "07_live_detection"
    
    def get_crops_dir(self) -> Path:
        """Get crops directory for segmentation."""
        return self.project_root / "08_crops_for_segmentation"
    
    # ==================== CLASS MANAGEMENT ====================
    
    def add_class(self, class_id: int, class_name: str, color: str = "#FF0000"):
        """Add a new class to the project."""
        self.config.classes[class_id] = class_name
        self.config.class_colors[class_id] = color
        self.save_config()
        logger.info(f"Added class {class_id}: {class_name}")
    
    def get_classes(self) -> Dict[int, str]:
        """Get all classes in the project."""
        return self.config.classes.copy()
    
    def get_class_colors(self) -> Dict[int, str]:
        """Get class colors."""
        return self.config.class_colors.copy()
    
    def update_classes(self, classes: Dict[int, str], colors: Dict[int, str] = None):
        """Update all classes at once."""
        self.config.classes = classes.copy()
        if colors:
            self.config.class_colors = colors.copy()
        self.save_config()
    
    # ==================== MODEL MANAGEMENT ====================
    
    def get_recommended_model_type(self) -> str:
        """Get recommended model type based on annotations."""
        return get_recommended_model_type(self.project_root)
    
    def get_default_model_path(self, model_type: str = None) -> str:
        """Get default model path."""
        if model_type is None:
            model_type = self.get_recommended_model_type()
        return get_default_model_path(model_type)
    
    def get_latest_model_path(self) -> Optional[Path]:
        """Get path to latest trained model."""
        return find_latest_model(self.get_models_dir())
    
    def get_current_model_path(self) -> Optional[Path]:
        """Get current active model path."""
        # Could be implemented to track "active" model
        return self.get_latest_model_path()
    
    def get_next_experiment_name(self) -> str:
        """Get next experiment name for training."""
        return get_next_experiment_name(self.get_models_dir())
    
    def get_last_experiment_name(self) -> str:
        """Get the name of the last experiment (most recent)."""
        try:
            models_dir = self.get_models_dir()
            if not models_dir.exists():
                return "experiment_001"
            
            # Find all experiment directories
            experiment_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
            if not experiment_dirs:
                return "experiment_001"
            
            # Sort by modification time (most recent first)
            experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return experiment_dirs[0].name
        
        except Exception as e:
            logger.error(f"Error getting last experiment name: {e}")
            return "experiment_001"
    
    # ==================== YAML MANAGEMENT ====================
    
    def get_yaml_file(self) -> Path:
        """Get dataset YAML file path."""
        return self.get_split_dir() / "data.yaml"
    
    def set_yaml_path(self, yaml_path: str):
        """Set YAML path (for compatibility)."""
        # This could be used to track custom YAML locations
        logger.info(f"YAML path set to: {yaml_path}")
    
    # ==================== SETTINGS MANAGEMENT ====================
    
    def get_augmentation_settings(self) -> Dict:
        """Get augmentation settings."""
        settings_file = self.project_root / ".project" / "augmentation_settings.json"
        return load_json_settings(settings_file, {
            'methods': {},
            'flip_settings': {'horizontal': False, 'vertical': False}
        })
    
    def update_augmentation_settings(self, settings: Dict):
        """Update augmentation settings."""
        settings_file = self.project_root / ".project" / "augmentation_settings.json"
        save_json_settings(settings_file, settings)
    
    def get_training_settings(self) -> Dict:
        """Get training settings."""
        settings_file = self.project_root / ".project" / "training_settings.json"
        return load_json_settings(settings_file)
    
    def update_training_settings(self, settings: Dict):
        """Update training settings."""
        settings_file = self.project_root / ".project" / "training_settings.json"
        save_json_settings(settings_file, settings)
    
    def get_live_detection_settings(self) -> Dict:
        """Get live detection settings."""
        settings_file = self.project_root / "detection_settings.json"
        return load_json_settings(settings_file, {
            'model_path': '',
            'yaml_path': '',
            'motion_threshold': 110,
            'iou_threshold': 0.45,
            'class_thresholds': {},
            'enabled': False
        })
    
    def update_live_detection_settings(self, settings: Dict):
        """Update live detection settings."""
        settings_file = self.project_root / "detection_settings.json"
        current_settings = self.get_live_detection_settings()
        current_settings.update(settings)
        save_json_settings(settings_file, current_settings)
    
    # ==================== WORKFLOW DELEGATION ====================
    
    def mark_step_completed(self, step: WorkflowStep):
        """Mark a workflow step as completed."""
        self.workflow_manager.mark_step_completed(step)
    
    def is_step_completed(self, step: WorkflowStep) -> bool:
        """Check if a workflow step is completed."""
        return self.workflow_manager.is_step_completed(step)
    
    def validate_workflow_step(self, step: WorkflowStep) -> Tuple[bool, str]:
        """Validate if a workflow step can be executed."""
        return self.workflow_manager.validate_workflow_step(step)
    
    def get_workflow_progress(self) -> Tuple[int, int]:
        """Get workflow progress."""
        return self.workflow_manager.get_workflow_progress()
    
    def reset_workflow(self):
        """Reset workflow status."""
        self.workflow_manager.reset_workflow()
    
    def reset_from_step(self, step: WorkflowStep):
        """Reset workflow from specific step."""
        self.workflow_manager.reset_from_step(step)
    
    # ==================== UTILITY METHODS ====================
    
    def get_project_summary(self) -> Dict[str, Any]:
        """Get comprehensive project summary."""
        completed, total = self.get_workflow_progress()
        
        return {
            'name': self.config.project_name,
            'root': str(self.project_root),
            'created': self.config.created_date,
            'modified': self.config.last_modified,
            'description': self.config.description,
            'author': self.config.author,
            'classes': len(self.config.classes),
            'workflow_progress': f"{completed}/{total}",
            'recommended_model_type': self.get_recommended_model_type()
        }
    
    def export_settings(self, export_path: Path):
        """Export all project settings to a file."""
        try:
            settings = {
                'config': self.config.__dict__,
                'augmentation_settings': self.get_augmentation_settings(),
                'training_settings': self.get_training_settings(),
                'detection_settings': self.get_live_detection_settings(),
                'workflow_status': list(self.workflow_manager.completed_steps)
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Settings exported to: {export_path}")
        
        except Exception as e:
            logger.error(f"Error exporting settings: {e}")
            raise
    
    def import_settings(self, import_path: Path):
        """Import project settings from a file."""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # Import configuration
            if 'config' in settings:
                config_data = settings['config']
                # Convert string keys back to int for classes
                if 'classes' in config_data:
                    config_data['classes'] = {int(k): v for k, v in config_data['classes'].items()}
                if 'class_colors' in config_data:
                    config_data['class_colors'] = {int(k): v for k, v in config_data['class_colors'].items()}
                
                self.config = ProjectConfig(**config_data)
                self.save_config()
            
            # Import other settings
            if 'augmentation_settings' in settings:
                self.update_augmentation_settings(settings['augmentation_settings'])
            
            if 'training_settings' in settings:
                self.update_training_settings(settings['training_settings'])
            
            if 'detection_settings' in settings:
                self.update_live_detection_settings(settings['detection_settings'])
            
            # Import workflow status
            if 'workflow_status' in settings:
                self.workflow_manager.completed_steps = set(settings['workflow_status'])
                self.workflow_manager.save_status()
            
            logger.info(f"Settings imported from: {import_path}")
        
        except Exception as e:
            logger.error(f"Error importing settings: {e}")
            raise