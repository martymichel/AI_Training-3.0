"""Utility functions for project management."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)

def detect_annotation_format(label_files: List[Path]) -> str:
    """Detect annotation format from label files.
    
    Args:
        label_files: List of label file paths
        
    Returns:
        str: 'bbox', 'polygon', 'mixed', or 'unknown'
    """
    has_bbox = False
    has_polygon = False
    
    for label_path in label_files[:20]:  # Check first 20 files for efficiency
        try:
            if not label_path.exists():
                continue
                
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    continue
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    if len(parts) == 5:
                        has_bbox = True
                    elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                        has_polygon = True
                    
                    # Early exit if both found
                    if has_bbox and has_polygon:
                        return 'mixed'
        except Exception as e:
            logger.warning(f"Error reading {label_path}: {e}")
            continue
    
    if has_polygon:
        return 'polygon'
    elif has_bbox:
        return 'bbox'
    else:
        return 'unknown'

def get_recommended_model_type(project_root: Path) -> str:
    """Get recommended model type based on project annotations.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        str: 'detection' or 'segmentation'
    """
    try:
        labeled_dir = project_root / "02_labeled"
        if not labeled_dir.exists():
            return "detection"
        
        label_files = list(labeled_dir.glob("*.txt"))
        if not label_files:
            return "detection"
        
        annotation_format = detect_annotation_format(label_files)
        logger.info(f"Detected annotation type: {annotation_format}")
        
        if annotation_format in ['polygon', 'mixed']:
            return "segmentation"
        else:
            return "detection"
    
    except Exception as e:
        logger.warning(f"Error detecting model type: {e}")
        return "detection"

def get_default_model_path(model_type: str = "detection") -> str:
    """Get default model path based on type.
    
    Args:
        model_type: 'detection' or 'segmentation'
        
    Returns:
        str: Default model filename
    """
    if model_type.lower() == "segmentation":
        return "yolo11n-seg.pt"
    else:
        return "yolo11n.pt"

def load_json_settings(file_path: Path, default: Dict = None) -> Dict:
    """Load JSON settings with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default dict to return on error
        
    Returns:
        Dict: Loaded settings or default
    """
    if default is None:
        default = {}
    
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading settings from {file_path}: {e}")
    
    return default.copy()

def save_json_settings(file_path: Path, settings: Dict):
    """Save JSON settings with error handling.
    
    Args:
        file_path: Path to JSON file
        settings: Settings dictionary to save
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving settings to {file_path}: {e}")
        raise

def find_latest_model(models_dir: Path) -> Optional[Path]:
    """Find the latest trained model.
    
    Args:
        models_dir: Directory containing model subdirectories
        
    Returns:
        Optional[Path]: Path to latest model or None
    """
    try:
        if not models_dir.exists():
            return None
        
        model_files = list(models_dir.rglob("best.pt"))
        if not model_files:
            return None
        
        # Sort by modification time
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return model_files[0]
    
    except Exception as e:
        logger.error(f"Error finding latest model: {e}")
        return None

def get_next_experiment_name(models_dir: Path, base_name: str = "experiment") -> str:
    """Get next available experiment name.
    
    Args:
        models_dir: Directory containing experiments
        base_name: Base name for experiments
        
    Returns:
        str: Next available experiment name
    """
    try:
        if not models_dir.exists():
            return f"{base_name}_001"
        
        existing_dirs = [d.name for d in models_dir.iterdir() if d.is_dir()]
        
        # Find highest numbered experiment
        max_num = 0
        for dir_name in existing_dirs:
            if dir_name.startswith(base_name):
                try:
                    num_part = dir_name.replace(base_name, "").replace("_", "")
                    if num_part.isdigit():
                        max_num = max(max_num, int(num_part))
                except Exception:
                    continue
        
        return f"{base_name}_{max_num + 1:03d}"
    
    except Exception as e:
        logger.error(f"Error getting next experiment name: {e}")
        return f"{base_name}_001"

def create_project_structure(project_root: Path):
    """Create standard project directory structure.
    
    Args:
        project_root: Root directory for the project
    """
    directories = [
        "01_raw_images",
        "02_labeled",
        "03_augmented", 
        "04_splitted",
        "05_models",
        "06_verification",
        "07_live_detection",
        "08_crops_for_segmentation",
        ".project"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def validate_project_structure(project_root: Path) -> Tuple[bool, str]:
    """Validate project directory structure.
    
    Args:
        project_root: Path to project root
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        project_root = Path(project_root)
        
        if not project_root.exists():
            return False, f"Project directory does not exist: {project_root}"
        
        if not project_root.is_dir():
            return False, f"Project path is not a directory: {project_root}"
        
        config_file = project_root / ".project" / "config.json"
        if not config_file.exists():
            return False, f"Project configuration not found: {config_file}"
        
        # Try to load and validate config
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            required_fields = ['project_name', 'project_root', 'created_date']
            missing_fields = [field for field in required_fields if field not in config_data]
            
            if missing_fields:
                return False, f"Missing required config fields: {', '.join(missing_fields)}"
        
        except Exception as e:
            return False, f"Invalid project configuration: {e}"
        
        return True, "Project structure is valid"
    
    except Exception as e:
        return False, f"Error validating project: {e}"

def get_project_info(project_root: Path) -> Dict:
    """Get basic project information.
    
    Args:
        project_root: Path to project root
        
    Returns:
        Dict: Project information
    """
    try:
        config_file = project_root / ".project" / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error reading project info: {e}")
    
    return {
        'project_name': project_root.name,
        'project_root': str(project_root),
        'created_date': 'unknown',
        'last_modified': 'unknown'
    }