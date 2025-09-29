"""Workflow management for projects."""

from enum import Enum
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class WorkflowStep(Enum):
    """Enumeration of workflow steps."""
    CAMERA = "camera"
    LABELING = "labeling"
    AUGMENTATION = "augmentation"
    SPLITTING = "splitting"
    TRAINING = "training"
    VERIFICATION = "verification"
    LIVE_DETECTION = "live_detection"

class WorkflowManager:
    """Manages workflow steps and their dependencies."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.status_file = self.project_root / ".project" / "workflow_status.json"
        self.completed_steps = set()
        self.load_status()
    
    def load_status(self):
        """Load workflow status from file."""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                self.completed_steps = set(data.get('completed_steps', []))
        except Exception as e:
            logger.error(f"Error loading workflow status: {e}")
            self.completed_steps = set()
    
    def save_status(self):
        """Save workflow status to file."""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'completed_steps': list(self.completed_steps),
                'last_updated': str(Path(__file__).stat().st_mtime)
            }
            with open(self.status_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving workflow status: {e}")
    
    def mark_step_completed(self, step: WorkflowStep):
        """Mark a workflow step as completed."""
        self.completed_steps.add(step.value)
        self.save_status()
        logger.info(f"Workflow step completed: {step.value}")
    
    def is_step_completed(self, step: WorkflowStep) -> bool:
        """Check if a workflow step is completed."""
        return step.value in self.completed_steps
    
    def get_step_dependencies(self) -> Dict[WorkflowStep, List[WorkflowStep]]:
        """Get dependencies for each workflow step."""
        return {
            WorkflowStep.CAMERA: [],
            WorkflowStep.LABELING: [WorkflowStep.CAMERA],
            WorkflowStep.AUGMENTATION: [WorkflowStep.LABELING],
            WorkflowStep.SPLITTING: [WorkflowStep.LABELING],
            WorkflowStep.TRAINING: [WorkflowStep.SPLITTING],
            WorkflowStep.VERIFICATION: [WorkflowStep.TRAINING],
            WorkflowStep.LIVE_DETECTION: [WorkflowStep.VERIFICATION]
        }
    
    def validate_workflow_step(self, step: WorkflowStep) -> Tuple[bool, str]:
        """Validate if a workflow step can be executed."""
        dependencies = self.get_step_dependencies()
        
        if step not in dependencies:
            return False, f"Unknown workflow step: {step}"
        
        # Check if all dependencies are met
        for dep_step in dependencies[step]:
            if not self.is_step_completed(dep_step):
                return False, f"Dependency not met: {dep_step.value} must be completed first"
        
        # Additional validation based on actual project state
        try:
            if step == WorkflowStep.LABELING:
                raw_images_dir = self.project_root / "01_raw_images"
                if not raw_images_dir.exists() or not any(raw_images_dir.glob("*.jpg")):
                    return False, "No raw images found. Please capture images first."
            
            elif step == WorkflowStep.AUGMENTATION:
                labeled_dir = self.project_root / "02_labeled"
                if not labeled_dir.exists() or not any(labeled_dir.glob("*.jpg")):
                    return False, "No labeled images found. Please complete labeling first."
            
            elif step == WorkflowStep.SPLITTING:
                # Check for either labeled or augmented data
                labeled_dir = self.project_root / "02_labeled"
                augmented_dir = self.project_root / "03_augmented"
                
                has_labeled = labeled_dir.exists() and any(labeled_dir.glob("*.jpg"))
                has_augmented = augmented_dir.exists() and any(augmented_dir.glob("*.jpg"))
                
                if not (has_labeled or has_augmented):
                    return False, "No labeled or augmented data found. Please complete labeling first."
            
            elif step == WorkflowStep.TRAINING:
                yaml_file = self.project_root / "04_splitted" / "data.yaml"
                if not yaml_file.exists():
                    return False, "No dataset configuration found. Please complete dataset splitting first."
            
            elif step == WorkflowStep.VERIFICATION:
                models_dir = self.project_root / "05_models"
                if not models_dir.exists():
                    return False, "No models directory found. Please complete training first."
                
                # Check for any .pt files in subdirectories
                has_model = any(models_dir.rglob("*.pt"))
                if not has_model:
                    return False, "No trained models found. Please complete training first."
            
            elif step == WorkflowStep.LIVE_DETECTION:
                verification_dir = self.project_root / "06_verification"
                if not verification_dir.exists():
                    return False, "No verification results found. Please complete verification first."
        
        except Exception as e:
            logger.warning(f"Error validating step {step}: {e}")
            # Don't block workflow for validation errors
        
        return True, "Step can be executed"
    
    def get_next_available_step(self) -> Optional[WorkflowStep]:
        """Get the next workflow step that can be executed."""
        for step in WorkflowStep:
            if not self.is_step_completed(step):
                can_execute, _ = self.validate_workflow_step(step)
                if can_execute:
                    return step
        return None
    
    def get_workflow_progress(self) -> Tuple[int, int]:
        """Get workflow progress as (completed_steps, total_steps)."""
        total_steps = len(WorkflowStep)
        completed_steps = len(self.completed_steps)
        return completed_steps, total_steps
    
    def reset_workflow(self):
        """Reset all workflow steps."""
        self.completed_steps.clear()
        self.save_status()
        logger.info("Workflow reset - all steps marked as incomplete")
    
    def reset_from_step(self, step: WorkflowStep):
        """Reset workflow from a specific step onwards."""
        steps_to_reset = []
        step_order = list(WorkflowStep)
        
        if step in step_order:
            step_index = step_order.index(step)
            steps_to_reset = step_order[step_index:]
        
        for reset_step in steps_to_reset:
            if reset_step.value in self.completed_steps:
                self.completed_steps.remove(reset_step.value)
        
        self.save_status()
        logger.info(f"Workflow reset from step: {step.value}")