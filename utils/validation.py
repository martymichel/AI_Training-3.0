"""Validation utilities for the YOLO trainer."""

import os
import yaml
import torch
from typing import Tuple, Optional
import logging

# Logger konfigurieren
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def validate_yaml(path: str) -> Tuple[bool, Optional[str]]:
    """Validate YAML dataset file.
    
    Args:
        path (str): Path to YAML file
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not os.path.exists(path):
        logger.error(f"YAML file not found: {path}")
        return False, f"File not found: {path}"
        
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        required_keys = ['path', 'train', 'val', 'names']
        missing = [k for k in required_keys if k not in data]
        if missing:
            error_msg = f"Missing required keys in YAML: {', '.join(missing)}"
            logger.error(error_msg)
            return False, error_msg
            
        logger.info("YAML validation successful")
        return True, None
        
    except Exception as e:
        error_msg = f"Error parsing YAML: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def check_gpu() -> Tuple[bool, str]:
    """Check GPU availability and CUDA status.
    
    Returns:
        Tuple[bool, str]: (gpu_available, message)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - using CPU")
        return False, "CUDA not available. Training will use CPU (slow)."
        
    try:
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        message = f"Using GPU: {gpu_name} ({memory:.1f}GB)"
        logger.info(message)
        return True, message
        
    except Exception as e:
        error_msg = f"Error checking GPU: {str(e)}"
        logger.error(error_msg)
        return False, error_msg