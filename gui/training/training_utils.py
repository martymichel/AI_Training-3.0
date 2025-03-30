"""Utility functions for the training module."""

import os
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger("training_utils")
logger.setLevel(logging.INFO)

def check_and_load_results_csv(project, experiment, last_check_time=0):
    """Check if results.csv exists and has been updated."""
    try:
        # Find results.csv in the experiment directory
        base_path = os.path.join(project, experiment)
        if not os.path.exists(base_path):
            return None
                
        results_path = None
        for root, dirs, files in os.walk(base_path):
            if "results.csv" in files:
                results_path = os.path.join(root, "results.csv")
                break
                
        if not results_path:
            return None
                
        # Check if the file has been modified
        mod_time = os.path.getmtime(results_path)
        if mod_time <= last_check_time:
            return None
                
        # Read results.csv
        try:
            df = pd.read_csv(results_path)
            # Store the file path for later use
            df.filepath = results_path
            return df
        except Exception as e:
            logger.error(f"Error reading results.csv: {e}")
            return None
                
    except Exception as e:
        logger.error(f"Error checking results.csv: {e}")
        return None