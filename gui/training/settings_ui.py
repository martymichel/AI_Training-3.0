"""UI creation functions for the training settings panel."""

# This file is no longer needed as all UI creation is now handled in settings_window.py
# This prevents import/circular dependency issues and ensures proper initialization order

# Legacy compatibility - redirect imports to main window
def create_settings_ui(window):
    """Legacy function for backwards compatibility."""
    # All settings UI creation is now handled directly in TrainSettingsWindow.init_ui()
    # This function exists only for compatibility but does nothing
    pass

def update_model_options(window):
    """Legacy function for backwards compatibility."""
    # Model options update is now handled in TrainSettingsWindow.update_model_options()
    if hasattr(window, 'update_model_options'):
        window.update_model_options()

def create_basic_settings(window):
    """Legacy function for backwards compatibility."""
    pass

def create_advanced_settings(window):
    """Legacy function for backwards compatibility."""
    pass

def browse_project(window):
    """Legacy function for backwards compatibility."""
    if hasattr(window, 'browse_project'):
        window.browse_project()

def browse_data(window):
    """Legacy function for backwards compatibility."""
    if hasattr(window, 'browse_data'):
        window.browse_data()