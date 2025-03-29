"""Common utilities for the augmentation application."""

def get_method_key(method_name):
    """Convert German method name to English key."""
    method_map = {
        "Verschiebung": "Shift",
        "Rotation": "Rotate",
        "Zoom": "Zoom",
        "Helligkeit": "Brightness",
        "Unschärfe": "Blur"
    }
    return method_map.get(method_name, method_name)