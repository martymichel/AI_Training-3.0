import json


class CameraRoi:
    """Camera Region of Interest settings"""
    
    def __init__(self, x=0, y=0, width=1920, height=1080):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    @classmethod
    def from_json(cls, data):
        """Create CameraRoi from JSON data"""
        if isinstance(data, dict):
            return cls(
                x=data.get('x', 0),
                y=data.get('y', 0),
                width=data.get('width', 1920),
                height=data.get('height', 1080)
            )
        return cls()
    
    def to_json(self):
        """Convert to JSON dict"""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }