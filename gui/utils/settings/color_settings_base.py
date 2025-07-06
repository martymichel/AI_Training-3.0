from enum import Enum
import json


class ColorSettingsBase:
    """Base color settings"""
    
    class WHITE_BALANCE(Enum):
        OFF = "Off"
        ONCE = "Once"
        CONTINUOUS = "Continuous"
    
    def __init__(self):
        self.blue_gain = None
        self.green_gain = None
        self.red_gain = None
        self.white_balance = None
    
    def from_json(self, data):
        """Load settings from JSON"""
        if isinstance(data, dict):
            self.blue_gain = data.get('BlueGain')
            self.green_gain = data.get('GreenGain')
            self.red_gain = data.get('RedGain')
            self.white_balance = data.get('WhiteBalance')
        return self
    
    def to_json(self):
        """Convert to JSON dict"""
        result = {}
        if self.blue_gain is not None:
            result['BlueGain'] = self.blue_gain
        if self.green_gain is not None:
            result['GreenGain'] = self.green_gain
        if self.red_gain is not None:
            result['RedGain'] = self.red_gain
        if self.white_balance is not None:
            result['WhiteBalance'] = self.white_balance
        return result