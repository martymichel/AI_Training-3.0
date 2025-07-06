from enum import Enum
import json


class TriggerSettingsBase:
    """Base trigger settings"""
    
    class TRIGGERTYPE(Enum):
        SOFTWARE = "Software"
        HARDWARE = "Hardware"
        FREERUN = "Freerun"
    
    class EDGE(Enum):
        RISING = "Rising"
        FALLING = "Falling"
    
    def __init__(self):
        self.trigger_type = None
        self.edge = None
        self.delay = None
        self.timeout = None
    
    def from_json(self, data):
        """Load settings from JSON"""
        if isinstance(data, dict):
            self.trigger_type = data.get('TriggerType')
            self.edge = data.get('Edge')
            self.delay = data.get('Delay')
            self.timeout = data.get('Timeout')
        return self
    
    def to_json(self):
        """Convert to JSON dict"""
        result = {}
        if self.trigger_type is not None:
            result['TriggerType'] = self.trigger_type
        if self.edge is not None:
            result['Edge'] = self.edge
        if self.delay is not None:
            result['Delay'] = self.delay
        if self.timeout is not None:
            result['Timeout'] = self.timeout
        return result