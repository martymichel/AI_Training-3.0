from enum import Enum
import json


class CameraSettingsBase:
    """Base camera settings"""
    
    class GAIN_AUTO(Enum):
        OFF = "Off"
        ONCE = "Once"
        CONTINUOUS = "Continuous"
    
    def __init__(self):
        self.exposure_time = None
        self.gain = None
        self.gain_auto = None
        self.flip_horizontal = None
        self.flip_vertical = None
        self.gamma_correction = None
    
    def from_json(self, data):
        """Load settings from JSON"""
        if isinstance(data, dict):
            self.exposure_time = data.get('ExposureTime')
            self.gain = data.get('Gain')
            self.gain_auto = data.get('GainAuto')
            self.flip_horizontal = data.get('FlipHorizontal')
            self.flip_vertical = data.get('FlipVertical')
            self.gamma_correction = data.get('GammaCorrection')
        return self
    
    def to_json(self):
        """Convert to JSON dict"""
        result = {}
        if self.exposure_time is not None:
            result['ExposureTime'] = self.exposure_time
        if self.gain is not None:
            result['Gain'] = self.gain
        if self.gain_auto is not None:
            result['GainAuto'] = self.gain_auto
        if self.flip_horizontal is not None:
            result['FlipHorizontal'] = self.flip_horizontal
        if self.flip_vertical is not None:
            result['FlipVertical'] = self.flip_vertical
        if self.gamma_correction is not None:
            result['GammaCorrection'] = self.gamma_correction
        return result