import json
from enum import Enum
from .stream_parent import StreamParent


class Stream3(StreamParent):
    """Stream3 configuration"""

    def __init__(self):
        super().__init__()

    def from_json(self, data: json):
        self.EncodingType = data.get("EncodingType")
        self.Framerate = data.get("Framerate")
        self.H26xBitrateMode = data.get("H26xBitrateMode")
        self.H26xKeyFrameInterval = data.get("H26xKeyFrameInterval")
        self.H26xTargetBitrate = data.get("H26xTargetBitrate")
        self._Height = data.get("Height")
        self.MJPEGQuality = data.get("MJPEGQuality")
        self._ReadOnly = data.get("ReadOnly")
        self.Resolution = data.get("Resolution")
        self._URL = data.get("URL")
        self._Width = data.get("Width")
        return self