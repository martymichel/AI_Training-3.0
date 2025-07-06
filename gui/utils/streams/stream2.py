import json
from enum import Enum
from utils.streams.stream_parent import StreamParent


class Stream2(StreamParent):
    """Stream2 configuration"""

    def __init__(self):
        super().__init__()
        self.AutoOverlay: bool = None

    def from_json(self, data: json):
        self.AutoOverlay = data.get("AutoOverlay")
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