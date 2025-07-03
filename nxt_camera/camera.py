"""NXT camera implementation."""

import cv2
import numpy as np
import requests
import logging
from urllib3.exceptions import InsecureRequestWarning
import warnings

# Suppress only the InsecureRequestWarning from urllib3
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

class NxtCamera:
    """Class for interfacing with IDS NXT camera."""

    def __init__(self, ip, username, password, ssl=True):
        """Initialize NXT camera connection.

        Args:
            ip (str): Camera IP address
            username (str): Authentication username
            password (str): Authentication password
            ssl (bool): Use SSL for connection
        """
        self.ip = ip
        self.auth = (username, password)
        self.protocol = "https" if ssl else "http"
        self.connected = False
        self.logger = logging.getLogger(__name__)

        self.base_url = None  # Determined during connect
        self.session = requests.Session()

        # Test connection
        self.connect()
    
    def connect(self):
        """Establish connection to camera.

        The exact REST endpoint can differ between firmware versions. We try
        several common variants used by the official `nxt-python-api` project
        until one succeeds.
        """

        endpoints = ["/api/info", "/api/v1/info", "/info"]
        last_error = None

        for ep in endpoints:
            url = f"{self.protocol}://{self.ip}{ep}"
            try:
                resp = self.session.get(
                    url,
                    auth=self.auth,
                    verify=False,
                    timeout=5,
                )
                if resp.status_code == 404:
                    last_error = f"404 for {url}"
                    continue
                resp.raise_for_status()
                self.connected = True
                self.base_url = url.rsplit("/", 1)[0]
                self.logger.info("Successfully connected to NXT camera")
                return
            except Exception as e:
                last_error = str(e)

        self.connected = False
        self.logger.error(
            f"Failed to connect to NXT camera: {last_error}"
        )
        raise RuntimeError("Unable to establish connection to NXT camera")
    
    def get_frame(self):
        """Get current frame from camera.
        
        Returns:
            numpy.ndarray: BGR image array
        """
        if not self.connected:
            raise RuntimeError("Camera not connected")
            
        try:
            url = f"{self.protocol}://{self.ip}/api/image"
            response = requests.get(
                url,
                auth=self.auth,
                verify=False,
                stream=True
            )
            response.raise_for_status()
            
            # Convert response to image
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise RuntimeError("Failed to decode image from camera")
                
            return frame
            
        except Exception as e:
            self.logger.error(f"Failed to get frame from camera: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from camera."""
        self.connected = False
        try:
            self.session.close()
        except Exception:
            pass
        self.logger.info("Disconnected from NXT camera")