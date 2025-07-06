from enum import Enum
import json


class RestResult:
    """Represents the result of a REST API call"""
    
    class RequestType(Enum):
        GET = "GET"
        POST = "POST"
        PUT = "PUT"
        PATCH = "PATCH"
        DELETE = "DELETE"
        OPTIONS = "OPTIONS"
    
    def __init__(self, path: str, request_type: RequestType):
        self.path = path
        self.request_type = request_type
        self.response_data = None
        self.response_status = None
        self.response_header = None
    
    def to_json(self):
        """Convert response data to JSON"""
        if self.response_data:
            try:
                return json.loads(self.response_data.decode('utf-8'))
            except:
                return {}
        return {}
    
    def get_response_data(self):
        """Get raw response data"""
        return self.response_data