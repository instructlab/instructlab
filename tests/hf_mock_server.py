# Standard
from http.server import BaseHTTPRequestHandler


class HuggingFaceTestServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
