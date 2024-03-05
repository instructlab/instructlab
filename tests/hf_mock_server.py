from http.server import BaseHTTPRequestHandler, HTTPServer

serverPort = 8080

class HuggingFaceTestServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
       

def HFTestServer():
    return HTTPServer(("localhost", serverPort), HuggingFaceTestServer)

if __name__ == "__main__":        
    webServer = HTTPServer(("localhost", serverPort), HuggingFaceTestServer)
    print("Server started http://%s:%s" % ("localhost", serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")