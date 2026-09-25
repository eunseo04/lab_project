from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok"}'
        elif self.path == "/internal/status":
            body = b'{"service":"transaction-operations","environment":"internal"}'
        else:
            body = b'{"error":"not_found"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *_):
        pass

HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
