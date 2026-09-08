#!/usr/bin/env python3
"""Static file server for the dashboard, pinned to a fixed directory.

python -m http.server cannot be used: its --directory default is os.getcwd()."""
import http.server
import os
import socketserver

PORT = 8077
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path.split("?")[0] == "/reset-memory":
            import json
            body = b'{"ok":true}'
            try:
                with open(os.path.join(DIRECTORY, "shared", "customers.json"),
                          "w") as f:
                    json.dump({}, f)
            except OSError as e:
                body = ('{"ok":false,"error":%r}' % str(e)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == "__main__":
    print("Dashboard: http://localhost:%d/dashboard.html  (dir: %s)"
          % (PORT, DIRECTORY))
    with Server(("", PORT), Handler) as httpd:
        httpd.serve_forever()
