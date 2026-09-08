#!/usr/bin/env python3
# robot_view_server.py — runs on the HOST (Mac). Serves the TIAGo head-camera
# frames as a live MJPEG page in the browser, as an alternative to the Gazebo
# 3D GUI (gzclient), which deadlocks under amd64 emulation on Apple Silicon.
#
# The frame grabber inside the hrai_sim container writes robot_frame.jpg into
# the shared exchange/ dir; this server just streams that file over HTTP.
#
# USE:
#   python3 exchange/robot_view_server.py
#   then open  http://localhost:8080  in the browser.
import os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
# FRAME/PORT overridable via env so a second instance can serve the overhead
# (bird's-eye) camera: PORT=8081 FRAME=.../overhead_frame.jpg python3 robot_view_server.py
FRAME = os.environ.get("FRAME", os.path.join(HERE, "robot_frame.jpg"))
PORT = int(os.environ.get("PORT", "8080"))

LABEL = os.environ.get("VIEW_LABEL", "vista camera dalla testa del robot")
PAGE = ("""<!doctype html><html><head><meta charset=utf-8>
<title>TIAGo &mdash; %s</title>
<style>html,body{margin:0;height:100%%;background:#111;color:#ccc;font-family:sans-serif;text-align:center}
h3{margin:6px}
/* riempi la finestra: l'immagine si ingrandisce a tutta la larghezza/altezza disponibile */
img{width:98vw;max-height:92vh;height:auto;object-fit:contain;border:1px solid #333;
    image-rendering:auto}</style></head>
<body><h3>TIAGo &mdash; %s</h3>
<img src="/stream.mjpg"></body></html>""" % (LABEL, LABEL)).encode()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    try:
                        with open(FRAME, "rb") as f:
                            data = f.read()
                    except OSError:
                        data = None
                    if data:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            ("Content-Length: %d\r\n\r\n" % len(data)).encode())
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.25)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(404)
        self.end_headers()

if __name__ == "__main__":
    print("Robot view server su http://localhost:%d  (frame: %s)" % (PORT, FRAME))
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
