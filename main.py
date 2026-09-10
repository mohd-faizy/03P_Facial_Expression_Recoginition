import os
import sys
import argparse
from flask import Flask, render_template, Response

# Ensure project root is in sys.path for clean src package imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.camera import VideoCamera

# Configure application paths relative to project root
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# Global camera instance (lazily initialized)
camera_instance = None
camera_source = None
camera_demo = False


def get_camera():
    global camera_instance
    if camera_instance is None:
        camera_instance = VideoCamera(source=camera_source, demo=camera_demo)
    return camera_instance


@app.route('/')
def index():
    """Renders the main dashboard template."""
    return render_template('index.html')


def generate_frames(camera):
    """Yields multipart JPEG video frames for the web stream."""
    while True:
        frame_bytes = camera.get_frame()
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route producing multipart HTTP responses."""
    return Response(
        generate_frames(get_camera()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


import socket


def find_available_port(host, preferred_port):
    """Checks if preferred_port is available; if not, finds the next open port."""
    test_host = '127.0.0.1' if host in ('0.0.0.0', '') else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((test_host, preferred_port))
            return preferred_port
        except OSError:
            pass

    for candidate_port in range(5001, 5050):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((test_host, candidate_port))
                return candidate_port
            except OSError:
                continue
    return preferred_port


def parse_args():
    parser = argparse.ArgumentParser(description="Facial Expression Recognition Flask Streaming Server")
    parser.add_argument('--source', default=None, help="Video source: camera index (e.g. 0) or video file path")
    parser.add_argument('--demo', action='store_true', help="Force demo simulation mode without webcam")
    parser.add_argument('--host', default='0.0.0.0', help="Server bind host (default: 0.0.0.0)")
    parser.add_argument('--port', type=int, default=5000, help="Server bind port (default: 5000)")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    camera_source = args.source
    camera_demo = args.demo

    # Auto-detect if port is in use (common on macOS due to AirPlay Receiver on port 5000)
    actual_port = find_available_port(args.host, args.port)
    if actual_port != args.port:
        print(f"[INFO] Port {args.port} is already in use (e.g., macOS AirPlay Receiver).")
        print(f"[INFO] Automatically switching to available port {actual_port}.")

    print("==================================================================")
    print("  Facial Expression Recognition - Real-Time Inference Server")
    print(f"  Server URL:  http://localhost:{actual_port}")
    print(f"  Network URL: http://{args.host}:{actual_port}")
    print(f"  Mode:        {'Demo Simulation Mode (--demo)' if camera_demo else 'Live Webcam (Auto-Fallback to Demo if absent)'}")
    print("==================================================================")
    app.run(host=args.host, port=actual_port, debug=args.debug)

