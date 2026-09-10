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
    print("==================================================================")
    print("  Facial Expression Recognition - Real-Time Inference Server")
    print(f"  Server URL:  http://localhost:{args.port}")
    print(f"  Network URL: http://{args.host}:{args.port}")
    print(f"  Mode:        {'Demo Simulation Mode (--demo)' if camera_demo else 'Live Webcam (Auto-Fallback to Demo if absent)'}")
    print("==================================================================")
    app.run(host=args.host, port=args.port, debug=args.debug)
