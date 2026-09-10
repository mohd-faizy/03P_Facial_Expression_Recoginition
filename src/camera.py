import os
import time
import cv2
import numpy as np

try:
    from src.model import FacialExpressionModel
except ImportError:
    try:
        from model import FacialExpressionModel
    except ImportError:
        FacialExpressionModel = None


def get_haar_cascade_path(project_root):
    """Locates haarcascade_frontalface_default.xml in models/, assets/, or from OpenCV built-in data."""
    local_name = "haarcascade_frontalface_default.xml"
    candidates = [
        os.path.join(project_root, "models", local_name),
        os.path.join(project_root, "assets", local_name),
        os.path.join(project_root, local_name),
        local_name,
    ]
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        candidates.append(os.path.join(cv2.data.haarcascades, local_name))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return local_name


# Project path configuration
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

cascade_path = get_haar_cascade_path(_PROJECT_ROOT)
face_cascade = cv2.CascadeClassifier(cascade_path)
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    """
    OpenCV video capture wrapper supporting webcam streams, local video files,
    and automatic demo simulation mode when no physical camera is detected.
    """

    def __init__(self, source=None, demo=False):
        self.project_root = _PROJECT_ROOT
        self.demo_mode = demo
        self.source = None
        self.video = None
        self.is_file = False
        self.demo_image = None
        self.demo_tick = 0
        self.tracked_faces = []  # Temporal tracking list: [{'box': (x,y,w,h), 'label': str, 'lost': int}]

        # Load model with automatic models/ directory weights discovery
        self.model = None
        if FacialExpressionModel is not None:
            try:
                self.model = FacialExpressionModel()
            except Exception as err:
                print(f"[INFO] Emotion model: {err}. Real-time face tracking active.")

        if not self.demo_mode:
            self.source = self._resolve_video_source(source)
            try:
                self.video = cv2.VideoCapture(self.source)
                if not self.video.isOpened():
                    print(f"[WARN] Video source '{self.source}' could not be opened. Enabling Demo Simulation Mode.")
                    self.demo_mode = True
                else:
                    self.is_file = isinstance(self.source, str) and os.path.exists(self.source)
            except Exception as e:
                print(f"[WARN] Error initializing VideoCapture: {e}. Enabling Demo Simulation Mode.")
                self.demo_mode = True

        if self.demo_mode:
            sample_img_path = os.path.join(self.project_root, 'assets', 'sample_face.jpg')
            if os.path.exists(sample_img_path):
                self.demo_image = cv2.imread(sample_img_path)
                print("[INFO] Running in Demo Mode using assets/sample_face.jpg")
            else:
                self.demo_image = np.zeros((480, 640, 3), dtype=np.uint8)

    def _resolve_video_source(self, source):
        """Resolves video source priority: explicit source -> local video -> Colab path -> webcam (0)."""
        if source is not None:
            return int(source) if str(source).isdigit() else source

        candidates = [
            os.path.join(self.project_root, "videos", "facial_exp.mkv"),
            os.path.join("videos", "facial_exp.mkv"),
            "/content/sample_data/Project/videos/facial_exp.mkv",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return 0

    def __del__(self):
        if self.video is not None and hasattr(self.video, 'isOpened') and self.video.isOpened():
            self.video.release()

    def get_frame(self):
        """
        Captures frame, runs face detection, predicts emotion per face crop,
        and returns JPEG bytes for multipart video streaming.
        """
        if self.demo_mode:
            time.sleep(0.04)  # Rate limit to ~25 FPS to prevent CPU spinning
            self.demo_tick += 1

            frame = self.demo_image.copy()
            h, w = frame.shape[:2]

            if w > 800 or h > 600:
                frame = cv2.resize(frame, (640, 640))

            frame = self._process_frame(frame, is_demo=True)
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        success, frame = self.video.read()

        if not success or frame is None:
            if self.is_file:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.video.read()

            if not success or frame is None:
                time.sleep(0.04)
                standby = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(standby, "Camera Offline - Reconnecting...", (100, 240), font, 0.8, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', standby)
                return jpeg.tobytes()

        frame = self._process_frame(frame, is_demo=False)
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    @staticmethod
    def _compute_iou(box1, box2):
        """Computes Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _process_frame(self, frame, is_demo=False):
        """Runs stabilized face detection and emotion classification on a frame."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_frame, w_frame = gray_frame.shape[:2]

        # Equalize histogram for contrast stability under changing camera lighting
        gray_eq = cv2.equalizeHist(gray_frame)
        raw_faces = face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(60, 60)
        )
        if len(raw_faces) == 0:
            raw_faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(60, 60)
            )

        updated_tracked = []
        raw_boxes = [tuple(int(v) for v in box) for box in raw_faces]
        matched_indices = set()

        # Match existing tracked faces to new detections to maintain continuity
        for tf in self.tracked_faces:
            best_iou = 0
            best_idx = None
            for idx, box in enumerate(raw_boxes):
                if idx in matched_indices:
                    continue
                iou = self._compute_iou(tf['box'], box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou > 0.25 and best_idx is not None:
                matched_indices.add(best_idx)
                # Exponential Moving Average (EMA) smoothing to eliminate bounding box jitter
                alpha = 0.65
                old_x, old_y, old_w, old_h = tf['box']
                new_x, new_y, new_w, new_h = raw_boxes[best_idx]
                smooth_box = (
                    int(alpha * new_x + (1 - alpha) * old_x),
                    int(alpha * new_y + (1 - alpha) * old_y),
                    int(alpha * new_w + (1 - alpha) * old_w),
                    int(alpha * new_h + (1 - alpha) * old_h),
                )

                label = tf['label']
                if self.model is not None:
                    sx, sy, sw, sh = smooth_box
                    cx1 = max(0, sx)
                    cy1 = max(0, sy)
                    cx2 = min(w_frame, sx + sw)
                    cy2 = min(h_frame, sy + sh)
                    if cx2 > cx1 and cy2 > cy1:
                        fc = gray_frame[cy1:cy2, cx1:cx2]
                        roi = cv2.resize(fc, (48, 48))
                        roi_tensor = roi[np.newaxis, :, :, np.newaxis]
                        try:
                            label = self.model.predict_emotion(roi_tensor)
                        except Exception:
                            pass

                updated_tracked.append({'box': smooth_box, 'label': label, 'lost': 0})
            else:
                # Face briefly missed by detector (motion blur, turn) -> persist for up to 8 frames (~0.3s)
                if tf['lost'] < 8:
                    tf['lost'] += 1
                    updated_tracked.append(tf)

        # Register any new faces detected in the current frame
        for idx, box in enumerate(raw_boxes):
            if idx in matched_indices:
                continue
            x, y, w, h = box
            label = "Face Detected"
            if self.model is not None:
                cx1 = max(0, x)
                cy1 = max(0, y)
                cx2 = min(w_frame, x + w)
                cy2 = min(h_frame, y + h)
                if cx2 > cx1 and cy2 > cy1:
                    fc = gray_frame[cy1:cy2, cx1:cx2]
                    roi = cv2.resize(fc, (48, 48))
                    roi_tensor = roi[np.newaxis, :, :, np.newaxis]
                    try:
                        label = self.model.predict_emotion(roi_tensor)
                    except Exception:
                        pass
            updated_tracked.append({'box': (int(x), int(y), int(w), int(h)), 'label': label, 'lost': 0})

        self.tracked_faces = updated_tracked

        # Draw stabilized bounding boxes and label badges
        for tf in self.tracked_faces:
            x, y, w, h = tf['box']
            label = tf['label']

            # Smooth cyan / golden bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 201, 76), 2)

            # Draw label banner background and text
            label_text = f" {label} "
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, 0.75, 2)
            banner_y1 = max(0, y - text_h - 10)
            banner_y2 = y
            cv2.rectangle(frame, (x, banner_y1), (x + text_w, banner_y2), (240, 201, 76), -1)
            cv2.putText(frame, label_text, (x, y - 5), font, 0.75, (20, 20, 20), 2, cv2.LINE_AA)

        if is_demo:
            demo_text = "DEMO MODE: No Webcam Detected"
            cv2.putText(frame, demo_text, (20, 35), font, 0.65, (0, 165, 255), 2, cv2.LINE_AA)

        return frame
