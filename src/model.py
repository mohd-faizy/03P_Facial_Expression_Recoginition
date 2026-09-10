import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Configure GPU dynamic memory growth for modern TensorFlow 2.x
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[WARN] GPU memory growth configuration: {e}")


class FacialExpressionModel(object):
    """
    Facial expression recognition inference model using Keras and TensorFlow.
    Predicts one of seven canonical emotion categories from 48x48 grayscale facial ROI.
    """

    EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def __init__(self, model_json_file="model.json", model_weights_file=None):
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)

        # Resolve architecture JSON in models/ directory
        json_path = self._resolve_json_path(model_json_file, project_root)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Model JSON file not found at: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # Resolve weights path in models/ directory (supports Keras 3 .weights.h5 and legacy .h5)
        weights_path = self._resolve_weights_path(model_weights_file, project_root)
        if weights_path and os.path.exists(weights_path):
            try:
                self.loaded_model.load_weights(weights_path)
                print(f"[INFO] Loaded model weights from: {os.path.basename(weights_path)}")
            except Exception as err:
                print(f"[WARN] Failed loading weights from {weights_path}: {err}. Using base weights.")
        else:
            print("[INFO] No weights file found. Initialized model architecture with base weights.")

        self.preds = None

    @staticmethod
    def _resolve_json_path(filename, project_root):
        candidates = [
            filename,
            os.path.join(project_root, 'models', filename),
            os.path.join(project_root, filename),
            os.path.join('models', filename)
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return os.path.abspath(c)
        return os.path.join(project_root, 'models', filename)

    @staticmethod
    def _resolve_weights_path(filename, project_root):
        candidates = []
        if filename:
            candidates.extend([
                filename,
                os.path.join(project_root, 'models', filename),
                os.path.join(project_root, filename),
                os.path.join('models', filename)
            ])
        # Auto-detect standard names in models/ directory
        for default_name in ['model.weights.h5', 'model_weights.h5', 'model.keras']:
            candidates.extend([
                os.path.join(project_root, 'models', default_name),
                os.path.join('models', default_name),
                os.path.join(project_root, default_name),
                default_name
            ])

        for c in candidates:
            if c and os.path.exists(c):
                return os.path.abspath(c)
        return None

    def predict_emotion(self, img):
        """
        Runs forward pass inference on a preprocessed 48x48 grayscale image tensor.
        Input shape: (1, 48, 48, 1)
        Returns predicted emotion label string.
        """
        self.preds = self.loaded_model.predict(img, verbose=0)
        return FacialExpressionModel.EMOTIONS_LIST[int(np.argmax(self.preds))]

    # Aliases for backward compatibility
    predict_emotions = predict_emotion
    pridict_emotions = predict_emotion