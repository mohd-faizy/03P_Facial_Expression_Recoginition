"""
Facial Expression Recognition Package
"""

try:
    from src.model import FacialExpressionModel
except ImportError:
    FacialExpressionModel = None

try:
    from src.camera import VideoCamera
except ImportError:
    VideoCamera = None

__all__ = ["FacialExpressionModel", "VideoCamera"]
