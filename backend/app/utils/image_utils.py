"""Image utility functions."""

import cv2
import base64
import numpy as np


def encode_frame_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Encode OpenCV BGR frame as base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def resize_frame(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """Resize frame to max_width while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))
