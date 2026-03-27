"""
Detection Service – business logic for running detection on a single frame.
Calls ML layer (YOLOModel + RoiFilter), does NOT touch ultralytics directly.
"""

from __future__ import annotations
import numpy as np

from app.core.logger import logger
from app.ml.yolo_model import yolo_model, RawDetection
from app.models.detection_model import BoundingBox, Detection


class DetectionService:
    """
    Orchestrates inference for one frame:
      1. YOLOModel.predict()   ← ml layer
      2. filter_by_roi()       ← ml layer
      3. Map to API models     ← models layer

    Service layer knows WHAT to do; ml layer knows HOW.
    """

    def detect(
        self,
        frame: np.ndarray,
        conf: float,
        roi_polygon: list,
        roi_active: bool,
    ) -> list[Detection]:
        """
        Run full detection pipeline.

        Args:
            frame:       BGR numpy array
            conf:        confidence threshold
            roi_polygon: list of (x,y) tuples
            roi_active:  whether ROI is active

        Returns:
            List of Detection Pydantic models (for API response / WebSocket)
        """
        # ── Step 1: ML inference ──────────────────────────────────────────────
        raw: list[RawDetection] = yolo_model.predict(frame, conf)

        # ROI is not applied here; ROI is used later for counting only.
        # ── Map to Pydantic API models ────────────────────────────────────────
        return [
            Detection(
                bbox=BoundingBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2),
                class_name=d.class_name,
                confidence=d.confidence,
            )
            for d in raw
        ]

    def detect_raw(
        self,
        frame: np.ndarray,
        conf: float,
        roi_polygon: list,
        roi_active: bool,
    ) -> list[RawDetection]:
        """
        Same as detect() but returns RawDetection (used by stream_service
        which needs .cx/.cy for tracking).
        ROI is not applied here; ROI filtering is handled later at the
        counting stage so visual detections are not hidden.
        """
        return yolo_model.predict(frame, conf)


# Singleton
detection_service = DetectionService()
