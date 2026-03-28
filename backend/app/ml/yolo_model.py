"""
ML Layer – YOLOv8 Model Wrapper
Encapsulates all direct interactions with the ultralytics YOLO library.
The service layer never imports ultralytics directly.

Supports both:
  - predict(): detection only (no tracking)
  - track(): detection + built-in tracking (ByteTrack / BoT-SORT)
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RawDetection:
    """Raw output from YOLOv8 before any business logic."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None  # populated when using track()

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


class YOLOModel:
    """
    Thin wrapper around ultralytics YOLO.

    Responsibilities:
    - Load / unload model weights
    - Run inference (predict) or tracking (track) on a single BGR frame
    - Return structured RawDetection list
    """

    def __init__(self) -> None:
        self._model = None
        self._model_path: str = ""
        self._class_names: dict[int, str] = {}
        self._device = "cpu"
        self._use_half = False

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> None:
        from ultralytics import YOLO
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weights not found: {path}")

        self._model = YOLO(str(path))
        self._model_path = path.name

        # Move to GPU if available
        self._device = 0 if torch.cuda.is_available() else "cpu"
        self._use_half = torch.cuda.is_available()  # FP16 on GPU only

        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            self._class_names = {int(i): str(n) for i, n in names.items()}
        else:
            self._class_names = {i: str(n) for i, n in enumerate(names)}

    def unload(self) -> None:
        self._model = None
        self._model_path = ""
        self._class_names = {}

    def predict(self, frame: np.ndarray, conf: float = 0.35) -> List[RawDetection]:
        """Run inference on a BGR numpy frame (detection only, no tracking)."""
        if not self.is_loaded:
            return []

        results = self._model(
            frame, verbose=False, conf=conf,
            device=self._device, half=self._use_half, imgsz=640,
        )[0]
        return self._parse_boxes(results)

    def track(
        self,
        frame: np.ndarray,
        conf: float = 0.35,
        tracker: str = "bytetrack.yaml",
        persist: bool = True,
    ) -> List[RawDetection]:
        """
        Run inference + built-in tracking (ByteTrack or BoT-SORT).

        Args:
            frame: BGR numpy array
            conf: confidence threshold
            tracker: "bytetrack.yaml" or "botsort.yaml"
            persist: keep track IDs across frames (must be True for continuous tracking)

        Returns:
            List of RawDetection with track_id populated
        """
        if not self.is_loaded:
            return []

        results = self._model.track(
            frame,
            verbose=False,
            conf=conf,
            tracker=tracker,
            persist=persist,
            device=self._device,
            half=self._use_half,
            imgsz=640,
        )[0]
        return self._parse_boxes(results)

    def reset_tracker(self) -> None:
        """Reset the internal tracker state (new IDs on next track() call)."""
        if self._model is not None and hasattr(self._model, "predictor"):
            predictor = self._model.predictor
            if predictor is not None and hasattr(predictor, "trackers"):
                predictor.trackers = []

    # ── Private ───────────────────────────────────────────────────────────────

    def _parse_boxes(self, results) -> List[RawDetection]:
        """Parse ultralytics Results into RawDetection list."""
        detections: List[RawDetection] = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # track_id is available when using model.track()
            tid = None
            if box.id is not None:
                tid = int(box.id[0])

            detections.append(
                RawDetection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=cls_id,
                    class_name=self._class_names.get(cls_id, str(cls_id)),
                    confidence=round(float(box.conf[0]), 3),
                    track_id=tid,
                )
            )

        return detections

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> str:
        return self._model_path


# Module-level singleton
yolo_model = YOLOModel()
