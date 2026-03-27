"""
ROI Service – manages ROI state (polygon + active flag).
Delegates geometric logic to app.ml.roi_filter.
"""

from __future__ import annotations
from app.core.logger import logger
from app.ml.roi_filter import point_in_polygon


class RoiService:
    """
    Stateful service: stores the current ROI polygon and active flag.
    Geometric computation is in ml.roi_filter – no math here.
    """

    def __init__(self) -> None:
        self._points: list[tuple[int, int]] = []
        self._active: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_roi(self, points: list[list[float]], active: bool = True) -> None:
        self._points = [(int(p[0]), int(p[1])) for p in points]
        self._active = active and len(self._points) >= 3
        logger.info("ROI updated: %d pts, active=%s", len(self._points), self._active)

    def clear(self) -> None:
        self._points = []
        self._active = False
        logger.info("ROI cleared")

    def is_inside(self, cx: float, cy: float) -> bool:
        """Quick check used by legacy callers. Delegates to ml layer."""
        if not self._active or len(self._points) < 3:
            return True
        return point_in_polygon(cx, cy, self._points)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def points(self) -> list[tuple[int, int]]:
        return self._points

    @property
    def active(self) -> bool:
        return self._active

    @property
    def mid_y(self) -> float | None:
        """Tọa độ Y giữa vùng ROI — dùng làm counting line khi ROI active."""
        if not self._active or len(self._points) < 3:
            return None
        ys = [p[1] for p in self._points]
        return (min(ys) + max(ys)) / 2.0

    @property
    def line_position_ratio(self) -> float | None:
        """Tỷ lệ Y giữa ROI / frame height. None nếu không active."""
        return None  # cần frame height, tính ở stream_service


# Singleton
roi_service = RoiService()
