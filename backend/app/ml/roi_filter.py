"""
ML Layer – ROI Filter
Stateless geometric functions for filtering detections by ROI polygon.
"""

from __future__ import annotations
from typing import List, Tuple


Point = Tuple[float, float]
Polygon = List[Tuple[int, int]]


def point_in_polygon(px: float, py: float, polygon: Polygon) -> bool:
    """
    Ray-casting algorithm – O(n) point-in-polygon test.
    Returns True if (px, py) is inside polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def filter_by_roi(
    detections: list,
    polygon: Polygon,
    active: bool,
) -> list:
    """
    Filter raw detections by ROI polygon.

    Args:
        detections: list of objects with .cx and .cy attributes
        polygon:    list of (x, y) int tuples defining the ROI
        active:     if False, all detections pass through

    Returns:
        Filtered list (same object references).
    """
    if not active or len(polygon) < 3:
        return detections  # pass-through

    return [d for d in detections if point_in_polygon(d.cx, d.cy, polygon)]
