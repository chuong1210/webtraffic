"""
Congestion Monitor — alerts when vehicle density exceeds threshold
for a sustained duration.

Logic (inspired by traffic-congestion-detection reference):
  - Count active tracks in ROI (or total if no ROI)
  - If count >= vehicle_threshold for >= stable_duration seconds → CONGESTED
  - If count drops below threshold → NORMAL (with cooldown to avoid flicker)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class CongestionState:
    """Current congestion state exposed to API/frontend."""
    is_congested: bool = False
    vehicle_count: int = 0
    threshold: int = 10
    duration_seconds: float = 0.0
    stable_duration: float = 5.0
    message: str = ""
    level: str = "normal"  # normal | warning | critical


class CongestionMonitor:
    """
    Monitors vehicle density and triggers congestion alerts.

    Parameters:
        vehicle_threshold: number of vehicles to trigger congestion
        stable_duration: seconds the count must stay above threshold
        critical_multiplier: multiplier on threshold for "critical" level
    """

    def __init__(
        self,
        vehicle_threshold: int = 10,
        stable_duration: float = 5.0,
        critical_multiplier: float = 1.5,
    ):
        self.vehicle_threshold = vehicle_threshold
        self.stable_duration = stable_duration
        self.critical_multiplier = critical_multiplier

        self._above_since: float | None = None
        self._state = CongestionState(
            threshold=vehicle_threshold,
            stable_duration=stable_duration,
        )

    def update(self, active_track_count: int) -> CongestionState:
        """
        Call once per frame with the number of active tracked vehicles.
        Returns the current congestion state.
        """
        now = time.time()
        self._state.vehicle_count = active_track_count

        if active_track_count >= self.vehicle_threshold:
            if self._above_since is None:
                self._above_since = now

            elapsed = now - self._above_since
            self._state.duration_seconds = round(elapsed, 1)

            if elapsed >= self.stable_duration:
                self._state.is_congested = True
                critical_thresh = int(
                    self.vehicle_threshold * self.critical_multiplier
                )
                if active_track_count >= critical_thresh:
                    self._state.level = "critical"
                    self._state.message = (
                        f"Ket xe nghiem trong! {active_track_count} phuong tien "
                        f"(>{critical_thresh}) trong {elapsed:.0f}s"
                    )
                else:
                    self._state.level = "warning"
                    self._state.message = (
                        f"Mat do cao: {active_track_count} phuong tien "
                        f"(>={self.vehicle_threshold}) trong {elapsed:.0f}s"
                    )
            else:
                self._state.level = "normal"
                self._state.message = ""
                self._state.is_congested = False
        else:
            self._above_since = None
            self._state.duration_seconds = 0.0
            self._state.is_congested = False
            self._state.level = "normal"
            self._state.message = ""

        self._state.threshold = self.vehicle_threshold
        self._state.stable_duration = self.stable_duration
        return self._state

    @property
    def state(self) -> CongestionState:
        return self._state

    def reset(self):
        self._above_since = None
        self._state = CongestionState(
            threshold=self.vehicle_threshold,
            stable_duration=self.stable_duration,
        )

    def update_settings(
        self,
        threshold: int | None = None,
        duration: float | None = None,
    ):
        if threshold is not None:
            self.vehicle_threshold = max(1, threshold)
        if duration is not None:
            self.stable_duration = max(1.0, duration)
