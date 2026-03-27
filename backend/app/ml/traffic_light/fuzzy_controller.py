"""
Fuzzy Logic Controller for Traffic Light Timing.

Mamdani-type fuzzy inference system:
  Inputs:  queue_length (vehicles), waiting_time (seconds)
  Output:  green_time (seconds)

Membership functions: Triangular (a, b, c)
  - queue_length:  Low, Medium, High, VeryHigh
  - waiting_time:  Short, Medium, Long, VeryLong
  - green_time:    VeryShort, Short, Medium, Long, VeryLong

Rule base: 4x4 = 16 rules (Mamdani min-max inference + centroid defuzzification)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TriMF:
    """Triangular membership function defined by (a, b, c)."""
    a: float  # left foot
    b: float  # peak
    c: float  # right foot

    def __call__(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        if x <= self.b:
            return (x - self.a) / max(self.b - self.a, 1e-9)
        return (self.c - x) / max(self.c - self.b, 1e-9)


@dataclass
class FuzzySet:
    """Named fuzzy set with a membership function."""
    name: str
    mf: TriMF


@dataclass
class FuzzyVar:
    """Fuzzy variable (input or output) with range and sets."""
    name: str
    lo: float
    hi: float
    sets: list[FuzzySet] = field(default_factory=list)

    def fuzzify(self, x: float) -> dict[str, float]:
        """Compute membership degree for each set."""
        x = np.clip(x, self.lo, self.hi)
        return {s.name: s.mf(x) for s in self.sets}


# ── Default Membership Function Parameters ───────────────────────────────────

DEFAULT_QUEUE_PARAMS = {
    # (a, b, c) for each linguistic term
    "Low":      (0,   0,  8),
    "Medium":   (4,  12, 20),
    "High":     (15, 25, 35),
    "VeryHigh": (28, 45, 45),
}

DEFAULT_WAIT_PARAMS = {
    "Short":    (0,   0,  20),
    "Medium":   (10,  30,  50),
    "Long":     (40,  70, 100),
    "VeryLong": (80, 120, 120),
}

DEFAULT_GREEN_PARAMS = {
    "VeryShort": (10, 10, 18),
    "Short":     (15, 22, 30),
    "Medium":    (25, 35, 45),
    "Long":      (40, 50, 60),
    "VeryLong":  (55, 70, 70),
}

# Rule matrix: RULE_TABLE[queue_idx][wait_idx] = green_term
# Rows = queue (Low → VeryHigh), Cols = wait (Short → VeryLong)
RULE_TABLE = [
    # Short    Medium    Long      VeryLong    ← waiting_time
    ["VeryShort", "Short",     "Short",  "Medium"],    # Low queue
    ["Short",     "Medium",    "Medium", "Long"],      # Medium queue
    ["Medium",    "Long",      "Long",   "VeryLong"],  # High queue
    ["Long",      "VeryLong",  "VeryLong", "VeryLong"],  # VeryHigh queue
]


def _build_var(name: str, lo: float, hi: float, params: dict[str, tuple]) -> FuzzyVar:
    sets = [FuzzySet(n, TriMF(*p)) for n, p in params.items()]
    return FuzzyVar(name=name, lo=lo, hi=hi, sets=sets)


class FuzzyController:
    """
    Mamdani fuzzy inference controller for traffic light green time.

    Usage:
        ctrl = FuzzyController()
        green_time = ctrl.decide(queue_length=15, waiting_time=45)
    """

    def __init__(
        self,
        queue_params: dict | None = None,
        wait_params: dict | None = None,
        green_params: dict | None = None,
        rule_table: list[list[str]] | None = None,
    ) -> None:
        qp = queue_params or DEFAULT_QUEUE_PARAMS
        wp = wait_params or DEFAULT_WAIT_PARAMS
        gp = green_params or DEFAULT_GREEN_PARAMS

        self.queue_var = _build_var("queue_length", 0, 45, qp)
        self.wait_var = _build_var("waiting_time", 0, 120, wp)
        self.green_var = _build_var("green_time", 10, 70, gp)
        self.rules = rule_table or RULE_TABLE

        # Pre-compute output domain for centroid
        self._out_x = np.linspace(self.green_var.lo, self.green_var.hi, 200)

    def decide(self, queue_length: float, waiting_time: float) -> float:
        """
        Compute optimal green time for given inputs.

        Returns green_time in seconds.
        """
        # 1. Fuzzify inputs
        q_degrees = self.queue_var.fuzzify(queue_length)
        w_degrees = self.wait_var.fuzzify(waiting_time)

        q_terms = list(q_degrees.keys())
        w_terms = list(w_degrees.keys())

        # 2. Evaluate rules (Mamdani: firing strength = min of antecedents)
        # Aggregate output: max over all rules for each output point
        aggregated = np.zeros_like(self._out_x)

        for i, qt in enumerate(q_terms):
            for j, wt in enumerate(w_terms):
                firing = min(q_degrees[qt], w_degrees[wt])
                if firing <= 0:
                    continue

                # Consequent fuzzy set
                green_term = self.rules[i][j]
                green_set = next(
                    (s for s in self.green_var.sets if s.name == green_term), None
                )
                if green_set is None:
                    continue

                # Clip output MF at firing strength, then max-aggregate
                clipped = np.minimum(
                    firing,
                    np.array([green_set.mf(x) for x in self._out_x])
                )
                aggregated = np.maximum(aggregated, clipped)

        # 3. Defuzzify (centroid method)
        total_area = np.sum(aggregated)
        if total_area < 1e-9:
            # Fallback: return midpoint
            return (self.green_var.lo + self.green_var.hi) / 2

        centroid = np.sum(self._out_x * aggregated) / total_area
        return float(np.clip(centroid, self.green_var.lo, self.green_var.hi))

    def get_params(self) -> dict:
        """Export current MF parameters (for GA optimization)."""
        return {
            "queue": {s.name: (s.mf.a, s.mf.b, s.mf.c) for s in self.queue_var.sets},
            "wait": {s.name: (s.mf.a, s.mf.b, s.mf.c) for s in self.wait_var.sets},
            "green": {s.name: (s.mf.a, s.mf.b, s.mf.c) for s in self.green_var.sets},
        }

    def set_params(self, params: dict) -> None:
        """Import MF parameters (from GA chromosome)."""
        for s in self.queue_var.sets:
            if s.name in params.get("queue", {}):
                a, b, c = params["queue"][s.name]
                s.mf = TriMF(a, b, c)
        for s in self.wait_var.sets:
            if s.name in params.get("wait", {}):
                a, b, c = params["wait"][s.name]
                s.mf = TriMF(a, b, c)
        for s in self.green_var.sets:
            if s.name in params.get("green", {}):
                a, b, c = params["green"][s.name]
                s.mf = TriMF(a, b, c)

    def to_chromosome(self) -> list[float]:
        """Flatten all MF params into a 1D chromosome for GA."""
        genes: list[float] = []
        for var in [self.queue_var, self.wait_var, self.green_var]:
            for s in var.sets:
                genes.extend([s.mf.a, s.mf.b, s.mf.c])
        return genes

    def from_chromosome(self, genes: list[float]) -> None:
        """Reconstruct MF params from GA chromosome."""
        idx = 0
        for var in [self.queue_var, self.wait_var, self.green_var]:
            for s in var.sets:
                a, b, c = genes[idx], genes[idx + 1], genes[idx + 2]
                # Enforce a <= b <= c
                a, b, c = min(a, b), sorted([a, b, c])[1], max(b, c)
                s.mf = TriMF(a, b, c)
                idx += 3
