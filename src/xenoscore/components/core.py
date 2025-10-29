from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import math

@dataclass
class FeatureComponent:
    """Base class for all components.
    Implement `compute(row)` to return a {feature_name: numeric_value} mapping.
    """
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def required_columns(self) -> List[str]:
        """Optional: list of columns this component expects."""
        return []

    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError

# Helper transforms
def piecewise_linear(x: float, points: List[tuple[float, float]]) -> float:
    """Map x via piecewise-linear (x_i -> y_i). points sorted by x_i."""
    if x is None or math.isnan(x):
        return 0.0
    pts = sorted(points, key=lambda p: p[0])
    # clamp
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    # interpolate
    for (x0,y0), (x1,y1) in zip(pts[:-1], pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t*(y1 - y0)
    return 0.0
