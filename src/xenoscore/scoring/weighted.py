from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd
from ..registry import get_component
from ..components.core import FeatureComponent

@dataclass
class WeightedScoreEngine:
    component_specs: List[Dict[str, Any]]  # [{"name": "...", "params": {...}}, ...]
    weights: Dict[str, float]             # {"infection_risk": 1.0, "genetic_protection": -1.5, ...}
    score_minmax: tuple[float, float] = (0.0, 100.0)

    def _instantiate_components(self) -> List[FeatureComponent]:
        comps: List[FeatureComponent] = []
        for spec in self.component_specs:
            cls = get_component(spec["name"])
            comps.append(cls(params=spec.get("params", {})))
        return comps

    def _compute_feature_row(self, row: Dict[str, Any], components: List[FeatureComponent]) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        for comp in components:
            feats.update(comp.compute(row))
        return feats

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        comps = self._instantiate_components()
        rows = []
        for _, row in df.iterrows():
            feats = self._compute_feature_row(row.to_dict(), comps)
            raw = 0.0
            for k, v in feats.items():
                w = self.weights.get(k, 0.0)
                raw += w * v
            rows.append({**feats, "raw_score": raw})
        out = pd.DataFrame(rows, index=df.index)
        # scale raw_score to [min,max] via min-max over batch for display
        if len(out) > 0:
            rs = out["raw_score"]
            lo, hi = rs.min(), rs.max()
            if hi > lo:
                out["risk_score"] = self.score_minmax[0] + (rs - lo) * (self.score_minmax[1] - self.score_minmax[0]) / (hi - lo)
            else:
                out["risk_score"] = self.score_minmax[0] + 0.5 * (self.score_minmax[1] - self.score_minmax[0])
        else:
            out["risk_score"] = []
        return out
