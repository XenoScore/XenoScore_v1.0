from __future__ import annotations
from typing import Dict, Any
from .core import FeatureComponent, piecewise_linear
from ..registry import register_component

@register_component("InfectionStatus")
class InfectionStatusComponent(FeatureComponent):
    """Map infection status to risk (active > recent > none)."""
    mapping = {"active": 1.0, "recent": 0.5, "none": 0.0}
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        status = str(row.get("infection_status") or "none").lower()
        return {"infection_risk": float(self.mapping.get(status, 0.0))}

@register_component("RenalFunction")
class RenalFunctionComponent(FeatureComponent):
    """Use eGFR to score renal risk (lower eGFR = higher risk)."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        egfr = row.get("egfr")
        # Example piecewise map: 15->1.0, 30->0.7, 60->0.3, 90->0.0
        y = piecewise_linear(egfr if egfr is not None else 90.0,
                             [(15,1.0),(30,0.7),(60,0.3),(90,0.0)])
        return {"renal_risk": float(y)}

@register_component("CardiovascularFunction")
class CardiovascularFunctionComponent(FeatureComponent):
    """Combine LVEF and MAP into a simple risk proxy."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        lvef = row.get("lvef", 60.0)  # percent
        mapx = row.get("map_mmHg", 75.0)
        # Lower EF -> higher risk; very low MAP -> higher risk
        ef_risk = piecewise_linear(lvef, [(20,1.0),(35,0.7),(50,0.3),(60,0.1),(70,0.0)])
        map_risk = piecewise_linear(mapx, [(50,1.0),(60,0.6),(70,0.3),(80,0.1),(90,0.0)])
        return {"cardio_risk": float(0.6*ef_risk + 0.4*map_risk)}

@register_component("PreXenoClinicalContext")
class PreXenoClinicalContextComponent(FeatureComponent):
    """Binary flags -> additive risk proxy."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        dialysis = 1.0 if row.get("dialysis") else 0.0
        mech = 1.0 if row.get("mechanical_support") else 0.0
        vaso = 1.0 if row.get("vasopressors") else 0.0
        # Normalize to [0,1]
        return {"context_risk": float(min(1.0, (dialysis + mech + vaso)/3.0))}
