from __future__ import annotations
from typing import Dict, Any
from .core import FeatureComponent, piecewise_linear
from ..registry import register_component

@register_component("DonorGenetics")
class DonorGeneticsComponent(FeatureComponent):
    """Protective gene edits/transgenes lower risk (negative contribution)."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        # Count protective edits/transgenes present
        edits = int(bool(row.get("ggta1_ko"))) + int(bool(row.get("cmah_ko"))) + int(bool(row.get("b4galnt2_ko")))
        transgenes = int(bool(row.get("hCD46"))) + int(bool(row.get("hTHBD")))
        # Map to protective score in [0,1]; more protection -> higher protection
        protection = min(1.0, 0.15*edits + 0.2*transgenes)  # tweakable
        # Return as negative "risk" (engine can add weights accordingly)
        return {"genetic_protection": float(protection)}

@register_component("DonorAgeSize")
class DonorAgeSizeComponent(FeatureComponent):
    """Age/size away from target windows increases risk."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        age = row.get("donor_age_months")
        wt = row.get("donor_weight_kg")
        # Example desired windows (edit as evidence evolves)
        age_r = piecewise_linear(age if age is not None else 8.0, [(2,0.2),(6,0.0),(12,0.1),(24,0.4),(36,0.7)])
        wt_r = piecewise_linear(wt if wt is not None else 60.0, [(30,0.2),(50,0.0),(80,0.2),(120,0.6)])
        return {"donor_age_size_risk": float(0.5*age_r + 0.5*wt_r)}

@register_component("DonorPCMV")
class DonorPCMVComponent(FeatureComponent):
    """pCMV positivity -> high risk."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        return {"donor_pcmv_risk": 1.0 if row.get("donor_pcmv") else 0.0}
