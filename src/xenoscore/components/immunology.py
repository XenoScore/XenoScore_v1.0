from __future__ import annotations
from typing import Dict, Any
from .core import FeatureComponent, piecewise_linear
from ..registry import register_component

@register_component("BaselineAntibody")
class BaselineAntibodyComponent(FeatureComponent):
    """Score baseline anti-pig IgG/IgM titers (higher -> higher risk)."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        igg = row.get("baseline_anti_pig_IgG")
        igm = row.get("baseline_anti_pig_IgM")
        igg_r = piecewise_linear(igg if igg is not None else 0.0, [(0,0.0),(32,0.3),(64,0.6),(128,1.0)])
        igm_r = piecewise_linear(igm if igm is not None else 0.0, [(0,0.0),(32,0.3),(64,0.6),(128,1.0)])
        return {"baseline_humoral_risk": float(0.5*igg_r + 0.5*igm_r)}

@register_component("FlowCrossmatch")
class FlowCrossmatchComponent(FeatureComponent):
    """Use MFI or boolean positivity to estimate risk."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        if row.get("flow_cxm_positive") is True:
            return {"cxm_risk": 1.0}
        mfi = row.get("flow_cxm_mfi")
        mfi_r = piecewise_linear(mfi if mfi is not None else 0.0,
                                 [(0,0.0),(500,0.2),(1000,0.5),(2000,0.9),(5000,1.0)])
        return {"cxm_risk": float(mfi_r)}

@register_component("EarlyHumoralResponse")
class EarlyHumoralResponseComponent(FeatureComponent):
    """Capture IgG/IgM rise from baseline to POD1-3."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        b_igg, b_igm = row.get("baseline_anti_pig_IgG"), row.get("baseline_anti_pig_IgM")
        d1_igg, d3_igg = row.get("pod1_IgG"), row.get("pod3_IgG")
        d1_igm, d3_igm = row.get("pod1_IgM"), row.get("pod3_IgM")
        rise_igg = max(0.0, (max(d1_igg or 0.0, d3_igg or 0.0) - (b_igg or 0.0)))
        rise_igm = max(0.0, (max(d1_igm or 0.0, d3_igm or 0.0) - (b_igm or 0.0)))
        # Normalize via piecewise for fold-equivalent rise
        igg_r = piecewise_linear(rise_igg, [(0,0.0),(16,0.3),(32,0.6),(64,1.0)])
        igm_r = piecewise_linear(rise_igm, [(0,0.0),(16,0.3),(32,0.6),(64,1.0)])
        return {"early_humoral_risk": float(0.5*igg_r + 0.5*igm_r)}

@register_component("ComplementConsumption")
class ComplementConsumptionComponent(FeatureComponent):
    """Decrease in C3/C4 indicates consumption -> higher risk."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        c3_drop = (row.get("baseline_C3") or 0.0) - (row.get("pod3_C3") or row.get("baseline_C3") or 0.0)
        c4_drop = (row.get("baseline_C4") or 0.0) - (row.get("pod3_C4") or row.get("baseline_C4") or 0.0)
        c3_r = piecewise_linear(c3_drop, [(0,0.0),(10,0.3),(30,0.7),(50,1.0)])
        c4_r = piecewise_linear(c4_drop, [(0,0.0),(5,0.3),(15,0.7),(30,1.0)])
        return {"complement_consumption_risk": float(0.5*c3_r + 0.5*c4_r)}

@register_component("ComplementActivation")
class ComplementActivationComponent(FeatureComponent):
    """Soluble C5b-9 (MAC) as activation marker (higher -> higher risk)."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        sc5b9 = row.get("sC5b9")
        r = piecewise_linear(sc5b9 if sc5b9 is not None else 0.0, [(0,0.0),(100,0.4),(250,0.7),(500,1.0)])
        return {"complement_activation_risk": float(r)}

@register_component("DSA")
class DSAComponent(FeatureComponent):
    """Donor-specific antibodies presence (boolean)."""
    def compute(self, row: Dict[str, Any]) -> Dict[str, float]:
        return {"dsa_risk": 1.0 if row.get("dsa_present") else 0.0}
