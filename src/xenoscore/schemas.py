from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field

class Sample(BaseModel):
    """Unified row-level schema; fields are optional to keep ingestion flexible.
    This allows incremental adoption and heterogeneous datasets.
    """
    # Pre-xeno / patient context
    infection_status: Optional[Literal["active","recent","none"]] = None
    egfr: Optional[float] = Field(None, description="Estimated GFR (ml/min/1.73m2)")
    creatinine: Optional[float] = None
    lvef: Optional[float] = Field(None, description="Left ventricular ejection fraction (0-100)")
    map_mmHg: Optional[float] = None
    dialysis: Optional[bool] = None
    mechanical_support: Optional[bool] = None
    vasopressors: Optional[bool] = None

    # Donor / graft
    donor_age_months: Optional[float] = None
    donor_weight_kg: Optional[float] = None
    donor_pcmv: Optional[bool] = None
    ggta1_ko: Optional[bool] = None
    cmah_ko: Optional[bool] = None
    b4galnt2_ko: Optional[bool] = None
    hCD46: Optional[bool] = None
    hTHBD: Optional[bool] = None

    # Immunology (baseline & early)
    baseline_anti_pig_IgG: Optional[float] = None
    baseline_anti_pig_IgM: Optional[float] = None
    flow_cxm_mfi: Optional[float] = None
    flow_cxm_positive: Optional[bool] = None
    pod1_IgG: Optional[float] = None
    pod3_IgG: Optional[float] = None
    pod1_IgM: Optional[float] = None
    pod3_IgM: Optional[float] = None
    baseline_C3: Optional[float] = None
    pod3_C3: Optional[float] = None
    baseline_C4: Optional[float] = None
    pod3_C4: Optional[float] = None
    sC5b9: Optional[float] = None
    dsa_present: Optional[bool] = None

    # Outcome placeholder (for training only; not required at inference)
    outcome: Optional[int] = Field(None, description="Binary outcome (e.g., graft failure=1) for training")
