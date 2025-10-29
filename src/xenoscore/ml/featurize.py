from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
from ..registry import get_component
from ..components.core import FeatureComponent

def featurize(df: pd.DataFrame, component_specs: List[Dict[str, Any]]) -> pd.DataFrame:
    comps: List[FeatureComponent] = [get_component(s["name"])(params=s.get("params", {})) for s in component_specs]
    rows = []
    for _, row in df.iterrows():
        feats = {}
        for comp in comps:
            feats.update(comp.compute(row.to_dict()))
        rows.append(feats)
    return pd.DataFrame(rows, index=df.index).fillna(0.0)
