from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import joblib
import pandas as pd
from ..registry import get_component
from ..components.core import FeatureComponent

@dataclass
class ModelScoreEngine:
    """Use ML model to learn weights/probabilities from component outputs."""
    component_specs: List[Dict[str, Any]]
    model_path: str

    def _instantiate_components(self) -> List[FeatureComponent]:
        return [get_component(spec["name"])(params=spec.get("params", {})) for spec in self.component_specs]

    def _featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        comps = self._instantiate_components()
        feats = []
        for _, row in df.iterrows():
            row_feats = {}
            for comp in comps:
                row_feats.update(comp.compute(row.to_dict()))
            feats.append(row_feats)
        return pd.DataFrame(feats, index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        model = joblib.load(self.model_path)
        X = self._featurize(df).fillna(0.0)
        proba = model.predict_proba(X)[:, 1]
        return pd.DataFrame({"model_probability": proba}, index=df.index)

def learn_weights_from_logistic(model_path: str, feature_names: List[str]) -> Dict[str, float]:
    """Convert a trained logistic regression's coefficients into feature weights."""
    model = joblib.load(model_path)
    coef = model.named_steps.get("clf", model).coef_.ravel()
    return {fn: float(w) for fn, w in zip(feature_names, coef)}
