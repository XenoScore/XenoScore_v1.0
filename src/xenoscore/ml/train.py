from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from ..ml.featurize import featurize

@dataclass
class TrainConfig:
    C: float = 1.0
    penalty: str = "l2"
    cv_folds: int = 5
    random_state: int = 42

def train_logistic(
    df: pd.DataFrame,
    target_col: str,
    component_specs: List[Dict[str, Any]],
    cfg: Optional[TrainConfig] = None,
    model_out: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = cfg or TrainConfig()
    y = df[target_col].astype(int)
    X = featurize(df.drop(columns=[target_col]), component_specs)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(C=cfg.C, penalty=cfg.penalty, solver="liblinear", random_state=cfg.random_state))
    ])
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    pipe.fit(X, y)
    if model_out:
        joblib.dump(pipe, model_out)
    return {
        "feature_names": list(X.columns),
        "cv_auc_mean": float(auc.mean()),
        "cv_auc_std": float(auc.std()),
        "n_samples": int(len(df)),
        "model_path": model_out
    }
