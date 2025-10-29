import pandas as pd
from xenoscore.scoring.weighted import WeightedScoreEngine
from xenoscore.config import load_component_config, load_weights_config

def test_weighted_scoring_runs():
    df = pd.read_csv("examples/example_dataset.csv")
    comp_cfg = load_component_config("configs/default_components.yaml")["components"]
    w = load_weights_config("configs/weights.example.yaml")
    eng = WeightedScoreEngine(comp_cfg, w)
    out = eng.score_dataframe(df)
    assert "risk_score" in out.columns
