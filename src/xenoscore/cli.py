from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
import pandas as pd
from .config import load_component_config, load_weights_config
from .data.io import read_any, write_csv
from .data.validation import validate_dataframe
from .scoring.weighted import WeightedScoreEngine
from .scoring.model import ModelScoreEngine
from .ml.train import train_logistic, TrainConfig
from .ml.featurize import featurize

app = typer.Typer(help="XenoScore CLI")

@app.command()
def score(
    input: str = typer.Option(..., "--input", "-i", help="Path to CSV/Parquet with samples"),
    config: str = typer.Option(..., "--config", "-c", help="YAML of components"),
    weights: str = typer.Option(None, "--weights", "-w", help="YAML of feature weights"),
    model: str = typer.Option(None, "--model", "-m", help="Path to trained model (joblib). If provided, uses ML engine."),
    out: str = typer.Option("predictions.csv", "--out", "-o", help="Output CSV path"),
):
    df = read_any(input)
    df, errors = validate_dataframe(df)
    if errors:
        print(f"[yellow]Validation warnings for {len(errors)} rows (continuing):[/yellow]")
        for idx, err in errors[:5]:
            print(f"  Row {idx}: {err}")
        if len(errors) > 5:
            print(f"  ... ({len(errors)-5} more)")

    comp_cfg = load_component_config(config)["components"]
    if model:
        eng = ModelScoreEngine(comp_cfg, model)
        preds = eng.predict_proba(df)
    else:
        if not weights:
            raise typer.BadParameter("Weights YAML is required when no model is provided.")
        w = load_weights_config(weights)
        eng = WeightedScoreEngine(comp_cfg, w)
        preds = eng.score_dataframe(df)

    out_df = pd.concat([df, preds], axis=1)
    write_csv(out_df, out)
    print(f"[green]Saved predictions to[/green] {out}")

@app.command()
def train(
    input: str = typer.Option(..., "--input", "-i", help="Path to CSV with labeled data"),
    target: str = typer.Option("outcome", "--target", "-t", help="Target column"),
    config: str = typer.Option(..., "--config", "-c", help="YAML of components"),
    model_out: str = typer.Option("model.joblib", "--model-out", "-m", help="Output model path"),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Cross-validation folds"),
    C: float = typer.Option(1.0, "--C", help="Inverse regularization strength"),
):
    df = read_any(input)
    comp_cfg = load_component_config(config)["components"]
    res = train_logistic(df, target, comp_cfg, TrainConfig(C=C, cv_folds=cv_folds), model_out)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    app()
