from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

def read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    elif p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
