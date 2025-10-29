from __future__ import annotations
from typing import Tuple
import pandas as pd
from ..schemas import Sample

def validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Validate each row against Sample schema. Returns cleaned df and a list of errors (row indices)."""
    errors = []
    cleaned = []
    for idx, row in df.iterrows():
        try:
            s = Sample(**row.to_dict())
            cleaned.append(s.model_dump())
        except Exception as e:
            errors.append((idx, str(e)))
            cleaned.append(row.to_dict())
    return pd.DataFrame(cleaned), errors
