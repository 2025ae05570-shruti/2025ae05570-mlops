"""Data loading and preprocessing helpers for the Heart Disease MLOps project."""
from pathlib import Path
import pandas as pd

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COLUMN = "target"

def load_raw_data(path):
    """Load either the canonical 14-column Cleveland CSV or a headered CSV."""
    path = Path(path)
    df = pd.read_csv(path, na_values=["?", "NA", ""])
    if set(FEATURE_COLUMNS).issubset(df.columns):
        target_candidates = [c for c in df.columns if c.lower() in {"target", "num", "condition", "heartdisease"}]
        if not target_candidates:
            raise ValueError("Headered dataset has features but no recognizable target column.")
        target_col = target_candidates[0]
        df = df[FEATURE_COLUMNS + [target_col]].copy()
        df = df.rename(columns={target_col: TARGET_COLUMN})
    else:
        # The official UCI processed Cleveland file has 13 features and a target, no header.
        if df.shape[1] != 14:
            raise ValueError(f"Expected 14 columns in raw Cleveland data, got {df.shape[1]}.")
        df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype("int64")
    return df

def clean_data(df):
    """Return a copy with duplicate rows removed; imputation happens inside the ML pipeline."""
    cleaned = df.copy().drop_duplicates().reset_index(drop=True)
    if cleaned[TARGET_COLUMN].isna().any():
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned
