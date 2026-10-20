import pandas as pd
from src.data_processing import clean_data, FEATURE_COLUMNS, TARGET_COLUMN

def test_clean_data_removes_duplicates():
    row = {c: 1 for c in FEATURE_COLUMNS}
    row[TARGET_COLUMN] = 0
    df = pd.DataFrame([row, row])
    assert len(clean_data(df)) == 1

def test_clean_data_preserves_binary_target():
    row0 = {c: 1 for c in FEATURE_COLUMNS}; row0[TARGET_COLUMN] = 0
    row1 = {c: 2 for c in FEATURE_COLUMNS}; row1[TARGET_COLUMN] = 1
    out = clean_data(pd.DataFrame([row0, row1]))
    assert set(out[TARGET_COLUMN].unique()) == {0, 1}
