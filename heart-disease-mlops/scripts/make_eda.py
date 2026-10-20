"""Create EDA tables and plots from the downloaded UCI dataset."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.data_processing import load_raw_data, clean_data, FEATURE_COLUMNS, TARGET_COLUMN

def main():
    data = ROOT / "data/raw/heart.csv"
    out = ROOT / "reports/figures"; out.mkdir(parents=True, exist_ok=True)
    df = clean_data(load_raw_data(data))
    df.to_csv(ROOT / "data/processed/heart_clean.csv", index=False)
    (ROOT / "reports/eda_summary.txt").write_text(
        f"Rows after duplicate removal: {len(df)}\nColumns: {len(df.columns)}\n\n"
        f"Missing values:\n{df.isna().sum().to_string()}\n\nClass balance:\n{df[TARGET_COLUMN].value_counts().to_string()}\n\n"
        f"Descriptive statistics:\n{df.describe().round(2).to_string()}\n", encoding="utf-8")
    df[FEATURE_COLUMNS].hist(figsize=(14,10), bins=15, edgecolor="black")
    plt.tight_layout(); plt.savefig(out/"feature_histograms.png", dpi=150); plt.close()
    plt.figure(figsize=(6,4)); df[TARGET_COLUMN].value_counts().sort_index().plot(kind="bar")
    plt.title("Target class balance"); plt.xlabel("Target (0/1)"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out/"class_balance.png", dpi=150); plt.close()
    plt.figure(figsize=(12,9)); corr = df.corr(numeric_only=True)
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90); plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature correlation heatmap"); plt.tight_layout(); plt.savefig(out/"correlation_heatmap.png", dpi=150); plt.close()
    print("EDA complete. See reports/figures and reports/eda_summary.txt.")
if __name__ == "__main__": main()
