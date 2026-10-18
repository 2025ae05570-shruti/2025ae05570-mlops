"""Download the official UCI Heart Disease archive and extract Cleveland data."""
from pathlib import Path
import io, zipfile, urllib.request
ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
OUT = ROOT / "data" / "raw" / "heart.csv"
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
           "exang","oldpeak","slope","ca","thal","target"]

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UCI Heart Disease archive from {URL}")
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith("processed.cleveland.data")]
        if not candidates:
            raise RuntimeError("Could not find processed.cleveland.data in the UCI archive.")
        raw = zf.read(candidates[0]).decode("utf-8")
    OUT.write_text(",".join(COLUMNS) + "\n" + raw, encoding="utf-8")
    print(f"Saved raw dataset to {OUT}")

if __name__ == "__main__":
    main()
