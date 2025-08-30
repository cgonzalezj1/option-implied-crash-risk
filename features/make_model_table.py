# features/make_model_table.py
import pandas as pd
from pathlib import Path

FEAT_FILE  = "features/spy_features_panel.csv"
LABEL_FILE = "features/spy_crash_labels.csv"
OUT_FILE   = "features/spy_model_table.csv"

Path("features").mkdir(exist_ok=True)

# 1) Load
X = pd.read_csv(FEAT_FILE, parse_dates=["date"])
y = pd.read_csv(LABEL_FILE, parse_dates=["date"])

# 2) Merge on date
df = X.merge(y, on="date", how="inner")

# 3) Drop rows with missing values (safe since some days may not have all feature buckets)
df = df.dropna().reset_index(drop=True)

# 4) Save
df.to_csv(OUT_FILE, index=False)

print(f"✅ Saved modeling table to {OUT_FILE}")
print(f"Rows: {len(df):,}")
print("Columns:", df.columns.tolist()[:10], "...")
print(df.head())


