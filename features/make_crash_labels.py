# labels/make_crash_labels.py
import numpy as np
import pandas as pd
from pathlib import Path

SPOT_FILE = "spy_spot_2010_2023.csv"     # your combined spot file
OUT_FILE  = "features/spy_crash_labels.csv"   # output path

Path("features").mkdir(exist_ok=True)

# 1) Load and sort
spot = pd.read_csv(SPOT_FILE, parse_dates=["date"]).sort_values("date")
spot = spot[["date", "close"]].dropna()

S = spot["close"].to_numpy()

# 2) Compute 5-trading-day forward drawdown:
# drawdown5(t) = min_{k=1..5} S_{t+k}/S_t - 1
# We build shifted arrays and take elementwise min.
shifts = []
for k in range(1, 6):  # 1..5
    shifted = np.concatenate([S[k:], np.full(k, np.nan)])
    shifts.append(shifted)

fwd_min = np.nanmin(np.vstack(shifts), axis=0)   # elementwise min over 1..5
drawdown5 = fwd_min / S - 1                      # relative drop vs today

# 3) Binary crash label (threshold = -5%)
CRASH_THRESH = -0.05
crash5 = (drawdown5 <= CRASH_THRESH).astype(float)

# 4) Pack and drop tail NaNs (last 5 days lack forward data)
labels = spot.copy()
labels["drawdown5"] = drawdown5
labels["crash5"] = crash5
labels = labels.dropna(subset=["drawdown5"]).reset_index(drop=True)

# 5) Save + quick summary
labels.to_csv(OUT_FILE, index=False)

print(f"✅ Saved labels to {OUT_FILE}")
print(f"Date range: {labels['date'].min().date()} → {labels['date'].max().date()}")
print(f"Crash threshold: {CRASH_THRESH:.0%}")
print(f"Crash rate: {labels['crash5'].mean():.3%}  (positives per day)")
print(labels.head())


