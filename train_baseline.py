# models/train_baseline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path

MODEL_TABLE = "features/spy_model_table.csv"
Path("models").mkdir(exist_ok=True)

# 1) Load
df = pd.read_csv(MODEL_TABLE, parse_dates=["date"])

# Features and target
feat_cols = [c for c in df.columns if c not in ["date","close","drawdown5","crash5"]]
X, y = df[feat_cols].to_numpy(), df["crash5"].to_numpy()

# 2) Time-series split
tscv = TimeSeriesSplit(n_splits=5)
aucs, aps = [], []

for fold, (tr, te) in enumerate(tscv.split(X)):
    base = GradientBoostingClassifier(random_state=0)
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic")  # probability calibration
    
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:,1]

    aucs.append(roc_auc_score(y[te], p))
    aps.append(average_precision_score(y[te], p))
    print(f"Fold {fold}: AUC={aucs[-1]:.3f}  AP={aps[-1]:.3f}")

print("Mean AUC:", np.mean(aucs))
print("Mean AP :", np.mean(aps))
