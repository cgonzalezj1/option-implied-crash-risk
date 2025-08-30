# models/feature_importance.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

DATA = "features/spy_model_table.csv"
Path("models").mkdir(exist_ok=True)

# 1) Load data
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
y = df["crash5"].to_numpy()
feat_cols = [c for c in df.columns if c not in ["date","close","drawdown5","crash5"]]
X = df[feat_cols].to_numpy()

# 2) Time-aware split: use the last fold as "test" for explainability
tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(X))
tr, te = splits[-1]

# 3) Train calibrated GBM on train period
base = GradientBoostingClassifier(random_state=0)
clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
clf.fit(X[tr], y[tr])

p = clf.predict_proba(X[te])[:,1]
print(f"Test AUC={roc_auc_score(y[te], p):.3f}  AP={average_precision_score(y[te], p):.3f}")

# 4) Native (tree) importances — quick sanity check
gbm = clf.calibrated_classifiers_[0].estimator  # <- FIX
native_imp = pd.Series(gbm.feature_importances_, index=feat_cols).sort_values(ascending=False)
native_imp.to_csv("models/native_feature_importance.csv")
print("\nTop (native) importances:\n", native_imp.head(10))


# 5) Permutation importance (more robust)
pi = permutation_importance(clf, X[te], y[te], n_repeats=20, random_state=0, scoring="average_precision")
perm_imp = pd.Series(pi.importances_mean, index=feat_cols).sort_values(ascending=False)
perm_std = pd.Series(pi.importances_std, index=feat_cols).loc[perm_imp.index]
imp_table = pd.DataFrame({"perm_importance_mean": perm_imp, "perm_importance_std": perm_std,
                          "native_importance": native_imp.reindex(perm_imp.index)})
imp_table.to_csv("models/permutation_feature_importance.csv")
print("\nTop (permutation) importances:\n", imp_table.head(10))

# 6) Plot top-15 permutation importances
topN = 15
plt.figure(figsize=(8, 6))
imp_table.head(topN).iloc[::-1]["perm_importance_mean"].plot(kind="barh")
plt.title("Permutation Importance (top 15)")
plt.xlabel("Gain in Average Precision when feature is intact")
plt.tight_layout()
plt.savefig("models/permutation_importance_top15.png", dpi=160)
print("\nSaved:")
print("  models/native_feature_importance.csv")
print("  models/permutation_feature_importance.csv")
print("  models/permutation_importance_top15.png")
