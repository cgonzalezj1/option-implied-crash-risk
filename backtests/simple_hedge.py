import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


# ==== Inputs ====
DATA = "features/spy_model_table.csv"
THRESH = 0.3   # crash prob threshold to trigger hedge
Path("backtests").mkdir(exist_ok=True)

# ==== 1) Load ====
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
feat_cols = [c for c in df.columns if c not in ["date","close","drawdown5","crash5"]]
X, y = df[feat_cols].to_numpy(), df["crash5"].to_numpy()

# ==== 2) Fit on entire sample (for demo; in reality use walk-forward) ====
base = GradientBoostingClassifier(random_state=0)
clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
clf.fit(X, y)
df["prob"] = clf.predict_proba(X)[:,1]

# ==== 3) Approx hedge P&L ====
# Simplify: daily SPY return
df["ret"] = df["close"].pct_change().fillna(0)

# Option premium proxy: proportional to ATM vol * sqrt(5/252)
# (not exact BS, but preserves scaling)
df["prem"] = 0.01 * df["iv_atm_30"] * np.sqrt(5/252) * df["close"]

# Hedge payoff after 5 days: max(0, K - S_T), K=0.95*S_t
df["signal"] = (df["prob"] >= THRESH).astype(int)
payoffs = []
for i in range(len(df)):
    if df.loc[i,"signal"]==1 and i+5 < len(df):
        S0 = df.loc[i,"close"]
        K  = 0.95*S0
        S5 = df.loc[i+5,"close"]
        payoff = max(0, K - S5)
        prem   = df.loc[i,"prem"]
        payoffs.append((i, payoff - prem))
    else:
        payoffs.append((i, 0))

df["hedge_pnl"] = [p[1] for p in payoffs]

# ==== 4) Portfolio paths ====
df["cum_ret_nohedge"]   = (1+df["ret"]).cumprod()
df["cum_ret_modelhedge"] = (1+df["ret"] + df["hedge_pnl"]/df["close"]).cumprod()

# ==== 5) Save & summary ====
df.to_csv("backtests/hedge_results.csv", index=False)

print("Final No Hedge:", df["cum_ret_nohedge"].iloc[-1])
print("Final Model-Hedge:", df["cum_ret_modelhedge"].iloc[-1])
print("Total hedge trades:", df["signal"].sum())

import numpy as np

def sharpe_ratio(returns, freq=252):
    """Compute annualized Sharpe ratio from daily returns"""
    mean_ret = np.mean(returns)
    std_ret  = np.std(returns)
    if std_ret == 0:
        return np.nan
    return (mean_ret / std_ret) * np.sqrt(freq)

# Compute daily returns for both strategies
df["ret_nohedge"] = df["cum_ret_nohedge"].pct_change().fillna(0)
df["ret_modelhedge"] = df["cum_ret_modelhedge"].pct_change().fillna(0)

sharpe_nohedge = sharpe_ratio(df["ret_nohedge"])
sharpe_modelhedge = sharpe_ratio(df["ret_modelhedge"])

print(f"Sharpe (No Hedge): {sharpe_nohedge:.2f}")
print(f"Sharpe (Model Hedge): {sharpe_modelhedge:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(df["date"], df["cum_ret_nohedge"], label="No Hedge")
plt.plot(df["date"], df["cum_ret_modelhedge"], label="Model Hedge")
plt.legend()
plt.title("Cumulative Returns: SPY vs Model Hedge")
plt.ylabel("Growth of $1")
plt.grid(True)
plt.show()
