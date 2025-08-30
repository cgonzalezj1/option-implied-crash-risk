import numpy as np, pandas as pd
import os
os.makedirs("features", exist_ok=True)


SRC = "spy_options_with_spot_2010_2023.parquet"
YEARS = range(2010, 2024)

def _first_or_nan(s):
    return s.iloc[0] if len(s) else np.nan

def day_features(df_day):
    # 30d / 90d buckets
    d30 = df_day[(df_day["dte"]>=20) & (df_day["dte"]<=40)]
    d90 = df_day[(df_day["dte"]>=80) & (df_day["dte"]<=100)]
    out = {"date": df_day["date"].iloc[0]}

    # ATM (|logM|<=0.02)
    atm_30 = d30.loc[d30["log_moneyness"].abs()<=0.02, "impl_volatility"].median()
    atm_90 = d90.loc[d90["log_moneyness"].abs()<=0.02, "impl_volatility"].median()
    out["iv_atm_30"] = atm_30
    out["iv_atm_90"] = atm_90
    out["term_slope_30_90"] = atm_30 - atm_90

    # 25-delta points (30d)
    c25 = d30[d30["option_type"]=="call"]
    p25 = d30[d30["option_type"]=="put"]
    iv_c25 = c25.iloc[(c25["delta"]-0.25).abs().argsort()[:1]]["impl_volatility"].pipe(_first_or_nan)
    iv_p25 = p25.iloc[(p25["delta"]+0.25).abs().argsort()[:1]]["impl_volatility"].pipe(_first_or_nan)
    out["rr25_30"] = iv_c25 - iv_p25
    out["bf25_30"] = 0.5*(iv_c25 + iv_p25) - atm_30

    # Liquidity / activity
    puts = df_day[df_day["option_type"]=="put"]
    calls = df_day[df_day["option_type"]=="call"]
    out["pc_volume_ratio"] = puts["volume"].sum() / max(1, calls["volume"].sum())

    # Greek aggregates in 30d
    out["sum_vega_30"] = d30["vega"].sum()
    out["med_abs_delta_30"] = d30["delta"].abs().median()
    out["sum_gamma_30"] = d30["gamma"].sum()

    # Optional: IV vs logM linear slope (skew proxy)
    dfit = d30[["impl_volatility","log_moneyness"]].dropna()
    if len(dfit) >= 8:
        x = dfit["log_moneyness"].to_numpy()
        X = np.c_[np.ones_like(x), x, x**2]
        beta = np.linalg.lstsq(X, dfit["impl_volatility"].to_numpy(), rcond=None)[0]
        out["skew_lin_30"] = beta[1]
        out["curv_quad_30"] = beta[2]
    else:
        out["skew_lin_30"] = np.nan
        out["curv_quad_30"] = np.nan

    return out

def build_features_year(y):
    y0, y1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01")
    df = pd.read_parquet(SRC, filters=[("date", ">=", y0), ("date", "<", y1)],
                         columns=["date","dte","option_type","delta","impl_volatility",
                                  "log_moneyness","bid","ask","mid","spread","volume",
                                  "vega","gamma"])
    # liquidity filters
    df = df[(df["dte"]>=5) & (df["dte"]<=365)]
    df = df[(df["bid"]>0) & (df["ask"]>0) & (df["impl_volatility"].notna())]
    df = df[(df["spread"] <= 0.25*df["mid"]) | (df["mid"]<=0)]  # allow zero-mid edge cases

    feats = [day_features(g) for _, g in df.groupby("date", sort=True)]
    return pd.DataFrame(feats)

if __name__ == "__main__":
    outs = []
    for y in YEARS:
        f = build_features_year(y)
        f.to_csv(f"features/spy_features_{y}.csv", index=False)
        outs.append(f)
    panel = pd.concat(outs).sort_values("date").reset_index(drop=True)
    panel.to_csv("features/spy_features_panel.csv", index=False)
    print("✅ features/spy_features_panel.csv  rows:", len(panel))
