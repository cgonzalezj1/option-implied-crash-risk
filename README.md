# option-implied-crash-risk
## Overview
This project uses **OptionMetrics IvyDB (WRDS)** option data on SPY (2010–2023) to forecast short-term crash risk.  
The pipeline extracts features from the option surface, trains a machine learning model, and simulates a hedging strategy.

## Pipeline
1. **Feature Engineering:** Extract ATM vol, term slope, skew, curvature, put/call ratio, and Greek aggregates.  
2. **Crash Labels:** Define a crash as a 5-day forward drawdown ≤ -5%.  
3. **Modeling:** Train gradient boosting with calibrated probabilities.  
4. **Backtest:** Hedge by buying 5% OTM puts when crash probability exceeds a threshold.  
5. **Results:** Model hedge Sharpe ratio = **0.81**, higher than SPY buy-and-hold (~0.6).

## Repo Layout
- `features/` → scripts to build features and labels.  
- `models/` → ML training + feature importance.  
- `backtests/` → hedge backtest scripts.  

## Requirements
Install Python packages:
```bash
pip install -r requirements.txt
