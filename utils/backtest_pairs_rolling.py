# utils/backtest_pairs_rolling.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from utils.rolling_ols import rolling_ols_alpha_beta, rolling_spread_and_z  # adjust import if needed

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
BT_DIR = BASE_DIR / "data" / "backtests"
BT_DIR.mkdir(parents=True, exist_ok=True)

def load_pair(tf_name: str, sym1: str, sym2: str) -> pd.DataFrame:
    p1 = pd.read_csv(RAW_DIR / tf_name / f"{sym1}.csv", parse_dates=["time"])
    p2 = pd.read_csv(RAW_DIR / tf_name / f"{sym2}.csv", parse_dates=["time"])
    df = (
        p1[["time","close"]].rename(columns={"close": f"close_{sym1}"})
        .merge(p2[["time","close"]].rename(columns={"close": f"close_{sym2}"}), on="time", how="inner")
        .sort_values("time")
        .dropna()
    )
    return df

def backtest_pair_rolling(
    df: pd.DataFrame,
    sym1: str,
    sym2: str,
    ols_window: int = 250,
    z_window: int | None = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
    tx_bps_per_leg: float = 0.5,     # transaction cost per leg per trade (bps)
):
    """
    Dollar-neutral (beta-hedged) pairs backtest with rolling OLS hedge.
    - Next-bar execution.
    - Position in sym1 = +1/-1; sym2 = -beta * pos1 (hedge).
    - PnL approximated with log returns.
    """
    c1 = df[f"close_{sym1}"].astype(float)
    c2 = df[f"close_{sym2}"].astype(float)
    log1 = np.log(c1)
    log2 = np.log(c2)

    # Rolling OLS on log-prices
    ols = rolling_ols_alpha_beta(x=log2, y=log1, window=ols_window, min_periods=max(20, int(ols_window*0.4)))
    alpha, beta = ols["alpha"], ols["beta"]

    # Build dynamic spread & z-score (using lagged alpha/beta)
    zdf = rolling_spread_and_z(x=log2, y=log1, alpha=alpha, beta=beta, z_window=z_window, min_periods=max(20, int((z_window or ols_window)*0.4)))
    z = zdf["z"]

    # Trading logic (next-bar execution)
    pos1 = np.zeros(len(df))     # +1 long spread, -1 short spread
    beta_lag = beta.shift(1).to_numpy()

    for t in range(1, len(df)):
        prev_pos = pos1[t-1]
        z_prev = z.iloc[t-1]

        # entry/exit rules
        if prev_pos == 0:
            if z_prev >= z_entry:
                pos1[t] = -1
            elif z_prev <= -z_entry:
                pos1[t] = +1
        else:
            if abs(z_prev) <= z_exit or abs(z_prev) >= z_stop:
                pos1[t] = 0
            else:
                pos1[t] = prev_pos

    # Hedge with beta (use lagged beta to avoid look-ahead)
    pos2 = -beta_lag * pos1

    # Log returns (mid-to-mid)
    r1 = log1.diff().fillna(0).to_numpy()
    r2 = log2.diff().fillna(0).to_numpy()

    # PnL on next bar (positions are already for bar t, applied to return t)
    gross_pnl = pos1 * r1 + pos2 * r2

    # Turnover & costs (apply bps on position changes per leg)
    turns1 = np.abs(np.diff(pos1, prepend=0))
    turns2 = np.abs(np.diff(pos2, prepend=0))

    # Notional per leg assumed = 1; you can scale by volatility or target risk
    cost = (tx_bps_per_leg/10000.0) * (turns1 + turns2)

    pnl = gross_pnl - cost

    out = df[["time"]].copy()
    out["z"] = z.values
    out["alpha"] = alpha.values
    out["beta"] = beta.values
    out["pos1"] = pos1
    out["pos2"] = pos2
    out["r1"] = r1
    out["r2"] = r2
    out["gross_pnl"] = gross_pnl
    out["cost"] = cost
    out["pnl"] = pnl
    out["equity"] = pnl.cumsum()

    return out

def run_single_backtest(
    tf_name: str,
    sym1: str,
    sym2: str,
    ols_window: int = 250,
    z_window: int | None = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
    tx_bps_per_leg: float = 0.5,
) -> Path:
    df = load_pair(tf_name, sym1, sym2)
    if df.empty:
        raise ValueError("No overlapping data for that pair/timeframe.")

    bt = backtest_pair_rolling(
        df, sym1, sym2,
        ols_window=ols_window,
        z_window=z_window or ols_window,
        z_entry=z_entry, z_exit=z_exit, z_stop=z_stop,
        tx_bps_per_leg=tx_bps_per_leg
    )

    out_path = BT_DIR / f"bt_{tf_name}__{sym1}__{sym2}.csv"
    bt.to_csv(out_path, index=False)
    return out_path
