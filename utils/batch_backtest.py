# utils/batch_backtest.py
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import argparse
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import pandas as pd

from config.settings import TIMEFRAMES  # loaded from your JSON
from utils.rolling_ols import rolling_ols_alpha_beta, rolling_spread_and_z


# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
BT_DIR = BASE_DIR / "data" / "backtests"
BT_DIR.mkdir(parents=True, exist_ok=True)


# ------------- Backtest Core -----------
def load_pair_ohlc(tf_name: str, sym1: str, sym2: str) -> pd.DataFrame:
    """
    Load two single-symbol CSVs for timeframe tf_name and inner-join by time.
    Returns columns: time, close_sym1, close_sym2
    """
    p1 = RAW_DIR / tf_name / f"{sym1}.csv"
    p2 = RAW_DIR / tf_name / f"{sym2}.csv"
    if not p1.exists() or not p2.exists():
        return pd.DataFrame()

    df1 = pd.read_csv(p1, parse_dates=["time"])[["time", "close"]].rename(columns={"close": f"close_{sym1}"})
    df2 = pd.read_csv(p2, parse_dates=["time"])[["time", "close"]].rename(columns={"close": f"close_{sym2}"})
    df = df1.merge(df2, on="time", how="inner").sort_values("time").dropna()
    return df


def backtest_pair_rolling(
    df: pd.DataFrame,
    sym1: str,
    sym2: str,
    ols_window: int = 250,
    z_window: Optional[int] = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
    tx_bps_per_leg: float = 0.5,
) -> pd.DataFrame:
    """
    Dollar-neutral pairs backtest with rolling OLS hedge and next-bar execution.
    """
    c1 = df[f"close_{sym1}"].astype(float)
    c2 = df[f"close_{sym2}"].astype(float)
    log1 = np.log(c1)
    log2 = np.log(c2)

    # Rolling OLS on log prices (y=log1, x=log2)
    ols = rolling_ols_alpha_beta(
        x=log2, y=log1,
        window=ols_window,
        min_periods=max(20, int(0.4 * ols_window))
    )
    alpha = ols["alpha"]
    beta = ols["beta"]

    # Dynamic spread & z-score; use lagged alpha/beta to avoid look-ahead
    if z_window is None:
        z_window = ols_window
    zdf = rolling_spread_and_z(
        x=log2, y=log1,
        alpha=alpha, beta=beta,
        z_window=z_window,
        min_periods=max(20, int(0.4 * z_window))
    )
    z = zdf["z"]

    # Trading logic (positions apply on next bar)
    pos1 = np.zeros(len(df))  # +1 long spread, -1 short spread
    beta_lag = beta.shift(1).to_numpy()

    for t in range(1, len(df)):
        prev_pos = pos1[t - 1]
        z_prev = z.iloc[t - 1]

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

    pos2 = -beta_lag * pos1

    # Log returns
    r1 = log1.diff().fillna(0).to_numpy()
    r2 = log2.diff().fillna(0).to_numpy()

    gross_pnl = pos1 * r1 + pos2 * r2

    # costs: bps on position change per leg
    turns1 = np.abs(np.diff(pos1, prepend=0))
    turns2 = np.abs(np.diff(pos2, prepend=0))
    cost = (tx_bps_per_leg / 10000.0) * (turns1 + turns2)

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


# --------- Universe Loading / Filtering ----------
@dataclass
class UniverseFilter:
    adf_p_max: float = 0.05
    corr_min: float = 0.80
    half_life_min: int = 3
    half_life_max: int = 500
    top_k_per_tf: Optional[int] = None  # e.g. 1000; None = all that pass


def load_universe(universe_path: Path) -> pd.DataFrame:
    """
    Accepts:
      - A single CSV/Parquet file (e.g., pair_scan_summary_adf.csv)
      - A directory containing per-timeframe files like:
          pair_scan_<TF>_adf.parquet / pair_scan_<TF>_adf.csv
      - A directory containing pair_scan_summary_adf.csv

    Normalizes columns to:
      timeframe, sym1, sym2, corr, half_life_bars, adf_p, adf_stat, adf_cv_5pct, n_obs
    """
    def _read_one(p: Path) -> pd.DataFrame:
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Common renames
        rename_map = {
            "rr": "corr",
            "half_life": "half_life_bars",
            "hl_bars": "half_life_bars",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Coalesce ADF cols (prefer *_y from 2nd pass, then *_x, then plain)
        def coalesce(cols: List[str]) -> pd.Series:
            for c in cols:
                if c in df.columns and df[c].notna().any():
                    return df[c]
            # if none exist, return all-NaN series aligned to index
            return pd.Series(np.nan, index=df.index)

        if "adf_p" not in df.columns:
            df["adf_p"] = coalesce(["adf_p_y", "adf_p_x", "adf_p"])
        if "adf_stat" not in df.columns:
            df["adf_stat"] = coalesce(["adf_stat_y", "adf_stat_x", "adf_stat"])
        if "adf_cv_5pct" not in df.columns:
            df["adf_cv_5pct"] = coalesce(["adf_cv_5pct_y", "adf_cv_5pct_x", "adf_cv_5pct"])

        # Prefer n_obs; fall back to n_obs_adf if needed
        if "n_obs" not in df.columns and "n_obs_adf" in df.columns:
            df["n_obs"] = df["n_obs_adf"]

        # Drop noisy duplicates
        drop_cols = [c for c in [
            "adf_p_x", "adf_p_y",
            "adf_stat_x", "adf_stat_y",
            "adf_cv_5pct_x", "adf_cv_5pct_y",
        ] if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        # Validate required columns
        required = {"timeframe", "sym1", "sym2", "corr", "half_life_bars"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Universe missing required columns after normalization: {missing}")

        return df

    if not universe_path.exists():
        raise FileNotFoundError(universe_path)

    if universe_path.is_dir():
        # Prefer combined summary if present
        combined_csv = universe_path / "pair_scan_summary_adf.csv"
        if combined_csv.exists():
            df = _read_one(combined_csv)
            return _normalize(df)

        # Else gather per-TF ADF files
        files = (
            glob.glob(str(universe_path / "pair_scan_*_adf.parquet")) +
            glob.glob(str(universe_path / "pair_scan_*_adf.csv"))
        )
        if not files:
            raise FileNotFoundError(
                f"No per-timeframe ADF files found in {universe_path}. "
                f"Expected 'pair_scan_*_adf.parquet/csv' or 'pair_scan_summary_adf.csv'."
            )
        dfs = [_read_one(Path(f)) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        return _normalize(df)

    # Single file
    df = _read_one(universe_path)
    return _normalize(df)


def filter_universe(df: pd.DataFrame, uf: UniverseFilter) -> pd.DataFrame:
    df = df.copy()
    req_cols = {"timeframe", "sym1", "sym2", "corr", "half_life_bars", "adf_p"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Universe missing columns: {missing}")

    sel = df[
        df["corr"].abs().ge(uf.corr_min)
        & df["adf_p"].le(uf.adf_p_max)
        & df["half_life_bars"].between(uf.half_life_min, uf.half_life_max, inclusive="both")
    ]

    if uf.top_k_per_tf is not None:
        sel = (
            sel.assign(abs_corr=lambda d: d["corr"].abs())
               .sort_values(["timeframe", "abs_corr"], ascending=[True, False])
               .groupby("timeframe")
               .head(uf.top_k_per_tf)
               .drop(columns=["abs_corr"])
        )
    return sel.reset_index(drop=True)


# --------- Portfolio Aggregation ----------
def aggregate_portfolio(
    pnl_map: Dict[tuple, pd.DataFrame],
    weighting: str = "equal",     # "equal" or "inv_vol"
    vol_lookback: int = 250
) -> pd.DataFrame:
    """
    Combine per-pair pnl streams into a single portfolio equity.
    weighting:
      - equal: equal weight per active pair each bar
      - inv_vol: inverse vol (rolling vol_lookback) normalized over active pairs
    """
    if not pnl_map:
        return pd.DataFrame()

    # Align all pnl series by time (outer join), fill NaNs with 0 (not active yet)
    pnl_frames = []
    for key, df in pnl_map.items():
        sym1, sym2, tf_name = key
        s = df[["time", "pnl"]].copy()
        s = s.rename(columns={"pnl": f"{tf_name}__{sym1}__{sym2}"})
        s = s.set_index("time")
        pnl_frames.append(s)

    wide = pd.concat(pnl_frames, axis=1, join="outer").sort_index().fillna(0.0)

    if weighting == "equal":
        weights = (wide != 0).astype(float)
        wsum = weights.sum(axis=1).replace(0, np.nan)
        w = weights.div(wsum, axis=0).fillna(0.0)
    elif weighting == "inv_vol":
        # rolling vol per column; avoid division by zero
        vol = wide.rolling(vol_lookback, min_periods=max(20, vol_lookback//5)).std()
        inv_vol = 1.0 / vol.replace(0, np.nan)
        inv_vol = inv_vol.where(np.isfinite(inv_vol), 0.0)
        wsum = inv_vol.sum(axis=1).replace(0, np.nan)
        w = inv_vol.div(wsum, axis=0).fillna(0.0)
    else:
        raise ValueError("weighting must be 'equal' or 'inv_vol'")

    port_pnl = (wide * w).sum(axis=1)
    out = pd.DataFrame({"time": port_pnl.index, "pnl": port_pnl.values})
    out["equity"] = out["pnl"].cumsum()
    return out


# ------------- Batch Runner -------------
def run_batch(
    universe_path: Path,
    output_tag: str = "default",
    ols_window: int = 250,
    z_window: Optional[int] = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
    tx_bps_per_leg: float = 0.5,
    weighting: str = "equal",
    vol_lookback: int = 250,
    only_timeframes: Optional[List[str]] = None,   # NEW: restrict run to specific TFs
) -> Dict[str, Any]:
    uni = load_universe(universe_path)
    uf = UniverseFilter()  # tweak if you want different filters here
    uni = filter_universe(uni, uf)

    if uni.empty:
        raise ValueError("Universe after filters is empty.")

    pnl_map: Dict[tuple, pd.DataFrame] = {}
    per_pair_paths: List[Path] = []

    # Run by timeframe to keep I/O local
    for tf in TIMEFRAMES:
        tf_name = tf["name"] if isinstance(tf, dict) else getattr(tf, "name")
        if only_timeframes and tf_name not in only_timeframes:
            continue

        df_tf = uni[uni["timeframe"] == tf_name]
        if df_tf.empty:
            continue

        for _, row in df_tf.iterrows():
            s1, s2 = row["sym1"], row["sym2"]
            pair_df = load_pair_ohlc(tf_name, s1, s2)
            if pair_df.empty or len(pair_df) < max(ols_window, 50):
                continue

            bt = backtest_pair_rolling(
                pair_df, s1, s2,
                ols_window=ols_window,
                z_window=z_window or ols_window,
                z_entry=z_entry, z_exit=z_exit, z_stop=z_stop,
                tx_bps_per_leg=tx_bps_per_leg
            )
            key = (s1, s2, tf_name)
            pnl_map[key] = bt

            # Save per-pair time series
            out_pair = BT_DIR / f"bt_{output_tag}__{tf_name}__{s1}__{s2}.csv"
            bt.to_csv(out_pair, index=False)
            per_pair_paths.append(out_pair)

    # Aggregate portfolio
    port = aggregate_portfolio(pnl_map, weighting=weighting, vol_lookback=vol_lookback)
    out_port = None
    if not port.empty:
        out_port = BT_DIR / f"portfolio_{output_tag}.csv"
        port.to_csv(out_port, index=False)

    # Summaries per pair
    perf_rows = []
    for key, df in pnl_map.items():
        s1, s2, tf_name = key
        pnl = df["pnl"].to_numpy()
        if len(pnl) == 0:
            continue
        # NOTE: This uses 252 for annualization; consider TF-aware scaling later.
        mu = np.mean(pnl) * 252
        sd = np.std(pnl) * np.sqrt(252)
        sharpe = mu / sd if sd > 0 else np.nan
        cum = df["equity"].iloc[-1]
        perf_rows.append({
            "timeframe": tf_name,
            "sym1": s1,
            "sym2": s2,
            "cum_pnl": float(cum),
            "ann_mean": float(mu),
            "ann_vol": float(sd),
            "sharpe": float(sharpe),
            "n_bars": int(len(pnl)),
        })
    perf_df = pd.DataFrame(perf_rows).sort_values(["sharpe"], ascending=False)
    out_perf = BT_DIR / f"performance_{output_tag}.csv"
    perf_df.to_csv(out_perf, index=False)

    return {
        "per_pair_csvs": per_pair_paths,
        "portfolio_csv": out_port,
        "performance_csv": out_perf,
        "n_pairs": len(pnl_map),
    }


# ---------- Multi-Timeframe Sweeping ----------
def _tf_bars_map() -> Dict[str, Dict[str, int]]:
    """
    Build {tf_name: {'short': n, 'medium': n, 'long': n}} from TIMEFRAMES JSON.
    Expects each TF entry to have {"name": ..., "bars": {"short": .., "medium": .., "long": ..}}
    """
    out: Dict[str, Dict[str, int]] = {}
    for tf in TIMEFRAMES:
        name = tf["name"] if isinstance(tf, dict) else getattr(tf, "name")
        bars = tf["bars"] if isinstance(tf, dict) else getattr(tf, "bars")
        out[name] = {k: int(v) for k, v in bars.items()}
    return out


def _run_one_combo(
    universe_path: Path,
    tf_name: str,
    bar_set_name: str,
    base_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run one (timeframe, bar_set) combo by calling run_batch() while constraining
    to a single timeframe. We clone kwargs and override tag + windows.
    """
    tf_map = _tf_bars_map()
    bars = tf_map[tf_name][bar_set_name]  # e.g., 720 for H1/short, etc.

    kwargs = deepcopy(base_kwargs)
    kwargs["output_tag"] = f'{kwargs.get("output_tag","default")}__{tf_name}__{bar_set_name}'
    kwargs["ols_window"] = bars
    kwargs["z_window"] = bars
    kwargs["only_timeframes"] = [tf_name]

    return run_batch(universe_path=universe_path, **kwargs)


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Batch backtest rolling-OLS pairs from universe.")
    p.add_argument("--universe", type=str, required=True,
                   help="Path to universe file or folder (e.g., data/processed/pair_scan_summary_adf.csv).")
    p.add_argument("--tag", type=str, default="default", help="Output tag suffix.")
    p.add_argument("--ols_window", type=int, default=250)
    p.add_argument("--z_window", type=int, default=250)
    p.add_argument("--z_entry", type=float, default=2.0)
    p.add_argument("--z_exit", type=float, default=0.5)
    p.add_argument("--z_stop", type=float, default=3.5)
    p.add_argument("--tx_bps", type=float, default=0.5, help="Cost per leg per trade in bps.")
    p.add_argument("--weighting", type=str, default="equal", choices=["equal", "inv_vol"])
    p.add_argument("--vol_lookback", type=int, default=250)

    # Sweeper controls
    p.add_argument("--run_all", action="store_true",
                   help="Sweep all timeframes × bar_sets from timeframes.json (short, medium, long).")
    p.add_argument("--bar_sets", type=str, default="short,medium,long",
                   help="Comma-separated subset of bar sets to run when --run_all (e.g., 'short,long').")
    p.add_argument("--max_workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="Parallel workers for --run_all.")
    return p.parse_args()


# ---------- Entry ----------
if __name__ == "__main__":
    args = parse_args()
    universe_path = Path(args.universe)

    if args.run_all:
        tf_map = _tf_bars_map()
        bar_sets = [s.strip() for s in args.bar_sets.split(",") if s.strip()]
        combos: List[Tuple[str, str]] = [(tf, b) for tf in tf_map.keys() for b in bar_sets]
        print(f"Running {len(combos)} combos: {combos}")

        # Common kwargs (will be customized per combo)
        base_kwargs = dict(
            output_tag=args.tag,
            # ols/z windows overridden per combo
            z_entry=args.z_entry, z_exit=args.z_exit, z_stop=args.z_stop,
            tx_bps_per_leg=args.tx_bps,
            weighting=args.weighting, vol_lookback=args.vol_lookback,
        )

        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {
                ex.submit(_run_one_combo, universe_path, tf_name, bset, base_kwargs): (tf_name, bset)
                for (tf_name, bset) in combos
            }
            for fut in as_completed(futures):
                tf_name, bset = futures[fut]
                try:
                    res = fut.result()
                    print(f"✓ {tf_name}/{bset} done: {res.get('n_pairs', 0)} pairs")
                except Exception as e:
                    print(f"✗ {tf_name}/{bset} failed: {e}")

        print("\nAll combos complete.")
    else:
        # Single run using explicit windows
        res = run_batch(
            universe_path=universe_path,
            output_tag=args.tag,
            ols_window=args.ols_window,
            z_window=args.z_window,
            z_entry=args.z_entry,
            z_exit=args.z_exit,
            z_stop=args.z_stop,
            tx_bps_per_leg=args.tx_bps,
            weighting=args.weighting,
            vol_lookback=args.vol_lookback,
        )
        print("Backtest complete:", res)
