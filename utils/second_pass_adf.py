from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

from config.settings import TIMEFRAMES, PAIRS  # PAIRS not strictly needed here, but handy

# -------------------- Config knobs --------------------
# Selection criteria for ADF second pass (tune to taste)
CORR_MIN = 0.80          # only test pairs with |corr| >= CORR_MIN
HALF_LIFE_MIN = 3        # min bars (avoid phi ~ 1 noise)
HALF_LIFE_MAX = 500      # max bars
MIN_OBS = 100            # at least this many overlapping bars
TOP_K_PER_TF = None      # None for “all that pass filters”; else limit per timeframe

# Parallel / performance
MAX_WORKERS = min(12, os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def _load_timeframe_aligned(tf_name: str) -> pd.DataFrame:
    """
    Load all symbols for a timeframe into a single wide DataFrame:
    index=time; columns=symbol; values=close.
    Keep rows with at least 2 symbols present.
    """
    tf_dir = RAW_DIR / tf_name
    if not tf_dir.exists():
        return pd.DataFrame()

    dfs = []
    for csv_file in tf_dir.glob("*.csv"):
        sym = csv_file.stem
        try:
            df = pd.read_csv(csv_file, usecols=["time", "close"], parse_dates=["time"])
            if df.empty:
                continue
            df = df.set_index("time")[["close"]].rename(columns={"close": sym})
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    wide = pd.concat(dfs, axis=1, join="outer").sort_index()
    wide = wide.dropna(thresh=2)  # keep rows where >=2 symbols present
    return wide


def _fast_ols_alpha_beta(log1: np.ndarray, log2: np.ndarray) -> Tuple[float, float]:
    """Closed-form OLS: log1 = alpha + beta*log2."""
    m2 = log2.mean(); m1 = log1.mean()
    d2 = log2 - m2; d1 = log1 - m1
    den = np.dot(d2, d2)
    if den == 0:
        return np.nan, np.nan
    beta = float(np.dot(d2, d1) / den)
    alpha = float(m1 - beta * m2)
    return alpha, beta


def _rebuild_spread_from_aligned(aligned: pd.DataFrame, s1: str, s2: str) -> Optional[np.ndarray]:
    """Return spread (log s1 - (alpha + beta log s2)) on the aligned intersection."""
    if s1 not in aligned.columns or s2 not in aligned.columns:
        return None
    df_pair = aligned[[s1, s2]].dropna()
    if len(df_pair) < MIN_OBS:
        return None

    p1 = df_pair[s1].to_numpy(dtype=float)
    p2 = df_pair[s2].to_numpy(dtype=float)
    log1 = np.log(p1); log2 = np.log(p2)

    alpha, beta = _fast_ols_alpha_beta(log1, log2)
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return None

    spread = log1 - (alpha + beta * log2)
    return spread


def _adf_worker(args: Tuple[str, str, np.ndarray]) -> Optional[Dict[str, Any]]:
    """Run ADF on provided spread (already aligned & built)."""
    s1, s2, spread = args
    try:
        adf_stat, adf_p, _, _, crit_vals, _ = adfuller(spread, autolag="AIC")
        return {
            "sym1": s1,
            "sym2": s2,
            "adf_stat": float(adf_stat),
            "adf_p": float(adf_p),
            "adf_cv_5pct": float(crit_vals["5%"]),
            "n_obs_adf": int(len(spread)),
        }
    except Exception:
        return {
            "sym1": s1,
            "sym2": s2,
            "adf_stat": np.nan,
            "adf_p": np.nan,
            "adf_cv_5pct": np.nan,
            "n_obs_adf": 0,
        }


# ---------- Main second pass ----------
def main():
    # We’ll read the per-timeframe first-pass files you already saved
    combined_rows: List[pd.DataFrame] = []

    for tf in TIMEFRAMES:
        tf_name = tf["name"]
        # locate first-pass file
        parquet_path = PROCESSED_DIR / f"pair_scan_{tf_name}.parquet"
        csv_path = PROCESSED_DIR / f"pair_scan_{tf_name}.csv"

        if parquet_path.exists():
            fp = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            fp = pd.read_csv(csv_path)
        else:
            print(f"[{tf_name}] No first-pass results found; skipping.")
            continue

        if fp.empty:
            print(f"[{tf_name}] First-pass file is empty; skipping.")
            continue

        # Filter “worth doing”:
        cand = fp.loc[
            fp["n_obs"].ge(MIN_OBS)
            & fp["corr"].abs().ge(CORR_MIN)
            & fp["half_life_bars"].between(HALF_LIFE_MIN, HALF_LIFE_MAX, inclusive="both")
        ].copy()

        if cand.empty:
            print(f"[{tf_name}] No candidates met filters.")
            continue

        # (Optional) keep only Top-K by |corr|
        if TOP_K_PER_TF is not None and len(cand) > TOP_K_PER_TF:
            cand = cand.reindex(cand["corr"].abs().sort_values(ascending=False).index).head(TOP_K_PER_TF)

        print(f"[{tf_name}] Candidates for ADF: {len(cand)}")

        # Load aligned matrix once for this timeframe
        aligned = _load_timeframe_aligned(tf_name)
        if aligned.empty:
            print(f"[{tf_name}] No aligned data; skipping ADF.")
            continue

        # Prepare spreads (on the same alignment) to avoid large pickling by workers
        tasks: List[Tuple[str, str, np.ndarray]] = []
        for _, row in cand.iterrows():
            s1, s2 = row["sym1"], row["sym2"]
            spread = _rebuild_spread_from_aligned(aligned, s1, s2)
            if spread is not None and len(spread) >= MIN_OBS:
                tasks.append((s1, s2, spread))

        if not tasks:
            print(f"[{tf_name}] No usable spreads for ADF.")
            continue

        # Parallel ADF on the spreads
        adf_rows: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(_adf_worker, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{tf_name} ADF"):
                res = fut.result()
                if res:
                    adf_rows.append(res)

        if not adf_rows:
            print(f"[{tf_name}] No ADF results produced.")
            continue

        # Merge back into cand on (sym1, sym2)
        adf_df = pd.DataFrame(adf_rows)
        merged = cand.merge(adf_df, on=["sym1", "sym2"], how="left")

        # Save per-timeframe ADF-augmented file
        out_tf_parq = PROCESSED_DIR / f"pair_scan_{tf_name}_adf.parquet"
        try:
            merged.to_parquet(out_tf_parq, index=False)
            print(f"[{tf_name}] ✓ Saved ADF-augmented to {out_tf_parq}")
        except Exception:
            out_tf_csv = PROCESSED_DIR / f"pair_scan_{tf_name}_adf.csv"
            merged.to_csv(out_tf_csv, index=False)
            print(f"[{tf_name}] ✓ Saved ADF-augmented to {out_tf_csv}")

        merged["timeframe"] = tf_name
        combined_rows.append(merged)

    # Combined summary over all timeframes
    if not combined_rows:
        print("\n⚠ No ADF results to aggregate.")
        return

    df_all = pd.concat(combined_rows, ignore_index=True)
    df_all = df_all.sort_values(["timeframe", "corr"], ascending=[True, False])

    out_summary = PROCESSED_DIR / "pair_scan_summary_adf.csv"
    df_all.to_csv(out_summary, index=False)
    print(f"\n✅ Saved combined ADF summary to {out_summary} (rows={len(df_all)})")


if __name__ == "__main__":
    main()
