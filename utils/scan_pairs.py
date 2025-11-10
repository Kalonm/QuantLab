# utils/scan_pairs.py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

from config.settings import TIMEFRAMES, PAIRS

# -------------------- Runtime tuning --------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

MAX_WORKERS = min(12, os.cpu_count() or 1)  # auto-detect CPUs
MIN_OBS = 50
SKIP_ADF = True  # Set True to skip slow ADF tests initially
# --------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- Fast I/O with Batch Loading -------
def _load_timeframe_aligned(tf_name: str) -> pd.DataFrame:
    """
    Load and align all symbols in ONE operation.
    Returns: DataFrame with time index and symbol columns.
    """
    tf_dir = RAW_DIR / tf_name
    if not tf_dir.exists():
        return pd.DataFrame()

    dfs = []
    csv_files = list(tf_dir.glob("*.csv"))
    
    print(f"[{tf_name}] Loading {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        sym = csv_file.stem
        try:
            df = pd.read_csv(csv_file, usecols=["time", "close"], parse_dates=["time"])
            if df.empty:
                continue
            df = df.set_index('time')[['close']].rename(columns={'close': sym})
            dfs.append(df)
        except Exception:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    print(f"[{tf_name}] Aligning {len(dfs)} symbols...")
    # Use outer join, then drop rows with too many NaNs
    aligned = pd.concat(dfs, axis=1, join='outer').sort_index()
    
    # Keep only rows where at least 2 symbols have data
    aligned = aligned.dropna(thresh=2)
    
    return aligned
# --------------------------------------------------------


# -------------------- Ultra-Fast Analytics --------------
def _fast_ols(log1: np.ndarray, log2: np.ndarray) -> Tuple[float, float]:
    """Closed-form OLS: log1 = alpha + beta * log2"""
    log2_mean = log2.mean()
    log1_mean = log1.mean()
    
    log2_dev = log2 - log2_mean
    log1_dev = log1 - log1_mean
    
    beta = np.dot(log2_dev, log1_dev) / np.dot(log2_dev, log2_dev)
    alpha = log1_mean - beta * log2_mean
    
    return alpha, beta


def _fast_ar1(spread: np.ndarray) -> float:
    """Fast AR(1) coefficient"""
    S_t = spread[:-1]
    S_tp1 = spread[1:]
    
    S_t_mean = S_t.mean()
    S_t_dev = S_t - S_t_mean
    
    phi = np.dot(S_t_dev, S_tp1 - S_tp1.mean()) / np.dot(S_t_dev, S_t_dev)
    return phi


def analyse_pair_ultrafast(p1: np.ndarray, p2: np.ndarray, skip_adf: bool = False) -> Optional[Dict[str, Any]]:
    """
    Ultra-fast analysis. ADF is optional (it's the slowest part by far).
    """
    if len(p1) < MIN_OBS or len(p2) < MIN_OBS:
        return None

    log1 = np.log(p1)
    log2 = np.log(p2)
    
    # Correlation
    corr = float(np.corrcoef(log1, log2)[0, 1])
    
    # Fast OLS
    alpha, beta = _fast_ols(log1, log2)
    
    # Spread
    spread = log1 - (alpha + beta * log2)
    
    # Fast AR(1)
    phi = _fast_ar1(spread)
    
    # Half-life
    if abs(phi) >= 1 or phi == 0:
        half_life_bars = float("inf")
    else:
        half_life_bars = float(np.log(2) / -np.log(abs(phi)))
    
    result = {
        "corr": corr,
        "alpha": alpha,
        "beta": beta,
        "phi": phi,
        "half_life_bars": half_life_bars,
        "n_obs": int(len(p1)),
    }
    
    # ADF is SLOW - make it optional
    if not skip_adf:
        try:
            adf_stat, adf_p, _, _, crit_vals, _ = adfuller(spread, autolag="AIC", maxlag=10)
            result.update({
                "adf_stat": float(adf_stat),
                "adf_p": float(adf_p),
                "adf_cv_5pct": float(crit_vals["5%"]),
            })
        except Exception:
            result.update({
                "adf_stat": float('nan'),
                "adf_p": float('nan'),
                "adf_cv_5pct": float('nan'),
            })
    else:
        result.update({
            "adf_stat": float('nan'),
            "adf_p": float('nan'),
            "adf_cv_5pct": float('nan'),
        })
    
    return result
# --------------------------------------------------------


# -------------------- Batch Processing ------------------
def process_pairs_batch(aligned: pd.DataFrame, pairs: List[Tuple[str, str]], 
                       skip_adf: bool = False) -> List[Dict[str, Any]]:
    """
    Process multiple pairs in batch (called by worker process).
    """
    results = []
    
    for sym1, sym2 in pairs:
        if sym1 not in aligned.columns or sym2 not in aligned.columns:
            continue
        
        pair_data = aligned[[sym1, sym2]].dropna()
        if len(pair_data) < MIN_OBS:
            continue
        
        p1 = pair_data[sym1].to_numpy()
        p2 = pair_data[sym2].to_numpy()
        
        metrics = analyse_pair_ultrafast(p1, p2, skip_adf=skip_adf)
        if metrics:
            metrics["sym1"] = sym1
            metrics["sym2"] = sym2
            results.append(metrics)
    
    return results


def _worker_batch(args: Tuple[pd.DataFrame, List[Tuple[str, str]], bool]) -> List[Dict[str, Any]]:
    """Process a batch of pairs in a worker process."""
    aligned, pairs_batch, skip_adf = args
    return process_pairs_batch(aligned, pairs_batch, skip_adf)
# --------------------------------------------------------


# -------------------- Main Orchestration ----------------
def main():
    all_rows: List[Dict[str, Any]] = []

    for tf in TIMEFRAMES:
        tf_name = tf["name"]
        
        print(f"\n{'='*60}")
        print(f"Processing timeframe: {tf_name}")
        print('='*60)
        
        # Load and align all data at once
        aligned = _load_timeframe_aligned(tf_name)
        
        if aligned.empty:
            print(f"[{tf_name}] No data found.")
            continue
        
        print(f"[{tf_name}] Aligned data shape: {aligned.shape}")
        
        # Filter usable pairs
        available_symbols = set(aligned.columns)
        usable_pairs = [(a, b) for (a, b) in PAIRS 
                       if a in available_symbols and b in available_symbols and a != b]
        
        if not usable_pairs:
            print(f"[{tf_name}] No usable pairs.")
            continue
        
        print(f"[{tf_name}] Processing {len(usable_pairs)} pairs...")
        
        # Split pairs into batches for process pool
        batch_size = max(10, len(usable_pairs) // (MAX_WORKERS * 2))
        pair_batches = [usable_pairs[i:i+batch_size] 
                       for i in range(0, len(usable_pairs), batch_size)]
        
        print(f"[{tf_name}] Using {len(pair_batches)} batches, {MAX_WORKERS} workers")
        
        # Process in parallel using ProcessPoolExecutor (better than threads for CPU-bound)
        rows_tf: List[Dict[str, Any]] = []
        
        tasks = [(aligned, batch, SKIP_ADF) for batch in pair_batches]
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_worker_batch, task): i 
                      for i, task in enumerate(tasks)}
            
            with tqdm(total=len(usable_pairs), desc=f"{tf_name} scan") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        rows_tf.extend(batch_results)
                        pbar.update(len(batch_results))
                    except Exception as e:
                        print(f"Batch failed: {e}")
                        continue
        
        if rows_tf:
            # Add timeframe info
            for row in rows_tf:
                row["timeframe"] = tf_name
            
            # Save per-timeframe
            tf_df = pd.DataFrame(rows_tf)
            
            # Sort by correlation for easy filtering
            tf_df = tf_df.sort_values('corr', ascending=False)
            
            tf_out = PROCESSED_DIR / f"pair_scan_{tf_name}.parquet"
            try:
                tf_df.to_parquet(tf_out, index=False, engine='pyarrow')
                print(f"[{tf_name}] ✓ Saved {len(rows_tf)} results to {tf_out}")
            except Exception:
                tf_out = PROCESSED_DIR / f"pair_scan_{tf_name}.csv"
                tf_df.to_csv(tf_out, index=False)
                print(f"[{tf_name}] ✓ Saved {len(rows_tf)} results to {tf_out}")
            
            all_rows.extend(rows_tf)
        else:
            print(f"[{tf_name}] No valid results.")
    
    # Aggregate
    if not all_rows:
        print("\n⚠ No results to save.")
        return
    
    df_all = pd.DataFrame(all_rows)
    df_all = df_all.sort_values(['timeframe', 'corr'], ascending=[True, False])
    
    out_path = PROCESSED_DIR / "pair_scan_summary.csv"
    df_all.to_csv(out_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved combined summary to {out_path}")
    print(f"   Total pairs analyzed: {len(df_all)}")
    print(f"   Timeframes: {df_all['timeframe'].nunique()}")
    print('='*60)


if __name__ == "__main__":
    main()