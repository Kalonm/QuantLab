# main_fetch_csv.py

from pathlib import Path
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd

from config.settings import TIMEFRAMES, PAIRS   # re-use PAIRS to get symbol list
from utils.timeframe_res import tf_to_timedelta

# project root = QuantLab
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# How far back to go when creating a *new* file (if no CSV exists yet)
DEFAULT_LOOKBACK_DAYS = 365  # 1 year

# ---- derive unique symbols from PAIRS ----
SYMBOLS = sorted({s for pair in PAIRS for s in pair})


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fetch_rates(symbol: str, tf_const, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch ALL OHLCV bars for one symbol from MT5 between start and end.
    """
    rates = mt5.copy_rates_range(symbol, tf_const, start, end)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()  # empty

    df = pd.DataFrame(rates)
    # MT5 'time' is seconds since epoch; convert to datetime
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def update_symbol_timeframe_csv(symbol: str, tf: dict):
    """
    For a given *single symbol* and timeframe:
    - Load existing CSV if it exists
    - Determine start time for new fetch
    - Fetch new data
    - Append & deduplicate
    - Save back to CSV
    """
    tf_name = tf["name"]
    tf_const = tf["mt5_const"]

    # Directory per timeframe, file per symbol
    tf_dir = DATA_DIR / tf_name
    ensure_dir(tf_dir)
    csv_path = tf_dir / f"{symbol}.csv"

    now = datetime.now()

    if csv_path.exists():
        # Load old data and find last timestamp
        df_old = pd.read_csv(csv_path, parse_dates=["time"])
        if df_old.empty:
            start_time = now - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        else:
            last_time = df_old["time"].max()
            step = tf_to_timedelta(tf_name)
            start_time = last_time + step
    else:
        # No previous file: start from DEFAULT_LOOKBACK_DAYS ago
        df_old = pd.DataFrame()
        start_time = now - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    if start_time >= now:
        # Nothing to fetch
        return

    df_new = fetch_rates(symbol, tf_const, start_time, now)

    if df_new.empty:
        return

    if df_old.empty:
        df_all = df_new
    else:
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        # Drop duplicate timestamps
        df_all = df_all.drop_duplicates(subset=["time"]).sort_values("time")

    # Save back
    df_all.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} (rows: {len(df_all)})")


def main():
    # Initialise MT5 once
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    try:
        for tf in TIMEFRAMES:
            tf_name = tf["name"]
            print(f"\n=== Timeframe: {tf_name} ===")

            for symbol in SYMBOLS:
                try:
                    update_symbol_timeframe_csv(symbol, tf)
                except Exception as e:
                    print(f"Error {symbol} ({tf_name}): {e}")

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
