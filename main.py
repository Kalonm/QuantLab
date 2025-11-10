import MetaTrader5 as mt5

from config.settings import TIMEFRAMES, PAIRS

def main():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    for tf in TIMEFRAMES:
        tf_name = tf["name"]
        tf_const = tf["mt5_const"]
        short_b = tf["bars_short"]

        print(f"\n=== Timeframe {tf_name} (short window {short_b} bars) ===")

    mt5.shutdown()

if __name__ == "__main__":
    main()