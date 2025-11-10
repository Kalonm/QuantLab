import json
from pathlib import Path
import MetaTrader5 as mt5

# Root of the project (.. = go up from config/)
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"


def load_json(name: str):
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- TIMEFRAMES ----------

_timeframes_cfg = load_json("timeframes.json")["timeframes"]

# Map string names in JSON to actual mt5 constants
_MT5_TF_MAP = {
    "TIMEFRAME_M5": mt5.TIMEFRAME_M5,
    "TIMEFRAME_M10": mt5.TIMEFRAME_M10,
    "TIMEFRAME_M15": mt5.TIMEFRAME_M15,
    "TIMEFRAME_M20": mt5.TIMEFRAME_M20,
    "TIMEFRAME_M30": mt5.TIMEFRAME_M30,
    "TIMEFRAME_H1": mt5.TIMEFRAME_H1,
    "TIMEFRAME_H2": mt5.TIMEFRAME_H2,
    "TIMEFRAME_H3": mt5.TIMEFRAME_H3,
    "TIMEFRAME_H4": mt5.TIMEFRAME_H4,
    "TIMEFRAME_H6": mt5.TIMEFRAME_H6,
    "TIMEFRAME_H8": mt5.TIMEFRAME_H8,
    "TIMEFRAME_H12": mt5.TIMEFRAME_H12,
    "TIMEFRAME_D1": mt5.TIMEFRAME_D1,
    "TIMEFRAME_W1": mt5.TIMEFRAME_W1,
    "TIMEFRAME_MN1": mt5.TIMEFRAME_MN1,
}

# Build a cleaned list you can loop over in main
TIMEFRAMES = [
    {
        "name": tf["name"],                        # e.g. "H1"
        "mt5_const": _MT5_TF_MAP[tf["mt5_constant"]],
        "bars_short": tf["bars"]["short"],
        "bars_medium": tf["bars"]["medium"],
        "bars_long": tf["bars"]["long"],
    }
    for tf in _timeframes_cfg
]


# ---------- PAIRS ----------

_pairs_cfg = load_json("pairs.json")

PAIRS = [(p["symbol1"], p["symbol2"]) for p in _pairs_cfg]
