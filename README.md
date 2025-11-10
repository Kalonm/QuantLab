# QuantLab

QuantLab is a research sandbox for quantitative pairs-trading workflows on top of MetaTrader 5 (MT5). The project focuses on three building blocks:

1. **Instrument & timeframe configuration** defined in JSON so you can drive experiments without touching code.
2. **Data collection** utilities that download OHLCV bars from a running MT5 terminal and persist them as CSV files.
3. **Rolling OLS backtesting** helpers for evaluating dollar-neutral spread trades with realistic execution rules.

The repository is deliberately modular – most scripts can be executed standalone or imported into notebooks for further analysis.

## Project layout

| Path | Description |
| --- | --- |
| `main.py` | Minimal example that iterates over the configured timeframes once MT5 is initialised. |
| `pipeline_runner.py` | Roadmap-aligned modelling pipeline covering foundational supervised learning baselines. |
| `config/timeframes.json` | Declares the bar windows and MT5 constants used throughout the toolkit. |
| `config/pairs.json` | Large universe of symbol pairs to scan/backtest. |
| `config/settings.py` | Loads JSON config and exposes `TIMEFRAMES` and `PAIRS` for other modules. |
| `utils/fetch_data.py` | Downloads & updates per-symbol CSV files for every configured timeframe. |
| `utils/backtest_pairs_rolling.py` | Runs a rolling OLS hedge/backtest on stored price data and writes performance series. |
| `data/` | Output directory populated with `raw/` price history and `backtests/` results. |

## Requirements

- Python 3.11+ (the included dependency pins were generated on 3.11).
- Local installation of the **MetaTrader 5 desktop terminal** logged into a broker account that serves the requested symbols.
- `MetaTrader5` Python package (installed automatically from `requirements.txt`).

> ⚠️ The MetaTrader 5 package only works on Windows/macOS/Linux when the native terminal is present. Running the scripts headless on a server without MT5 will fail at runtime.

## Getting started

```bash
# 1. Create an isolated environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. (One-off) Launch the MT5 terminal, log in, and keep it running
```

## Configure the universe

1. **Timeframes** – edit `config/timeframes.json` to change the MT5 timeframe constant or the bar lookback windows you want to store/backtest. Each entry exposes the timeframe name, the MetaTrader constant, and three window lengths (short/medium/long) that can be reused by analytics scripts.
2. **Pairs** – curate the symbol combinations in `config/pairs.json`. The list can be large; you can comment/remove entries to speed up downloads and tests. The loader in `config/settings.py` turns the JSON file into a list of `(symbol1, symbol2)` tuples.

Whenever you change these files, the utilities will automatically pick up the updates the next time they run.

## Fetch historical data

Use the data ingestor to build or refresh the CSV cache:

```bash
python utils/fetch_data.py
```

The script will:

1. Initialise MT5 once.
2. For each configured timeframe, iterate through the unique symbols implied by `PAIRS`.
3. Append new bars to `data/raw/<TIMEFRAME>/<SYMBOL>.csv`, creating folders as needed.

Existing files are deduplicated by timestamp so re-running the script is safe. Set `DEFAULT_LOOKBACK_DAYS` inside `utils/fetch_data.py` if you need a longer initial history.

## Run a rolling OLS backtest

With data on disk, you can evaluate a spread strategy programmatically. For example:

```python
from utils.backtest_pairs_rolling import load_pair, run_single_backtest

# Choose timeframe and symbols that have data
bt_path = run_single_backtest(
    tf_name="H1",
    sym1="XAUUSD",
    sym2="XAGUSD",
    ols_window=250,
    z_window=250,
    z_entry=2.0,
    z_exit=0.5,
    z_stop=3.5,
    tx_bps_per_leg=0.5,
)
print(f"Results saved to {bt_path}")
```

The helper returns a CSV in `data/backtests/` containing the rolling hedge parameters, positions, PnL decomposition, and cumulative equity curve for the pair.

## Run the roadmap modelling pipeline

The `pipeline_runner.py` script begins operationalising the research roadmap by focussing on the "Foundational supervised learning" workstream. It performs four stages: CSV ingestion, numeric feature preparation, baseline regression models (OLS, Ridge, LASSO, Elastic Net), and metric reporting.

```bash
python pipeline_runner.py \
  --dataset data/processed/pair_scan_summary.csv \
  --target half_life_bars \
  --output reports/baseline_regression_metrics.json
```

The output JSON includes timestamped metrics for each baseline model so future stages (tree ensembles, GARCH models, risk analytics, etc.) can be compared as they come online.

## Tips

- Store raw data under version control sparingly; the `data/` directory can grow quickly.
- Wrap scripts with schedulers (e.g., `cron`) if you want to refresh intraday data continuously – the fetcher is idempotent.
- Use the `TIMEFRAMES` metadata exposed by `config.settings` when writing notebooks so you remain in sync with the project configuration.

Happy researching!
