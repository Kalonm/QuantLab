# Technical Prerequisites

Ensure the following conditions are met before onboarding new contributors or deploying QuantLab workflows.

## Workstation Requirements

- **Operating system**: Windows 10/11, macOS 13+, or Ubuntu 22.04 with GUI access for MetaTrader 5.
- **Python**: Version 3.11 or newer with `venv` module available.
- **Hardware**: Minimum 16 GB RAM and SSD storage capable of holding several gigabytes of historical CSV data.
- **Network**: Stable, low-latency internet connection for broker connectivity and package installations.

## Software Dependencies

1. MetaTrader 5 desktop terminal installed and authenticated with a broker account that exposes the target symbols.
2. Python dependencies from `requirements.txt`, ideally installed inside a virtual environment.
3. Optional analytics stack:
   - JupyterLab or VS Code with the Python extension for notebook-driven research.
   - Docker Desktop if containerising pipelines for CI runs.

## Access & Credentials

- Broker demo or production credentials with trading-read-only permissions.
- GitHub (or equivalent) access to the QuantLab repository.
- Shared secrets vault (1Password, Bitwarden, or HashiCorp Vault) for API keys used in downstream integrations.

## Verification Checklist

- [ ] MT5 terminal launches and can subscribe to at least one configured pair.
- [ ] `python -m pip install -r requirements.txt` completes without errors.
- [ ] Sample data fetch (`python utils/fetch_data.py --dry-run`) succeeds against the configured account.
- [ ] Backtest script (`python utils/backtest_pairs_rolling.py`) completes on a representative pair.
