# utils/rolling_ols.py
from __future__ import annotations
import numpy as np
import pandas as pd

def rolling_ols_alpha_beta(
    x: pd.Series, 
    y: pd.Series, 
    window: int, 
    min_periods: int | None = None
) -> pd.DataFrame:
    """
    Rolling OLS of y ~ alpha + beta * x using sum identities.
      beta = Cov(x,y)/Var(x)
      alpha = E[y] - beta * E[x]
    Returns DataFrame with columns: alpha, beta, r2 (optional but handy).
    """
    if min_periods is None:
        min_periods = window

    x = x.astype(float)
    y = y.astype(float)

    # rolling sums (vectorized & fast; uses pandas' C engine)
    Sx  = x.rolling(window, min_periods=min_periods).sum()
    Sy  = y.rolling(window, min_periods=min_periods).sum()
    Sxx = (x*x).rolling(window, min_periods=min_periods).sum()
    Syy = (y*y).rolling(window, min_periods=min_periods).sum()
    Sxy = (x*y).rolling(window, min_periods=min_periods).sum()

    n = x.rolling(window, min_periods=min_periods).count()

    # means
    Ex = Sx / n
    Ey = Sy / n

    # centered sums
    VarX = Sxx - (Sx*Sx)/n
    CovXY = Sxy - (Sx*Sy)/n
    VarY = Syy - (Sy*Sy)/n

    beta = CovXY / VarX
    alpha = Ey - beta * Ex

    # optional R^2 (clipped to [0,1])
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = (CovXY*CovXY) / (VarX * VarY)
    r2 = r2.clip(lower=0, upper=1)

    out = pd.DataFrame({"alpha": alpha, "beta": beta, "r2": r2})
    return out

def rolling_spread_and_z(
    x: pd.Series,
    y: pd.Series,
    alpha: pd.Series,
    beta: pd.Series,
    z_window: int | None = None,
    min_periods: int | None = None
) -> pd.DataFrame:
    """
    Build spread(t) = y(t) - (alpha(t-1) + beta(t-1)*x(t))  (use lagged params)
    Then compute rolling z-score on spread with window=z_window (defaults to same as param window).
    """
    if min_periods is None:
        min_periods = z_window
    # use lagged parameters to prevent look-ahead
    a = alpha.shift(1)
    b = beta.shift(1)

    spread = y - (a + b * x)

    if z_window is None:
        z_window = int(pd.notna(a).rolling(100000, min_periods=1).sum().iloc[-1])  # fallback
    mu = spread.rolling(z_window, min_periods=min_periods).mean()
    sd = spread.rolling(z_window, min_periods=min_periods).std()
    z = (spread - mu) / sd

    return pd.DataFrame({"spread": spread, "mu": mu, "sd": sd, "z": z})
