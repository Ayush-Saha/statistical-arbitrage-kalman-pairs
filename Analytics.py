import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def verify_cointegration(s1: pd.Series, s2: pd.Series) -> float:
    """Calculates Engle-Granger p-value."""
    _, p_val, _ = coint(s1, s2)
    return float(p_val)


def calculate_half_life(spread: pd.Series) -> float:
    """Estimates Ornstein-Uhlenbeck mean-reversion half-life via AR(1) OLS."""
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    df = pd.DataFrame({"diff": spread_diff, "lag": spread_lag}).dropna()

    model = sm.OLS(df["diff"], sm.add_constant(df["lag"])).fit()
    gamma = model.params["lag"]

    if gamma >= 0:
        return np.nan
    return float(-np.log(2) / gamma)


def compute_performance_metrics(
    returns: pd.Series, annualization_factor: int = 252
) -> dict:
    """Calculates standard quantitative tear-sheet metrics."""
    cum_returns = (1.0 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max

    mean_ret = returns.mean() * annualization_factor
    vol = returns.std() * np.sqrt(annualization_factor)
    sharpe = mean_ret / vol if vol != 0 else 0.0

    neg_vol = returns[returns < 0].std() * np.sqrt(annualization_factor)
    sortino = mean_ret / neg_vol if neg_vol != 0 else 0.0

    return {
        "Annualized Return": mean_ret,
        "Annualized Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": drawdown.min(),
    }