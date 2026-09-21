import numpy as np
import pandas as pd
from src.analytics import calculate_half_life, compute_performance_metrics
from src.kalman import KalmanHedgeRatio


def test_kalman_convergence():
    """Verify that Kalman filter converges to known slope (beta=2.0)."""
    kf = KalmanHedgeRatio(delta=1e-5, R=1e-2)
    x = np.linspace(1, 100, 500)
    y = 2.0 * x + 5.0 + np.random.normal(0, 0.1, 500)

    last_beta = 0.0
    for xt, yt in zip(x, y):
        _, b, _ = kf.update(xt, yt)
        last_beta = b

    assert abs(last_beta - 2.0) < 0.1


def test_half_life_mean_reversion():
    """Ensure AR(1) estimation returns valid positive half-life on stationary spread."""
    np.random.seed(1)
    stationary_spread = pd.Series(np.random.normal(0, 1, 500))
    hl = calculate_half_life(stationary_spread)
    assert hl > 0