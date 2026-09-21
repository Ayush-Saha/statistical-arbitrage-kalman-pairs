import numpy as np
import pandas as pd
from analytics import (
    calculate_half_life,
    compute_performance_metrics,
    verify_cointegration,
)
from kalman import KalmanHedgeRatio


def run_pipeline():
    np.random.seed(42)
    n_bars = 1000

    # Simulate cointegrated price processes
    latent_factor = np.cumsum(np.random.normal(0, 1, n_bars))
    asset_x = 100 + latent_factor + np.random.normal(0, 0.5, n_bars)
    asset_y = 50 + 1.6 * latent_factor + np.random.normal(0, 1.2, n_bars)

    df = pd.DataFrame({"Asset_X": asset_x, "Asset_Y": asset_y})

    # 1. Cointegration Check
    p_value = verify_cointegration(df["Asset_Y"], df["Asset_X"])
    print(f"[Engle-Granger Test] Cointegration p-value: {p_value:.5f}")

    # 2. Online Kalman Dynamic Hedge Ratio
    kf = KalmanHedgeRatio(delta=1e-5, R=1e-2)
    alphas, betas, spreads = [], [], []

    for x_t, y_t in zip(df["Asset_X"], df["Asset_Y"]):
        a, b, spr = kf.update(x_t, y_t)
        alphas.append(a)
        betas.append(b)
        spreads.append(spr)

    df["Beta"] = betas
    df["Spread"] = spreads

    # 3. Half-Life & Z-Score
    hl = calculate_half_life(df["Spread"])
    lookback = int(max(5, round(hl))) if not np.isnan(hl) else 20
    print(f"[O-U Dynamics] Calculated Half-Life: {hl:.2f} bars (Window: {lookback})")

    roll_mean = df["Spread"].rolling(window=lookback).mean()
    roll_std = df["Spread"].rolling(window=lookback).std()
    df["Z_Score"] = (df["Spread"] - roll_mean) / roll_std

    # 4. Signal Generation (|z| > 1.5 entry, |z| < 0.2 exit)
    df["Position"] = 0
    pos = 0
    positions = []

    for z in df["Z_Score"]:
        if np.isnan(z):
            positions.append(0)
            continue
        if pos == 0:
            if z > 1.5:
                pos = -1
            elif z < -1.5:
                pos = 1
        elif pos == 1 and z >= -0.2:
            pos = 0
        elif pos == -1 and z <= 0.2:
            pos = 0
        positions.append(pos)

    df["Position"] = positions

    # 5. P&L Accounting with 7 bps round-trip friction
    ret_x = np.log(df["Asset_X"] / df["Asset_X"].shift(1)).fillna(0)
    ret_y = np.log(df["Asset_Y"] / df["Asset_Y"].shift(1)).fillna(0)

    lagged_pos = df["Position"].shift(1).fillna(0)
    gross_ret = lagged_pos * (ret_y - df["Beta"].shift(1) * ret_x)

    # 5 bps slippage + 2 bps transaction fees
    trades = df["Position"].diff().abs().fillna(0)
    friction = trades * 0.0007
    net_ret = gross_ret - friction

    metrics = compute_performance_metrics(net_ret)
    print("\n--- Strategy Performance Tear-Sheet ---")
    for k, v in metrics.items():
        if "Ratio" in k:
            print(f"{k:24}: {v:.2f}")
        else:
            print(f"{k:24}: {v*100:.2f}%")


if __name__ == "__main__":
    run_pipeline()