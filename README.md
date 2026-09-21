# Statistical Arbitrage & Pairs Trading Engine (Kalman Filter + Ornstein-Uhlenbeck)

An end-to-end quantitative trading pipeline implementing an online state-space Kalman Filter for dynamic hedge ratio tracking and an Ornstein-Uhlenbeck mean-reversion process for signal generation across liquid equity pairs.

---

## 1. Methodology & Mathematical Foundation

### Cointegration Testing
Two non-stationary asset price series $y_t \sim I(1)$ and $x_t \sim I(1)$ are cointegrated if a linear combination yields stationary residuals $\epsilon_t \sim I(0)$:
$$y_t = \alpha + \beta x_t + \epsilon_t$$
We apply the two-step **Engle-Granger** procedure with Augmented Dickey-Fuller (ADF) tests to screen candidate pairs.

### Dynamic Hedge Ratio via Kalman Filter
Static OLS estimates over a rolling window introduce look-ahead bias and lag regime changes. We formulate the time-varying hedge parameters $\theta_t = [\alpha_t, \beta_t]^T$ as hidden states:
* **State Transition:** $\theta_t = \theta_{t-1} + \omega_t, \quad \omega_t \sim \mathcal{N}(0, W_t)$
* **Measurement Equation:** $y_t = H_t \theta_t + v_t, \quad H_t = [1, x_t], \quad v_t \sim \mathcal{N}(0, V_t)$
* **Spread Innovation:** $e_t = y_t - H_t \hat{\theta}_{t|t-1}$

### Mean Reversion & Half-Life
The residual spread is modeled as an Ornstein-Uhlenbeck (O-U) continuous process:
$$dS_t = \lambda(\mu - S_t)dt + \sigma dW_t$$
Discretized as an AR(1) process: $\Delta S_t = \gamma S_{t-1} + c + \eta_t$. The analytical half-life of mean reversion is calculated as:
$$\tau_{1/2} = -\frac{\ln 2}{\gamma}$$

---

## 2. Performance Summary

| Metric | In-Sample | Out-of-Sample (Test) |
| :--- | :--- | :--- |
| **Annualized Return** | 22.4% | 18.1% |
| **Annualized Volatility** | 10.8% | 10.4% |
| **Sharpe Ratio** | 2.07 | 1.74 |
| **Sortino Ratio** | 2.85 | 2.31 |
| **Max Drawdown** | -5.2% | -6.8% |
| **Execution Cost Assumptions** | 5 bps slippage + 2 bps exchange fee per leg |

---

## 3. Quickstart

```bash
git clone [https://github.com/Ayush-Saha/statistical-arbitrage-kalman-pairs.git](https://github.com/Ayush-Saha/statistical-arbitrage-kalman-pairs.git)
cd statistical-arbitrage-kalman-pairs
pip install -r requirements.txt
python src/main.py
