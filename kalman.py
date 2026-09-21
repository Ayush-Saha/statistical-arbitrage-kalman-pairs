import numpy as np


class KalmanHedgeRatio:
    """
    Online 1D Kalman Filter to track dynamic intercept and beta (hedge ratio).
    """

    def __init__(self, delta: float = 1e-4, R: float = 1e-3):
        self.delta = delta
        self.R = R
        self.theta = np.zeros(2)  # [alpha, beta]
        self.P = np.eye(2) * 1.0

    def update(self, x: float, y: float) -> tuple[float, float, float]:
        """
        Performs recursive Kalman measurement update for time t.
        """
        # State transition covariance
        Q = (self.delta / (1.0 - self.delta)) * np.eye(2)
        P_prior = self.P + Q

        # Observation matrix
        H = np.array([1.0, x])

        # Prediction and measurement error
        y_pred = np.dot(H, self.theta)
        error = y - y_pred

        # Kalman gain
        S = np.dot(H, np.dot(P_prior, H.T)) + self.R
        K = np.dot(P_prior, H.T) / S

        # Posterior state estimate
        self.theta = self.theta + K * error
        self.P = P_prior - np.outer(K, H).dot(P_prior)

        return float(self.theta[0]), float(self.theta[1]), float(error)