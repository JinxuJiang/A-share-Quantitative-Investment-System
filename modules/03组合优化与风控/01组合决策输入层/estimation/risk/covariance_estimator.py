from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class LedoitWolfCovarianceEstimator:
    METHOD = "ledoit_wolf"

    def __init__(self, horizon_days: int = 20):
        self.horizon_days = int(horizon_days)

    def fit(
        self,
        returns: pd.DataFrame,
        signal_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        if returns.empty or returns.isna().any().any():
            raise ValueError("Ledoit-Wolf输入收益矩阵为空或含缺失")
        model = LedoitWolf(assume_centered=False).fit(returns.to_numpy(dtype="float64"))
        daily_cov = model.covariance_
        horizon_cov = daily_cov * self.horizon_days
        assets = list(returns.columns)

        rows = []
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                rows.append(
                    {
                        "signal_date": pd.Timestamp(signal_date).normalize(),
                        "asset_i": asset_i,
                        "asset_j": asset_j,
                        "covariance_20d": float(horizon_cov[i, j]),
                        "estimator": self.METHOD,
                        "lookback_returns": int(len(returns)),
                        "horizon_days": self.horizon_days,
                    }
                )
        covariance = pd.DataFrame(rows)
        daily_vol = np.sqrt(np.clip(np.diag(daily_cov), 0.0, None))
        horizon_vol = np.sqrt(np.clip(np.diag(horizon_cov), 0.0, None))
        asset_risk = pd.DataFrame(
            {
                "signal_date": pd.Timestamp(signal_date).normalize(),
                "stock_code": assets,
                "volatility_daily": daily_vol,
                "volatility_20d": horizon_vol,
                "valid_observations": len(returns),
                "estimator": self.METHOD,
                "horizon_days": self.horizon_days,
            }
        )
        diagnostics = {
            "estimator": self.METHOD,
            "shrinkage": float(model.shrinkage_),
            "n_assets": len(assets),
            "n_observations": len(returns),
            "min_eigenvalue_20d": float(np.linalg.eigvalsh(horizon_cov).min()),
            "max_eigenvalue_20d": float(np.linalg.eigvalsh(horizon_cov).max()),
        }
        return covariance, asset_risk, diagnostics
