from __future__ import annotations

import numpy as np
import pandas as pd


class RiskValidator:
    def __init__(self, weight_tolerance: float = 1e-8, psd_tolerance: float = 1e-10):
        self.weight_tolerance = float(weight_tolerance)
        self.psd_tolerance = float(psd_tolerance)

    def validate_inputs(
        self,
        weights: pd.DataFrame,
        covariance: pd.DataFrame,
        signal_date: pd.Timestamp,
        market_exposure: float,
        covariance_column: str,
        horizon_days: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        required = {"signal_date", "stock_code", "target_weight"}
        missing = required - set(weights.columns)
        if missing:
            raise ValueError(f"组合权重缺少字段: {sorted(missing)}")
        if weights.empty or weights["stock_code"].duplicated().any():
            raise ValueError("组合权重为空或包含重复股票")
        date = pd.Timestamp(signal_date).normalize()
        if not pd.to_datetime(weights["signal_date"]).dt.normalize().eq(date).all():
            raise ValueError("组合权重的signal_date与版本日期不一致")

        codes = weights["stock_code"].astype(str).tolist()
        target_weight = weights["target_weight"].to_numpy(dtype="float64")
        if not np.isfinite(target_weight).all() or (target_weight < -self.weight_tolerance).any():
            raise ValueError("target_weight包含无效值或负权重")
        if not np.isclose(target_weight.sum(), market_exposure, atol=self.weight_tolerance):
            raise ValueError("target_weight之和不等于市场总仓位")

        required_covariance = {"asset_i", "asset_j", covariance_column}
        missing_covariance = required_covariance - set(covariance.columns)
        if missing_covariance:
            raise ValueError(f"协方差缺少字段: {sorted(missing_covariance)}")
        matrix = covariance.pivot(
            index="asset_i", columns="asset_j", values=covariance_column
        ).reindex(index=codes, columns=codes)
        values = matrix.to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("协方差股票集合与权重股票集合不一致")
        if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
            raise ValueError("协方差矩阵不对称")
        min_eigenvalue = float(np.linalg.eigvalsh(values).min())
        if min_eigenvalue < -self.psd_tolerance:
            raise ValueError(f"协方差矩阵不是半正定，最小特征值={min_eigenvalue}")
        return target_weight, values, {
            "status": "passed",
            "asset_count": len(codes),
            "target_weight_sum": float(target_weight.sum()),
            f"min_eigenvalue_{horizon_days}d": min_eigenvalue,
        }

    def validate_results(self, result: dict) -> dict:
        volatility = float(result["portfolio_volatility"])
        if not np.isfinite(volatility) or volatility < 0.0:
            raise ValueError("组合波动率无效")
        previous_var = -np.inf
        previous_es = -np.inf
        for label, metric in result["metrics"].items():
            value_at_risk = float(metric["value_at_risk"])
            expected_shortfall = float(metric["expected_shortfall"])
            if not np.isfinite([value_at_risk, expected_shortfall]).all():
                raise ValueError(f"{label} VaR/ES包含NaN或无穷值")
            if expected_shortfall + 1e-12 < value_at_risk:
                raise ValueError(f"{label} ES小于VaR")
            if value_at_risk + 1e-12 < previous_var or expected_shortfall + 1e-12 < previous_es:
                raise ValueError("更高置信水平的VaR/ES反而更低")
            previous_var, previous_es = value_at_risk, expected_shortfall
        return {"status": "passed", "confidence_levels": list(result["metrics"])}
