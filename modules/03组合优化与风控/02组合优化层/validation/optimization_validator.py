from __future__ import annotations

import numpy as np
import pandas as pd


class OptimizationValidator:
    def __init__(
        self,
        min_stock_weight: float,
        max_stock_weight: float,
        max_industry_weight: float,
        tolerance: float = 1e-8,
    ):
        self.min_stock_weight = float(min_stock_weight)
        self.max_stock_weight = float(max_stock_weight)
        self.max_industry_weight = float(max_industry_weight)
        self.tolerance = float(tolerance)

    def validate(
        self,
        weights: pd.DataFrame,
        market_target_exposure: float,
        deployed_stock_exposure: float,
    ) -> dict:
        required = {
            "signal_date", "stock_code", "industry", "alpha_zscore",
            "base_weight", "target_weight",
        }
        missing = required - set(weights.columns)
        if missing:
            raise ValueError(f"优化权重缺少字段: {sorted(missing)}")
        if weights.empty or weights["stock_code"].duplicated().any():
            raise ValueError("优化权重为空或包含重复股票")
        numeric = weights[["alpha_zscore", "base_weight", "target_weight"]].to_numpy(dtype="float64")
        if not np.isfinite(numeric).all():
            raise ValueError("优化权重包含NaN或无穷值")

        base = weights["base_weight"].to_numpy(dtype="float64")
        target = weights["target_weight"].to_numpy(dtype="float64")
        if not np.isclose(base.sum(), 1.0, atol=self.tolerance):
            raise ValueError(f"内部权重和不等于1: {base.sum()}")
        if (target < self.min_stock_weight - self.tolerance).any():
            raise ValueError("存在低于账户单票最小约束的权重")
        if (target > self.max_stock_weight + self.tolerance).any():
            raise ValueError("存在超过账户单票最大约束的权重")
        if not np.allclose(target, base * deployed_stock_exposure, atol=self.tolerance):
            raise ValueError("账户权重不等于内部权重乘实际股票仓位")
        if not np.isclose(target.sum(), deployed_stock_exposure, atol=self.tolerance):
            raise ValueError("股票账户权重和不等于实际部署仓位")
        if deployed_stock_exposure > market_target_exposure + self.tolerance:
            raise ValueError("实际股票仓位超过市场目标上限")

        industry_weight = weights.groupby("industry", dropna=False)["target_weight"].sum()
        if (industry_weight > self.max_industry_weight + self.tolerance).any():
            bad = industry_weight[industry_weight > self.max_industry_weight + self.tolerance]
            raise ValueError(f"行业权重超过上限: {bad.to_dict()}")
        return {
            "status": "passed",
            "asset_count": int(len(weights)),
            "base_weight_sum": float(base.sum()),
            "target_weight_sum": float(target.sum()),
            "market_target_exposure": float(market_target_exposure),
            "deployed_stock_exposure": float(deployed_stock_exposure),
            "exposure_shortfall": float(market_target_exposure - deployed_stock_exposure),
            "max_stock_weight": float(target.max()),
            "max_industry_weight": float(industry_weight.max()),
        }
