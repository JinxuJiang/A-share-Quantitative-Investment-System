from __future__ import annotations

import numpy as np
from scipy.stats import norm


class ParametricNormalVarEs:
    """基于正态分布和协方差矩阵计算组合VaR与ES。

    VaR和ES都表示正数形式的损失比例。第一版预期收益设为0，不把截面
    alpha_zscore误当作可直接相加的收益率预测。
    """

    METHOD = "parametric_normal"

    def __init__(self, confidence_levels: list[float], expected_return: float = 0.0):
        levels = sorted(float(level) for level in confidence_levels)
        if not levels or any(level <= 0.5 or level >= 1.0 for level in levels):
            raise ValueError("置信水平必须位于(0.5, 1.0)")
        if len(set(levels)) != len(levels):
            raise ValueError("置信水平不能重复")
        self.confidence_levels = levels
        self.expected_return = float(expected_return)

    def calculate(self, weights: np.ndarray, covariance: np.ndarray) -> dict:
        weight = np.asarray(weights, dtype="float64")
        cov = np.asarray(covariance, dtype="float64")
        if weight.ndim != 1 or cov.shape != (len(weight), len(weight)):
            raise ValueError("权重和协方差维度不一致")
        if not np.isfinite(weight).all() or not np.isfinite(cov).all():
            raise ValueError("权重或协方差包含NaN或无穷值")

        variance = float(weight @ cov @ weight)
        if variance < -1e-12:
            raise ValueError(f"组合方差为负数: {variance}")
        volatility = float(np.sqrt(max(variance, 0.0)))
        metrics: dict[str, dict] = {}
        for confidence in self.confidence_levels:
            z_score = float(norm.ppf(confidence))
            tail_factor = float(norm.pdf(z_score) / (1.0 - confidence))
            value_at_risk = z_score * volatility - self.expected_return
            expected_shortfall = tail_factor * volatility - self.expected_return
            label = f"{confidence:.0%}"
            metrics[label] = {
                "confidence_level": confidence,
                "z_score": z_score,
                "value_at_risk": float(value_at_risk),
                "expected_shortfall": float(expected_shortfall),
                "value_at_risk_pct": round(value_at_risk * 100.0, 2),
                "expected_shortfall_pct": round(expected_shortfall * 100.0, 2),
            }
        return {
            "method": self.METHOD,
            "expected_return": self.expected_return,
            "portfolio_variance": variance,
            "portfolio_volatility": volatility,
            "portfolio_volatility_pct": round(volatility * 100.0, 2),
            "metrics": metrics,
        }


def calculate_var_scale(value_at_risk: float, var_budget: float) -> float:
    """返回只减仓、不加杠杆的 VaR 仓位缩放系数。"""
    current_var = float(value_at_risk)
    budget = float(var_budget)
    if not np.isfinite(current_var) or current_var < 0.0:
        raise ValueError("VaR 必须是有限的非负数")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("VaR 预算必须是有限的正数")
    if current_var == 0.0:
        return 1.0
    return float(min(1.0, budget / current_var))
