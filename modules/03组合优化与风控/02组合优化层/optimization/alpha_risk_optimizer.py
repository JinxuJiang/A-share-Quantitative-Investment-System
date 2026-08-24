from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    alpha_utility: np.ndarray
    diagnostics: dict


class AlphaRiskOptimizer:
    """在候选股票内平衡Alpha偏好和协方差风险。

    Alpha是截面效用而不是收益率，因此先在候选股票内标准化；组合风险则除以
    等权组合方差。两个目标都变为无量纲后，配置中的权重才不会被原始单位支配。
    """

    METHOD = "scaled_alpha_risk_slsqp"

    def __init__(
        self,
        risk_weight: float,
        alpha_weight: float,
        min_stock_weight: float,
        max_stock_weight: float,
        max_industry_weight: float,
        turnover_penalty: float = 0.0,
        turnover_smoothing: float = 1.0e-6,
        max_iterations: int = 2000,
        tolerance: float = 1e-12,
    ):
        self.risk_weight = float(risk_weight)
        self.alpha_weight = float(alpha_weight)
        self.min_stock_weight = float(min_stock_weight)
        self.max_stock_weight = float(max_stock_weight)
        self.max_industry_weight = float(max_industry_weight)
        self.turnover_penalty = float(turnover_penalty)
        self.turnover_smoothing = float(turnover_smoothing)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def solve(
        self,
        alpha_zscore: np.ndarray,
        covariance: np.ndarray,
        industries: np.ndarray,
        target_exposure: float = 1.0,
        previous_account_weights: np.ndarray | None = None,
        previous_outside_weight: float = 0.0,
    ) -> OptimizationResult:
        alpha = np.asarray(alpha_zscore, dtype="float64")
        cov = np.asarray(covariance, dtype="float64")
        industry = np.asarray(industries, dtype=str)
        n_assets = len(alpha)
        previous = (
            np.zeros(n_assets, dtype="float64")
            if previous_account_weights is None
            else np.asarray(previous_account_weights, dtype="float64")
        )
        outside_weight = float(previous_outside_weight)
        if n_assets < 2 or cov.shape != (n_assets, n_assets) or len(industry) != n_assets:
            raise ValueError("Alpha、协方差和行业数组的维度不一致")
        if not np.isfinite(alpha).all() or not np.isfinite(cov).all():
            raise ValueError("优化输入包含NaN或无穷值")
        if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
            raise ValueError("协方差矩阵不对称")
        requested_exposure = float(target_exposure)
        if previous.shape != (n_assets,) or not np.isfinite(previous).all():
            raise ValueError("上一期账户权重维度错误或包含非有限值")
        if (previous < 0.0).any() or outside_weight < 0.0:
            raise ValueError("上一期账户权重不能为负")
        if self.turnover_penalty < 0.0 or self.turnover_smoothing <= 0.0:
            raise ValueError("换手惩罚必须非负，平滑参数必须为正")
        if not 0.0 <= requested_exposure <= 1.0:
            raise ValueError("市场目标仓位必须位于[0, 1]")
        if not 0.0 <= self.min_stock_weight <= self.max_stock_weight <= 1.0:
            raise ValueError("账户单票权重约束无效")
        if not 0.0 < self.max_industry_weight <= 1.0:
            raise ValueError("账户行业权重上限必须位于(0, 1]")

        alpha_std = float(alpha.std(ddof=0))
        if alpha_std <= 0.0:
            raise ValueError("候选股票Alpha没有截面差异")
        alpha_utility = (alpha - float(alpha.mean())) / alpha_std

        industry_indices = {
            name: np.flatnonzero(industry == name) for name in np.unique(industry)
        }
        account_capacity = float(
            min(
                1.0,
                sum(
                    min(len(indices) * self.max_stock_weight, self.max_industry_weight)
                    for indices in industry_indices.values()
                ),
            )
        )
        deployed_exposure = min(requested_exposure, account_capacity)
        exposure_shortfall = requested_exposure - deployed_exposure

        # 市场目标为0时没有实际股票投资，内部权重仅保留为可审计的等权占位。
        if deployed_exposure <= 1e-15:
            equal_weight = np.full(n_assets, 1.0 / n_assets, dtype="float64")
            portfolio_variance = float(equal_weight @ cov @ equal_weight)
            diagnostics = {
                "method": self.METHOD,
                "solver": "not_required_zero_exposure",
                "success": True,
                "status": 0,
                "message": "zero market exposure",
                "iterations": 0,
                "objective_value": 0.0,
                "reference_equal_weight_variance_20d": portfolio_variance,
                "optimized_variance_20d": portfolio_variance,
                "optimized_volatility_20d": float(np.sqrt(max(portfolio_variance, 0.0))),
                "alpha_utility_exposure": float(alpha_utility @ equal_weight),
                "market_target_exposure": requested_exposure,
                "deployed_stock_exposure": 0.0,
                "maximum_feasible_exposure": account_capacity,
                "exposure_shortfall": exposure_shortfall,
                "turnover_penalty": self.turnover_penalty,
                "estimated_trade_weight": float(previous.sum() + outside_weight),
                "turnover_penalty_value": float(
                    self.turnover_penalty * (previous.sum() + outside_weight)
                ),
            }
            return OptimizationResult(equal_weight, alpha_utility, diagnostics)

        # 优化变量仍是股票仓位内部归一化权重，但所有硬约束来自账户实际权重。
        # 例如目标仓位60%、账户单票上限10%，等价的内部单票上限为1/6。
        internal_min_weight = self.min_stock_weight / deployed_exposure
        internal_max_weight = min(1.0, self.max_stock_weight / deployed_exposure)
        internal_max_industry = min(1.0, self.max_industry_weight / deployed_exposure)
        if internal_min_weight * n_assets > 1.0 + 1e-12:
            raise ValueError("账户单票最小权重导致目标仓位不可行")
        if internal_max_weight * n_assets < 1.0 - 1e-12:
            raise ValueError("账户单票最大权重导致目标仓位不可行")

        equal_weight = np.full(n_assets, 1.0 / n_assets, dtype="float64")
        reference_variance = float(equal_weight @ cov @ equal_weight)
        if reference_variance <= 0.0:
            raise ValueError("等权组合方差必须为正数")

        def objective(weights: np.ndarray) -> float:
            risk_ratio = float(weights @ cov @ weights) / reference_variance
            alpha_reward = float(alpha_utility @ weights)
            difference = deployed_exposure * weights - previous
            smooth_turnover = float(
                np.sum(
                    np.sqrt(difference * difference + self.turnover_smoothing**2)
                    - self.turnover_smoothing
                )
                + outside_weight
            )
            return (
                self.risk_weight * risk_ratio
                - self.alpha_weight * alpha_reward
                + self.turnover_penalty * smooth_turnover
            )

        def gradient(weights: np.ndarray) -> np.ndarray:
            risk_gradient = 2.0 * (cov @ weights) / reference_variance
            difference = deployed_exposure * weights - previous
            turnover_gradient = (
                deployed_exposure
                * difference
                / np.sqrt(difference * difference + self.turnover_smoothing**2)
            )
            return (
                self.risk_weight * risk_gradient
                - self.alpha_weight * alpha_utility
                + self.turnover_penalty * turnover_gradient
            )

        constraints: list[dict] = [
            {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}
        ]
        for indices in industry_indices.values():
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, idx=indices: float(
                        internal_max_industry - weights[idx].sum()
                    ),
                }
            )

        equal_is_feasible = (
            equal_weight.max() <= internal_max_weight + 1e-12
            and equal_weight.min() >= internal_min_weight - 1e-12
            and all(
                equal_weight[indices].sum() <= internal_max_industry + 1e-12
                for indices in industry_indices.values()
            )
        )
        if equal_is_feasible:
            initial_weight = equal_weight
        else:
            feasibility = linprog(
                np.zeros(n_assets),
                A_ub=np.vstack(
                    [
                        np.isin(np.arange(n_assets), indices).astype("float64")
                        for indices in industry_indices.values()
                    ]
                ),
                b_ub=np.full(len(industry_indices), internal_max_industry),
                A_eq=np.ones((1, n_assets)),
                b_eq=np.array([1.0]),
                bounds=[(internal_min_weight, internal_max_weight)] * n_assets,
                method="highs",
            )
            if not feasibility.success:
                raise RuntimeError(f"账户仓位约束不可行: {feasibility.message}")
            initial_weight = np.asarray(feasibility.x, dtype="float64")

        result = minimize(
            objective,
            initial_weight,
            method="SLSQP",
            jac=gradient,
            bounds=[(internal_min_weight, internal_max_weight)] * n_assets,
            constraints=constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"组合优化未收敛: {result.message}")

        weights = np.asarray(result.x, dtype="float64")
        portfolio_variance = float(weights @ cov @ weights)
        estimated_trade_weight = float(
            np.abs(deployed_exposure * weights - previous).sum() + outside_weight
        )
        diagnostics = {
            "method": self.METHOD,
            "solver": "SLSQP",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective_value": float(result.fun),
            "reference_equal_weight_variance_20d": reference_variance,
            "optimized_variance_20d": portfolio_variance,
            "optimized_volatility_20d": float(np.sqrt(max(portfolio_variance, 0.0))),
            "alpha_utility_exposure": float(alpha_utility @ weights),
            "market_target_exposure": requested_exposure,
            "deployed_stock_exposure": deployed_exposure,
            "maximum_feasible_exposure": account_capacity,
            "exposure_shortfall": exposure_shortfall,
            "turnover_penalty": self.turnover_penalty,
            "estimated_trade_weight": estimated_trade_weight,
            "turnover_penalty_value": self.turnover_penalty * estimated_trade_weight,
        }
        return OptimizationResult(weights, alpha_utility, diagnostics)
