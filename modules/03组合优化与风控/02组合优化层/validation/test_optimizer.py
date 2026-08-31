from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VALIDATION_ROOT = ROOT / "validation"
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from optimization.alpha_risk_optimizer import AlphaRiskOptimizer
from optimization_validator import OptimizationValidator

WORKFLOW_SPEC = importlib.util.spec_from_file_location(
    "portfolio_optimization_workflow", ROOT / "optimization.py"
)
assert WORKFLOW_SPEC is not None and WORKFLOW_SPEC.loader is not None
WORKFLOW = importlib.util.module_from_spec(WORKFLOW_SPEC)
WORKFLOW_SPEC.loader.exec_module(WORKFLOW)


class OptimizerTests(unittest.TestCase):
    def test_rebalance_dates_require_a_completed_official_period(self) -> None:
        schedule = pd.DataFrame(
            {
                "cal_date": [
                    "20260126",
                    "20260127",
                    "20260128",
                    "20260129",
                    "20260130",
                    "20260202",
                    "20260203",
                    "20260204",
                    "20260205",
                    "20260206",
                ],
                "is_open": [1] * 10,
            }
        )
        available = pd.date_range("2026-01-26", "2026-02-05", freq="B")

        weekly = WORKFLOW.select_rebalance_dates(available, schedule, "weekly")
        monthly = WORKFLOW.select_rebalance_dates(available, schedule, "monthly")

        self.assertEqual(weekly, [pd.Timestamp("2026-01-30")])
        self.assertEqual(monthly, [pd.Timestamp("2026-01-30")])

    def test_optimizer_respects_constraints(self) -> None:
        alpha = np.array([2.0, 1.0, 0.0, -1.0])
        covariance = np.array(
            [
                [0.04, 0.01, 0.00, 0.00],
                [0.01, 0.03, 0.01, 0.00],
                [0.00, 0.01, 0.05, 0.01],
                [0.00, 0.00, 0.01, 0.02],
            ]
        )
        industries = np.array(["A", "A", "B", "B"])
        result = AlphaRiskOptimizer(1.0, 0.1, 0.0, 0.4, 0.7).solve(
            alpha, covariance, industries, target_exposure=0.45
        )
        frame = pd.DataFrame(
            {
                "signal_date": pd.Timestamp("2026-08-14"),
                "stock_code": ["1", "2", "3", "4"],
                "industry": industries,
                "alpha_zscore": alpha,
                "base_weight": result.weights,
                "target_weight": result.weights * 0.45,
            }
        )
        check = OptimizationValidator(0.0, 0.4, 0.7).validate(frame, 0.45, 0.45)
        self.assertEqual(check["status"], "passed")
        self.assertGreater(result.weights[0], result.weights[2])

    def test_account_constraints_reach_market_target_when_feasible(self) -> None:
        industries = np.array(["建筑"] * 16 + ["装修"] * 2 + ["化工", "钢铁"])
        alpha = np.linspace(2.0, -2.0, len(industries))
        covariance = np.eye(len(industries)) * 0.04
        result = AlphaRiskOptimizer(1.0, 0.1, 0.0, 0.10, 0.25).solve(
            alpha, covariance, industries, target_exposure=0.60
        )
        target = result.weights * result.diagnostics["deployed_stock_exposure"]
        self.assertAlmostEqual(result.diagnostics["deployed_stock_exposure"], 0.60)
        self.assertLessEqual(float(target.max()), 0.10 + 1e-8)
        self.assertLessEqual(float(target[industries == "建筑"].sum()), 0.25 + 1e-8)

    def test_infeasible_market_target_falls_back_to_cash(self) -> None:
        industries = np.array(["建筑"] * 19 + ["化工"])
        alpha = np.linspace(2.0, -2.0, len(industries))
        covariance = np.eye(len(industries)) * 0.04
        result = AlphaRiskOptimizer(1.0, 0.1, 0.0, 0.10, 0.25).solve(
            alpha, covariance, industries, target_exposure=0.60
        )
        deployed = result.diagnostics["deployed_stock_exposure"]
        target = result.weights * deployed
        self.assertAlmostEqual(deployed, 0.35)
        self.assertAlmostEqual(result.diagnostics["exposure_shortfall"], 0.25)
        self.assertLessEqual(float(target.max()), 0.10 + 1e-8)
        self.assertLessEqual(float(target[industries == "建筑"].sum()), 0.25 + 1e-8)

    def test_turnover_penalty_keeps_weights_closer_to_previous_period(self) -> None:
        alpha = np.array([-2.0, -1.0, 1.0, 2.0])
        covariance = np.eye(4) * 0.04
        industries = np.array(["A", "B", "C", "D"])
        previous = np.array([0.20, 0.15, 0.05, 0.05])
        without_penalty = AlphaRiskOptimizer(
            1.0, 0.3, 0.0, 0.30, 0.45, turnover_penalty=0.0
        ).solve(
            alpha,
            covariance,
            industries,
            target_exposure=0.45,
            previous_account_weights=previous,
        )
        with_penalty = AlphaRiskOptimizer(
            1.0, 0.3, 0.0, 0.30, 0.45, turnover_penalty=0.5
        ).solve(
            alpha,
            covariance,
            industries,
            target_exposure=0.45,
            previous_account_weights=previous,
        )
        turnover_without = np.abs(without_penalty.weights * 0.45 - previous).sum()
        turnover_with = np.abs(with_penalty.weights * 0.45 - previous).sum()
        self.assertLess(turnover_with, turnover_without)
        self.assertAlmostEqual(with_penalty.weights.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
