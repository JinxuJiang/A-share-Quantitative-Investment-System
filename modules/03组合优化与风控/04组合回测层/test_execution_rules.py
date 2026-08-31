from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("backtrader.eval.py")
SPEC = importlib.util.spec_from_file_location("portfolio_backtest", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"无法加载回测模块: {MODULE_PATH}")
BACKTEST_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKTEST_MODULE)
should_skip_minimum_rebalance = BACKTEST_MODULE.should_skip_minimum_rebalance
apply_risk_scaling = BACKTEST_MODULE.apply_risk_scaling


class MinimumRebalanceTests(unittest.TestCase):
    def test_risk_scaling_preserves_relative_weights(self) -> None:
        date = pd.Timestamp("2026-07-31")
        weights = pd.DataFrame(
            {
                "signal_date": [date, date],
                "stock_code": ["000001.XSHE", "000002.XSHE"],
                "target_weight": [0.30, 0.60],
            }
        )
        risk = pd.DataFrame({"signal_date": [date], "risk_scale": [0.75]})
        scaled = apply_risk_scaling(weights, risk, enabled=True)
        self.assertAlmostEqual(float(scaled["target_weight"].sum()), 0.675)
        self.assertAlmostEqual(
            float(scaled.iloc[1]["target_weight"] / scaled.iloc[0]["target_weight"]),
            2.0,
        )

    def test_enabled_risk_scaling_requires_complete_dates(self) -> None:
        weights = pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2026-07-31")],
                "stock_code": ["000001.XSHE"],
                "target_weight": [0.90],
            }
        )
        risk = pd.DataFrame(
            {"signal_date": [pd.Timestamp("2026-06-30")], "risk_scale": [0.75]}
        )
        with self.assertRaisesRegex(ValueError, "没有覆盖"):
            apply_risk_scaling(weights, risk, enabled=True)

    def test_small_adjustment_is_skipped_when_exposure_is_unchanged(self) -> None:
        self.assertTrue(
            should_skip_minimum_rebalance(
                target_weight=0.050,
                current_weight=0.046,
                current_shares=500,
                desired_target_shares=600,
                threshold=0.005,
                target_exposure=0.45,
                previous_target_exposure=0.45,
            )
        )

    def test_entry_exit_and_exposure_change_are_never_skipped(self) -> None:
        common = dict(
            target_weight=0.050,
            current_weight=0.046,
            threshold=0.005,
            target_exposure=0.45,
            previous_target_exposure=0.45,
        )
        self.assertFalse(
            should_skip_minimum_rebalance(
                **common, current_shares=0, desired_target_shares=500
            )
        )
        self.assertFalse(
            should_skip_minimum_rebalance(
                **common, current_shares=500, desired_target_shares=0
            )
        )
        changed = dict(common)
        changed["target_exposure"] = 0.90
        self.assertFalse(
            should_skip_minimum_rebalance(
                **changed, current_shares=500, desired_target_shares=600
            )
        )


if __name__ == "__main__":
    unittest.main()
