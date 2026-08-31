from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

MODULE_PATH = Path(__file__).with_name("generate_rebalance_report.py")
SPEC = importlib.util.spec_from_file_location("rebalance_report", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"无法加载调仓报告模块: {MODULE_PATH}")
REPORT_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["rebalance_report"] = REPORT_MODULE
SPEC.loader.exec_module(REPORT_MODULE)
apply_risk_scaling = REPORT_MODULE.apply_risk_scaling
build_actions = REPORT_MODULE.build_actions
format_rank = REPORT_MODULE.format_rank


class RebalanceThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = pd.Timestamp("2026-08-07")
        self.current = pd.Timestamp("2026-08-14")
        self.weights = pd.DataFrame(
            {
                "signal_date": [self.previous, self.current],
                "stock_code": ["000001.XSHE", "000001.XSHE"],
                "industry": ["银行", "银行"],
                "selection_rank": [5, 6],
                "target_weight": [0.050, 0.054],
            }
        )

    @patch(
        "rebalance_report.STOCK_INFO_PATH",
        Path("__missing_stock_info_for_test__.parquet"),
    )
    def test_small_held_change_is_skipped_only_when_exposure_is_unchanged(self) -> None:
        unchanged = build_actions(
            self.weights,
            self.previous,
            self.current,
            minimum_rebalance_weight=0.005,
            exposure_changed=False,
        ).iloc[0]
        self.assertTrue(bool(unchanged["threshold_skipped"]))
        self.assertAlmostEqual(float(unchanged["execution_weight"]), 0.050)
        self.assertAlmostEqual(float(unchanged["weight_change"]), 0.0)

        changed = build_actions(
            self.weights,
            self.previous,
            self.current,
            minimum_rebalance_weight=0.005,
            exposure_changed=True,
        ).iloc[0]
        self.assertFalse(bool(changed["threshold_skipped"]))
        self.assertAlmostEqual(float(changed["execution_weight"]), 0.054)

    def test_missing_rank_uses_actual_candidate_pool_size(self) -> None:
        self.assertEqual(format_rank(float("nan"), 30), "30名外")
        self.assertEqual(format_rank(21, 30), "21")

    def test_candidate_pool_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_stock_count"):
            format_rank(float("nan"), 0)

    def test_rebalance_weights_use_layer_three_risk_scale(self) -> None:
        weights = pd.DataFrame(
            {
                "signal_date": [self.current, self.current],
                "stock_code": ["000001.XSHE", "000002.XSHE"],
                "target_weight": [0.30, 0.60],
            }
        )
        summary = pd.DataFrame(
            {
                "signal_date": [self.current],
                "stock_exposure": [0.90],
                "cash_weight": [0.10],
            }
        )
        risk = pd.DataFrame(
            {
                "signal_date": [self.current],
                "risk_scale": [0.75],
                "scaled_stock_exposure": [0.675],
                "scaled_cash_weight": [0.325],
                "var_budget": [0.15],
            }
        )
        scaled_weights, scaled_summary = apply_risk_scaling(
            weights, summary, risk
        )
        self.assertAlmostEqual(float(scaled_weights["target_weight"].sum()), 0.675)
        self.assertAlmostEqual(
            float(scaled_summary.iloc[0]["stock_exposure"]), 0.675
        )
        self.assertAlmostEqual(float(scaled_summary.iloc[0]["cash_weight"]), 0.325)


if __name__ == "__main__":
    unittest.main()
