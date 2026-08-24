from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from generate_rebalance_report import build_actions, format_rank


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
        "generate_rebalance_report.STOCK_INFO_PATH",
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


if __name__ == "__main__":
    unittest.main()
